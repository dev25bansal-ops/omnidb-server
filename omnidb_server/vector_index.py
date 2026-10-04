from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from .tenancy import validate_tenant_id


class VectorIndexError(RuntimeError):
    pass


class BruteForceIndex:
    """Exact cosine index over L2-normalized vectors. Always available.

    Vectors live in a capacity-doubling preallocated buffer (PERF-02): add()
    amortizes to O(1) per row instead of vstack-copying the whole matrix on
    every call, which mattered once indexes reach the 10k+ row scale."""

    kind = "brute"

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim
        self._ids_buf = np.empty((0,), dtype=np.int64)
        self._matrix_buf = np.zeros((0, dim), dtype=np.float32)
        self._n = 0

    @classmethod
    def load(cls, path: Path, dim: int) -> "BruteForceIndex":
        index = cls(path, dim)
        if path.exists():
            try:
                with path.open("rb") as handle:
                    data = np.load(handle)
                    matrix = data["matrix"].astype(np.float32)
                    if matrix.shape[1] != dim:
                        raise VectorIndexError(
                            f"index dim {matrix.shape[1]} != configured dim {dim}"
                        )
                    index._ids_buf = data["ids"].astype(np.int64)
                    index._matrix_buf = np.ascontiguousarray(matrix)
                    index._n = len(index._ids_buf)
            except VectorIndexError:
                # A dim mismatch is a configuration fault with a specific remedy
                # (re-embed at the configured dim), so it keeps its own message.
                raise
            except (KeyError, ValueError, OSError, EOFError,
                    zipfile.BadZipFile) as exc:
                # Mirrors HnswlibIndex's conversion above. Without it, a corrupt
                # or foreign .npz escapes as a raw KeyError/ValueError on both
                # open paths: `open_read_only_store` catches only
                # VectorIndexError, so the operator got a numpy traceback
                # instead of the fallback the hnsw path already performs
                # (audit C6).
                raise VectorIndexError(
                    f"brute sidecar unusable at {path.name} ({exc})") from exc
        return index

    @property
    def count(self) -> int:
        return int(self._n)

    def indexed_ids(self) -> set[int]:
        return {int(x) for x in self._ids_buf[: self._n]}

    def active_count(self) -> int:
        return int(self._n)

    @property
    def dead_ratio(self) -> float:
        return 0.0  # brute removals are physical; nothing accumulates

    def add(self, ids, vecs) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        total = self._n + len(ids)
        if len(self._ids_buf) < total:
            cap = max(total, 2 * len(self._ids_buf), 64)
            grown_ids = np.empty(cap, dtype=np.int64)
            grown_ids[: self._n] = self._ids_buf[: self._n]
            grown_matrix = np.zeros((cap, self.dim), dtype=np.float32)
            grown_matrix[: self._n] = self._matrix_buf[: self._n]
            self._ids_buf = grown_ids
            self._matrix_buf = grown_matrix
        self._ids_buf[self._n : total] = ids
        self._matrix_buf[self._n : total] = vecs
        self._n = total

    def update(self, mid: int, vec) -> None:
        vec = np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1)
        pos = np.nonzero(self._ids_buf[: self._n] == int(mid))[0]
        if len(pos):
            self._matrix_buf[pos[0]] = vec[0]
        else:
            self.add([int(mid)], [vec[0]])

    def remove(self, ids) -> None:
        drop = {int(i) for i in ids}
        if self._n == 0 or not drop:
            return
        keep = np.fromiter(
            (int(i) not in drop for i in self._ids_buf[: self._n]),
            dtype=bool,
            count=self._n,
        )
        kept = int(keep.sum())
        # compact in place: surviving rows move to the front of the buffers
        self._ids_buf[:kept] = self._ids_buf[: self._n][keep]
        self._matrix_buf[:kept] = self._matrix_buf[: self._n][keep]
        self._n = kept

    def query(self, vec, k: int) -> list[tuple[int, float]]:
        if self._n == 0:
            return []
        sims = self._matrix_buf[: self._n] @ vec
        n = min(k, len(sims))
        if len(sims) > n:
            top = np.argpartition(-sims, n - 1)[:n]
        else:
            top = np.argsort(-sims)
        top = top[np.argsort(-sims[top])]
        return [(int(self._ids_buf[i]), float(sims[i])) for i in top]

    def persist(self) -> None:
        """Write a full O(N) snapshot for fixed dimension, not a delta.

        AUDIT-I18: the store's operation-count threshold deliberately overrides
        adaptive elapsed-time scheduling to bound pending recovery work. Each
        snapshot writes about N * (4 * dim + 8) raw array bytes; frequent count
        triggers therefore do not promise bounded amortized persistence cost.
        """
        # Do not debounce here: successful return lets the store clear dirty state.
        _atomic_write(self.path, lambda handle: np.savez(
            handle, ids=self._ids_buf[: self._n], matrix=self._matrix_buf[: self._n]
        ))


class HnswlibIndex:
    """HNSW index (canonical hnswlib bindings), inner-product space on
    L2-normalized vectors == cosine. Labels are memory ids; deletions are
    mark-deleted (excluded from results) with compaction deferred."""

    kind = "hnsw"

    def __init__(self, path: Path, dim: int, count_hint: int = 0,
                 m: int = 16, ef_construction: int = 200, ef_search: int = 64):
        try:
            import hnswlib
        except ImportError as exc:
            raise VectorIndexError("hnswlib is not installed") from exc
        self.path = path
        self.dim = dim
        # hnswlib's element_count includes mark-deleted points, so track them
        # ourselves to know the true active count (a query against a fully
        # deleted index would otherwise fail to fill k results).
        self._deleted_path = path.with_name(path.name + ".deleted.json")
        self.cap = max(64, int(count_hint) * 2)
        # F31: a corrupt sidecar raises RuntimeError from hnswlib (a C++
        # std::runtime_error) and a corrupt tombstone sidecar raises ValueError from
        # json.loads. Neither is a VectorIndexError, so after the F13 narrowing they
        # propagated and crashed the open instead of falling back - the opposite of
        # ARCH-04. Convert both, so "this index cannot be used" always arrives as
        # VectorIndexError and the fallback keeps working.
        try:
            self.index = hnswlib.Index(space="ip", dim=dim)
            if path.exists():
                self.index.load_index(str(path), max_elements=self.cap)
                if self._deleted_path.exists():
                    self._deleted = {
                        int(x) for x in json.loads(self._deleted_path.read_text())}
                else:
                    self._deleted = set()
            else:
                self.index.init_index(
                    max_elements=self.cap, ef_construction=ef_construction, M=m)
                self._deleted = set()
            self.index.set_ef(max(ef_search, 16))
        except VectorIndexError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise VectorIndexError(
                f"hnsw sidecar unusable at {path.name} ({exc})") from exc

    @property
    def count(self) -> int:
        return int(self.index.element_count)

    def indexed_ids(self) -> set[int]:
        return {int(x) for x in self.index.get_ids_list()} - self._deleted

    def active_count(self) -> int:
        return max(self.count - len(self._deleted), 0)

    @property
    def dead_ratio(self) -> float:
        total = self.count
        return len(self._deleted) / total if total else 0.0

    def _ensure_capacity(self, incoming: int) -> None:
        """Grow the hnswlib index so `incoming` more items fit.

        BUG-A1-11: only ``add`` grew the index, so a workload that grew the
        corpus purely through ``update`` hit hnswlib's capacity error. Both
        entry points share this now.
        """
        needed = self.count + incoming
        if needed > self.cap:
            self.cap = max(needed * 2, self.cap * 2)
            self.index.resize_index(self.cap)

    def update(self, mid: int, vec) -> None:
        # this hnswlib build treats a re-add of an existing label as an update
        mid = int(mid)
        # A label already present replaces its vector and does NOT grow the
        # index; only a genuinely new label needs capacity. Reserving
        # unconditionally would double the graph on every content revision,
        # which is the common case for an update-heavy workload.
        if mid not in self.index:
            self._ensure_capacity(1)
        vec = np.ascontiguousarray([vec], dtype=np.float32)
        self.index.add_items(vec, [mid], num_threads=1)

    def add(self, ids, vecs) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self._ensure_capacity(len(ids))
        self.index.add_items(vecs, [int(i) for i in ids], num_threads=1)

    def remove(self, ids) -> None:
        for mid in ids:
            try:
                self.index.mark_deleted(int(mid))
                self._deleted.add(int(mid))
            except RuntimeError:
                pass

    def query(self, vec, k: int) -> list[tuple[int, float]]:
        active = self.active_count()
        if active == 0:
            return []
        k_eff = min(k, active)
        labels, dists = self.index.knn_query(
            np.asarray([vec], dtype=np.float32), k=k_eff
        )
        return [
            (int(label), float(1.0 - dist))
            for label, dist in zip(labels[0], dists[0])
            if int(label) not in self._deleted
        ]

    def persist(self) -> None:
        """Write the full graph and tombstones, not only pending mutations.

        AUDIT-I18: snapshot cost is O(N) for fixed dimension and graph degree.
        The store's count threshold intentionally trades this cost for less
        post-crash reconstruction, even when its adaptive time trigger waits.
        """
        # Do not debounce here: successful return lets the store clear dirty state.
        # hnswlib writes via its own FILE*; do not hand it an open Python fd.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".hnsw.tmp")
        os.close(fd)
        fd_del, tmp_del = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".deleted.json.tmp"
        )
        try:
            with os.fdopen(fd_del, "w", encoding="utf-8") as handle:
                json.dump(sorted(self._deleted), handle)
            self.index.save_index(tmp_name)
            _replace_with_retry(tmp_name, str(self.path))
            _replace_with_retry(tmp_del, str(self._deleted_path))
        finally:
            for leftover in (tmp_name, tmp_del):
                if os.path.exists(leftover):
                    os.unlink(leftover)


def _replace_with_retry(src: str, dst: str, attempts: int = 6) -> None:
    # Windows: AV/indexers hold files briefly; os.replace raises WinError 5
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def open_vector_index(home: Path, kind: str, dim: int, count_hint: int,
                      ef_search: int = 64):
    """kind: auto | hnsw | brute. auto prefers hnswlib, falls back to brute."""
    hnsw_path = home / "index.hnsw"
    brute_path = home / "vectors.npz"
    if kind == "hnsw":
        if not hnsw_path.exists() and brute_path.exists():
            raise VectorIndexError(
                "this home was built with the brute-force backend; "
                "use OMNIDB_INDEX=brute (or unset) to keep its vectors reachable"
            )
        return HnswlibIndex(hnsw_path, dim, count_hint=count_hint,
                            ef_search=ef_search)
    if kind == "brute":
        if not brute_path.exists() and hnsw_path.exists():
            raise VectorIndexError(
                "this home was built with the hnsw backend; "
                "use OMNIDB_INDEX=hnsw (or unset) to keep its vectors reachable"
            )
        return BruteForceIndex.load(brute_path, dim)
    if hnsw_path.exists():
        try:
            return HnswlibIndex(hnsw_path, dim, count_hint=count_hint,
                                ef_search=ef_search)
        except VectorIndexError:
            pass  # hnswlib not installed: expected offline fallback
        except Exception as exc:
            # ARCH-04: a corrupt/foreign sidecar silently downgrading is a
            # support nightmare — say why on stderr before falling back
            print(
                f"omnidb-server: hnsw sidecar {hnsw_path.name} failed to open "
                f"({exc}); falling back to the brute-force backend",
                file=sys.stderr,
            )
    if brute_path.exists():
        return BruteForceIndex.load(brute_path, dim)
    try:
        return HnswlibIndex(hnsw_path, dim, count_hint=count_hint,
                            ef_search=ef_search)
    except VectorIndexError:
        return BruteForceIndex(brute_path, dim)
    except Exception as exc:
        print(
            f"omnidb-server: hnsw sidecar {hnsw_path.name} failed to open "
            f"({exc}); falling back to the brute-force backend",
            file=sys.stderr,
        )
        return BruteForceIndex(brute_path, dim)


# ======================================================
# MT-06 Phase 1: Per-Tenant Index Router (standalone, tested in isolation before integration)
# ======================================================

class MultiTenantIndexError(Exception):
    """Raised when an invalid operation is attempted on the MultiTenantIndexRouter."""
    pass


class MultiTenantIndexRouter:
    """Thread-safe per-tenant vector index router implementing the HnswlibIndex/BruteForceIndex protocol."""
    
    __slots__ = ("_home", "_dim", "_backends", "_lock", "_kind_preference", "_ef_search")

    def __init__(self, home: Path, dim: int, kind: str | None = "brute", ef_search: int = 64):
        self._home = home.resolve()
        self._dim = dim
        self._kind_preference = kind if kind else "brute"
        self._ef_search = max(ef_search, 16)
        self._backends: dict[str, "HnswlibIndex | BruteForceIndex"] = {}
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Root path shared by all tenant indices."""
        return self._home

    @property
    def dim(self) -> int:
        """Embedding dimension shared across all tenants."""
        return self._dim

    @property
    def kind(self) -> str:
        """Report a single view of backend."""
        with self._lock:
            backends = list(self._backends.values())
            has_hnsw = any(isinstance(b, HnswlibIndex) for b in backends)
            has_brute = any(isinstance(b, BruteForceIndex) for b in backends)
            if not backends:
                return "multi"
            if has_hnsw and not has_brute:
                return "hnsw"
            if has_brute and not has_hnsw:
                return "brute"
            return "multi"

    def _get_backend_for_tenant(self, tenant: str) -> "HnswlibIndex | BruteForceIndex":
        """Get or lazily construct a tenant's backend index.

        The tenant id reaches a filesystem path here, so it is validated before
        the join — audit finding C4. `home/tenants/<tenant>` plus an unvalidated
        id meant `get_or_create("../../escaped")` created a directory *beside*
        the home. `tenancy.index_name` and `tenant_schema` both validate first;
        the router is the third place a tenant id becomes a filename and it was
        the one that did not.
        """
        validate_tenant_id(tenant)
        with self._lock:
            if tenant in self._backends:
                return self._backends[tenant]
            backend_kind = self._kind_preference if self._kind_preference != "auto" else "brute"
            idx_path_home = self._home / "tenants" / tenant
            idx_path_home.mkdir(parents=True, exist_ok=True)
            hnsw_path = idx_path_home / "index.hnsw"
            brute_path = idx_path_home / "vectors.npz"
            fresh: "HnswlibIndex | BruteForceIndex"
            if backend_kind == "hnsw":
                try:
                    fresh = HnswlibIndex(hnsw_path, self._dim, count_hint=64 * 2, ef_search=self._ef_search)
                    self._backends[tenant] = fresh
                    return fresh
                except Exception:
                    pass
            try:
                fresh = BruteForceIndex.load(brute_path, self._dim) if brute_path.exists() else BruteForceIndex(brute_path, self._dim)
            except Exception as exc:
                raise MultiTenantIndexError(f"Failed to init brute index for tenant {tenant}: {exc}")
            self._backends[tenant] = fresh
            return fresh

    def add(self, ids, vecs, tenant: str) -> None:
        backend = self._get_backend_for_tenant(tenant)
        backend.add(ids, vecs)

    def update(self, mid: int, vec, tenant: str) -> None:
        backend = self._get_backend_for_tenant(tenant)
        backend.update(mid, vec)

    def remove(self, ids, tenant: str) -> None:
        backend = self._get_backend_for_tenant(tenant)
        backend.remove(ids)

    def query(self, vec, k: int, tenant: str) -> list[tuple[int, float]]:
        backend = self._get_backend_for_tenant(tenant)
        return backend.query(vec, k)

    def active_count(self, tenant: str | None = None) -> int:
        with self._lock:
            if tenant:
                return self._get_backend_for_tenant(tenant).active_count()
            return sum(idx.active_count() for idx in self._backends.values())

    def indexed_ids(self, tenant: str | None = None) -> set[int]:
        with self._lock:
            if tenant:
                return self._get_backend_for_tenant(tenant).indexed_ids()
            return {mid for idx in self._backends.values() for mid in idx.indexed_ids()}

    def persist(self, tenant: str | None = None) -> None:
        with self._lock:
            if tenant:
                self._get_backend_for_tenant(tenant).persist()
            else:
                for idx in list(self._backends.values()):
                    idx.persist()

    @property
    def dead_ratio(self) -> float:
        with self._lock:
            total_active = sum(idx.active_count() for idx in self._backends.values())
            if total_active == 0:
                return 0.0
            total_dead_weight = sum(getattr(idx, "dead_ratio", 0.0) * idx.count for idx in self._backends.values())
            total_elements = sum(idx.count for idx in self._backends.values())
            return total_dead_weight / total_elements if total_elements > 0 else 0.0

    def flush_all(self) -> None:
        self.persist(None)

    def close(self, tenant: str | None = None) -> None:
        with self._lock:
            if tenant:
                self._backends.pop(tenant, None)
            else:
                self._backends.clear()

    @property
    def backend_count(self) -> int:
        """Number of tenant backends currently loaded."""
        with self._lock:
            return len(self._backends)

    def get_or_create(self, tenant: str) -> "HnswlibIndex | BruteForceIndex":
        return self._get_backend_for_tenant(tenant)

    def __repr__(self) -> str:
        return f"<MultiTenantIndexRouter home={self._home} dim={self._dim} backends={len(self._backends)}>"


def make_multi_tenant_router(home: Path, dim: int = 512, backend: str = "brute", ef_search: int = 64) -> MultiTenantIndexRouter:
    """Helper for tests: build a router instance for direct unit testing."""
    return MultiTenantIndexRouter(home, dim=dim, kind=backend, ef_search=ef_search)
