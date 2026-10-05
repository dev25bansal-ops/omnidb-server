from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from .tenancy import index_name, validate_tenant_id


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


def rename_shared_sidecar(home: Path) -> list[str]:
    """Move a pre-W3-3 shared sidecar onto the default tenant's name.

    ADR-08 / W3-3. Before this change every tenant's vectors lived in one
    `index.hnsw` / `vectors.npz`. Those are the *default* tenant's vectors — every
    pre-tenant row was written under that id — so the upgrade is a rename, not a
    rebuild: the contents are already correct and moving the file is lossless and
    free, where rebuilding would re-embed the whole home for nothing.

    Returns the names moved, for logging and for tests.

    Deliberately conservative about the collision case. If a per-tenant sidecar
    already exists alongside the legacy one, the home is mid-upgrade or
    hand-edited, and the safe reading is to leave both in place and let
    reconciliation notice the drift — deleting either would lose vectors.
    """
    from .tenancy import DEFAULT_TENANT

    moved: list[str] = []
    # (legacy name, suffix) — the deleted-id tombstone travels with its index.
    for legacy_name, suffix in (
        ("index.hnsw", "hnsw"),
        ("vectors.npz", "npz"),
        ("index.hnsw.deleted.json", "hnsw.deleted.json"),
    ):
        legacy = home / legacy_name
        if not legacy.exists():
            continue
        target_name = index_name(DEFAULT_TENANT, suffix)
        target = home / target_name
        if target.exists():
            # Both present: leave the legacy file alone rather than clobber
            # either side. `check()` reports the divergence.
            continue
        os.replace(legacy, target)
        moved.append(f"{legacy_name} -> {target_name}")
    return moved


def open_vector_index(home: Path, kind: str, dim: int, count_hint: int,
                      ef_search: int = 64):
    """Open the home's vector index.

    ADR-08 / W3-3: this returns a `MultiTenantIndexRouter`, so every tenant's
    vectors live in its own sidecar and a filesystem actor reading the home gets
    one tenant's embeddings rather than all of them.

    `kind` keeps its old meaning — the *backend* preference — and is what
    `meta.vec_kind` round-trips: auto | hnsw | brute, with auto preferring
    hnswlib and falling back to brute. The backend-mismatch guards below are
    evaluated against the legacy names too, because a home that was written
    before this change still has its vectors in `index.hnsw` / `vectors.npz`
    until `rename_shared_sidecar` has run.
    """
    from .tenancy import DEFAULT_TENANT

    # Upgrade first: a legacy home's sidecar must reach its per-tenant name
    # before anything looks for one. Idempotent, and a no-op on a fresh home.
    rename_shared_sidecar(home)

    router_kind = "brute" if kind == "brute" else (
        "hnsw" if kind == "hnsw" else "auto")
    legacy_hnsw = home / "index.hnsw"
    legacy_brute = home / "vectors.npz"
    per_tenant_hnsw = home / index_name(DEFAULT_TENANT, "hnsw")
    per_tenant_brute = home / index_name(DEFAULT_TENANT, "npz")

    if router_kind == "hnsw":
        have_brute = legacy_brute.exists() or per_tenant_brute.exists()
        have_hnsw = legacy_hnsw.exists() or per_tenant_hnsw.exists()
        # Both guards are conditional on the *other* backend existing. A home
        # with no sidecar at all (a fresh `init`, or one whose rows are all
        # unindexed) has nothing to contradict, and must open rather than fail.
        if have_brute and not have_hnsw:
            raise VectorIndexError(
                "this home was built with the brute-force backend; "
                "use OMNIDB_INDEX=brute (or unset) to keep its vectors reachable"
            )
    elif router_kind == "brute":
        have_hnsw = legacy_hnsw.exists() or per_tenant_hnsw.exists()
        have_brute = legacy_brute.exists() or per_tenant_brute.exists()
        if have_hnsw and not have_brute:
            raise VectorIndexError(
                "this home was built with the hnsw backend; "
                "use OMNIDB_INDEX=hnsw (or unset) to keep its vectors reachable"
            )
    return MultiTenantIndexRouter(home, dim, kind=router_kind,
                                  ef_search=ef_search)



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
        """Display view of the backends behind this router.

        Reports "multi" whenever more than one backend kind is in play, or when
        none has been constructed yet. This is a *label*, not a backend name —
        see `configured_kind` for the value that has to survive a round-trip
        through `meta.vec_kind`.
        """
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

    @property
    def configured_kind(self) -> str:
        """The backend name to persist in `meta.vec_kind`.

        `store.py` writes `index.kind` to `vec_kind` on open and `server.py` reads
        it back to pick the backend on the next open, so this must be a name
        `open_vector_index` can construct. `kind` cannot serve that role: it
        reports "multi" whenever no backend exists yet, and backends are lazy, so
        persisting that would make the reopen fall through to `auto` — silently
        moving a home off the backend it was built with.
        """
        with self._lock:
            backends = list(self._backends.values())
            has_hnsw = any(isinstance(b, HnswlibIndex) for b in backends)
            has_brute = any(isinstance(b, BruteForceIndex) for b in backends)
            if has_hnsw and not has_brute:
                return "hnsw"
            if has_brute and not has_hnsw:
                return "brute"
            if has_hnsw and has_brute:
                return "multi"
        preference = self._kind_preference
        if preference in ("hnsw", "brute"):
            return preference
        # "auto" with no backend constructed yet. Backends are lazy, so this is
        # the *normal* state at open time, and it has to name the backend this
        # router will actually build.
        #
        # Deliberately "brute", not hnswlib-availability: the router only
        # constructs hnsw when its preference is hnsw (see
        # `_get_backend_for_tenant`, where "auto" maps to brute). Reporting
        # "hnsw" here wrote a vec_kind the home never used, and the next open
        # then refused it as a backend mismatch — a home that could not reopen
        # itself.
        return "brute"

    def tenants(self) -> list[str]:
        """Every tenant with a backend constructed, sorted for stable output."""
        with self._lock:
            return sorted(self._backends)

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
            # ADR-08 / W3-3: sidecars sit beside omnidb.db as
            # tenant-<id>.{hnsw,npz}, named by tenancy.index_name — the same
            # helper the rename migration uses, so writer and migration cannot
            # disagree about where a tenant's vectors live.
            hnsw_path = self._home / index_name(tenant, "hnsw")
            brute_path = self._home / index_name(tenant, "npz")
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
            except VectorIndexError:
                # Preserve the type. A corrupt sidecar converts to
                # VectorIndexError precisely so `open_read_only_store` and
                # `open_owned_store` can catch it and fall back (audit C6);
                # re-wrapping it in MultiTenantIndexError hid that and turned
                # a recoverable sidecar into a failed open.
                raise
            except Exception as exc:
                raise MultiTenantIndexError(f"Failed to init brute index for tenant {tenant}: {exc}")
            self._backends[tenant] = fresh
            return fresh

    @classmethod
    def single(cls, backend, home: Path) -> "MultiTenantIndexRouter":
        """A router that already holds `backend` as the default tenant's.

        ADR-08 / W3-3. Production builds a router through `open_vector_index`,
        but `MemoryStore` can also be handed a bare backend — by a backup
        restore, or by a unit test exercising one backend's own semantics.
        Wrapping those keeps the store's ~19 index call sites uniform instead of
        making each one handle both shapes, and leaves the bytes where the
        caller put them because the backend is adopted, not rebuilt.
        """
        router = cls(home, getattr(backend, "dim", 0),
                     kind=getattr(backend, "kind", "brute"),
                     ef_search=getattr(backend, "_ef_search", 64))
        router._backends["default"] = backend
        return router

    def backend_for(self, tenant: str) -> "HnswlibIndex | BruteForceIndex":
        """The tenant's backend, constructing it if this is the first request.

        Read-only counterpart to `replace_backend`: callers that need to inspect
        a backend's own attributes (`path`, `_deleted_path`) rather than the
        router's aggregating view use this.
        """
        return self._get_backend_for_tenant(tenant)

    def replace_backend(self, tenant: str, backend) -> None:
        """Swap in a freshly built backend for a tenant.

        Used by `rebuild_index`, which constructs the replacement itself so the
        BUG-A1-12 scratch-then-publish ordering stays in one place. The previous
        backend is dropped only after the new one is live, so a failure during
        the build leaves the old index in place.
        """
        validate_tenant_id(tenant)
        with self._lock:
            self._backends[tenant] = backend

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
