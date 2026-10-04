from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .db import get_meta, set_meta
from .store import E_CONSTRAINT, E_NOT_FOUND, E_TYPE, StoreError, _commit_or_rollback
from .validation import validate_content, validate_trigger_patterns
from .tenancy import DEFAULT_TENANT

_KINDS = ("episodic", "semantic", "procedural")

# sidecar files copied alongside a backup snapshot / handled by re-embed
_SIDECAR_NAMES = ("vectors.npz", "index.hnsw", "index.hnsw.deleted.json")


def backup_home(store, dest: str | Path, keep: int = 3) -> dict:
    """Hot physical backup (E-13): a crash-consistent database snapshot via
    SQLite's online backup API, plus the vector sidecar files, written into
    `dest/snapshot-<UTC timestamp>/`. The server can keep writing during the
    snapshot.

    AUDIT-I03: the sidecar copy happens outside the DB snapshot's consistent
    cut and may be torn or belong to a different generation, so the snapshot
    always carries `index_dirty=1`. Reconciliation-on-open treats the rows as
    source of truth and re-embeds when the copy disagrees, making the restore
    internally consistent regardless of sidecar freshness. `keep` rotates the
    oldest snapshot directories.
    """
    t0 = time.perf_counter()
    dest = Path(dest)
    # microsecond stamp: fixed-width, sortable, collision-free for rapid runs
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = dest / f"snapshot-{stamp}"
    target.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(target), suffix=".db.tmp")
    os.close(fd)
    os.unlink(tmp_name)  # sqlite connect creates its own file
    try:
        dst = sqlite3.connect(tmp_name)
        try:
            # hold the store lock so an in-process writer cannot interleave;
            # cross-process writers are safe (WAL readers don't block them)
            with store.locked():
                store.conn.backup(dst)
            # AUDIT-I03: mark the snapshot inside the locked consistent cut:
            # a restore reconciles vectors from rows instead of trusting a
            # sidecar copy that may predate or postdate this snapshot.
            dst.execute(
                "INSERT INTO meta(key, value) VALUES ('index_dirty', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            dst.commit()
        finally:
            dst.close()
        os.replace(tmp_name, target / "omnidb.db")
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    copied = []
    for name in _SIDECAR_NAMES:
        src = store._home() / name
        if src.is_file():
            shutil.copy2(src, target / name)
            copied.append(name)

    removed, prune_errors = _rotate_snapshots(dest, keep)
    for error in prune_errors:
        print(f"omnidb-server: could not prune snapshot: {error}", file=sys.stderr)
    return {"dest": str(target), "db_bytes": (target / "omnidb.db").stat().st_size,
            "sidecars": copied, "pruned": removed, "prune_errors": prune_errors,
            "took_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _rotate_snapshots(dest: Path, keep: int) -> tuple[list[str], list[str]]:
    if keep < 1:
        keep = 1
    snapshots = sorted(
        p for p in dest.iterdir()
        if p.is_dir() and p.name.startswith("snapshot-")
    )
    removed = []
    errors = []
    for old in snapshots[:-keep] if len(snapshots) > keep else []:
        try:
            shutil.rmtree(old)
        except OSError as exc:
            errors.append(f"{old.name}: {exc}")
        else:
            removed.append(old.name)
    return removed, errors


def _restore_rebuild_artifacts(backups: list[tuple[Path, Path]]) -> None:
    for source, target in reversed(backups):
        if target.exists():
            target.unlink(missing_ok=True)
        os.replace(source, target)


def _remove_rebuild_artifacts(backups: list[tuple[Path, Path]]) -> None:
    for source, _ in reversed(backups):
        source.unlink(missing_ok=True)


def _embedder_from_stored_name(name: str):
    """Rebuild the embedder recorded in `meta.embedder` for a migration."""
    from .embeddings import embedder_from_name

    try:
        return embedder_from_name(name)
    except StoreError as exc:
        raise StoreError(
            f"this home was built with embedder {name!r}, which this version "
            "cannot load; re-embedding is not possible",
            exc.code,
        ) from exc


def reembed_home(home: Path, target_embedder, *, assume_yes: bool = False) -> dict:
    """Migrate every live memory to a new embedder in place (E-01).

    Opens the home with its stored embedder, records a durable
    `reembed_pending` marker, re-encodes all live rows with `target_embedder`
    into a fresh sidecar, then publishes the new identity and clears the
    marker in one transaction (AUDIT-I03). A crash at any boundary leaves
    either the marker set (next open rebuilds vectors from the rows under
    the stored identity — correct even for a same-dimension model swap) or
    the generation fully published. Rows are the source of truth, so every
    crash window heals.
    """
    from . import db as dbmod
    from .config import Config
    from .server import _acquire_home_lock, _release_home_lock
    from .store import _SELECTIVE, MemoryStore
    from .vector_index import BruteForceIndex, HnswlibIndex, VectorIndexError, open_vector_index

    def stage_rebuild(path: Path) -> Path | None:
        if not path.exists():
            return None
        tmp = path.with_name(f".{path.name}.rebuild-{os.getpid()}")
        try:
            os.replace(path, tmp)
        except OSError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        return tmp

    t0 = time.perf_counter()
    cfg = Config.load(home)
    # AUDIT-I02: the home lock is acquired BEFORE dbmod.connect — connecting
    # runs migrations and WAL pragmas, i.e. writable initialization. A
    # rejected opener must touch nothing.
    lock_fd = _acquire_home_lock(cfg.home)
    if isinstance(lock_fd, OSError):
        raise StoreError(
            f"another process is using {cfg.home}; stop it before re-embedding",
            E_CONSTRAINT,
        )
    conn = None
    try:
        conn = dbmod.connect(cfg.home)
        stored_name = get_meta(conn, "embedder")
        if not stored_name:
            raise StoreError(
                "this home has no embedder identity yet (nothing to migrate); "
                "open it normally first", E_CONSTRAINT,
            )
        if stored_name == target_embedder.name:
            raise StoreError(
                f"home is already built with {stored_name!r}; nothing to migrate",
                E_CONSTRAINT,
            )

        emb_src = _embedder_from_stored_name(stored_name)
        if emb_src.name != stored_name:
            raise StoreError(
                f"stored embedder {stored_name!r} != loadable {emb_src.name!r}",
                E_CONSTRAINT,
            )

        count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        stored_kind = get_meta(conn, "vec_kind") or "auto"
        index_kind = stored_kind if stored_kind in ("hnsw", "brute") else "auto"
        idx = open_vector_index(cfg.home, index_kind, emb_src.dim, count,
                                ef_search=cfg.ef_search)
        store = MemoryStore(conn, emb_src, idx, cfg)
        conn = None  # store.close() below owns the connection from here on
        try:
            rows = store.conn.execute(
                f"SELECT id, content FROM memories WHERE {_SELECTIVE} ORDER BY id"
            ).fetchall()

            if not assume_yes and sys.stdin is not None and sys.stdin.isatty():
                answer = input(
                    f"Re-embed {len(rows)} memories from {stored_name!r} to "
                    f"{target_embedder.name!r} (dim {emb_src.dim} -> "
                    f"{target_embedder.dim})? [y/N] "
                )
                if answer.strip().lower() not in ("y", "yes"):
                    raise StoreError("re-embed cancelled", E_CONSTRAINT)
            elif not assume_yes:
                raise StoreError(
                    "refusing to re-embed non-interactively without --yes",
                    E_CONSTRAINT,
                )

            # AUDIT-I03: durable migration marker BEFORE the old sidecar is
            # replaced. Any interruption — including a same-dimension model
            # swap a dim check cannot detect — leaves the marker set, and the
            # next open rebuilds vectors from rows under the stored identity
            # instead of trusting a half-published generation.
            set_meta(store.conn, "reembed_pending", json.dumps({
                "embedder": target_embedder.name,
                "dim": target_embedder.dim,
            }))
            home_dir = store._home()
            fresh: HnswlibIndex | BruteForceIndex
            backup_paths: list[tuple[Path, Path]] = []
            if index_kind == "hnsw":
                hnsw_path = home_dir / "index.hnsw"
                deleted_path = home_dir / "index.hnsw.deleted.json"
                for path in (hnsw_path, deleted_path):
                    staged = stage_rebuild(path)
                    if staged is not None:
                        backup_paths.append((staged, path))
                try:
                    fresh = HnswlibIndex(hnsw_path, target_embedder.dim,
                                         count_hint=len(rows),
                                         ef_search=cfg.ef_search)
                except VectorIndexError:
                    _restore_rebuild_artifacts(backup_paths)
                    backup_paths = []
                    print(
                        "omnidb-server: hnswlib unavailable; the migrated "
                        "index will use the brute-force backend",
                        file=sys.stderr,
                    )
                    fresh = BruteForceIndex(home_dir / "vectors.npz",
                                            target_embedder.dim)
            else:
                path = home_dir / "vectors.npz"
                staged = stage_rebuild(path)
                if staged is not None:
                    backup_paths.append((staged, path))
                fresh = BruteForceIndex(path, target_embedder.dim)

            # dirty flag first (ARCH-02 discipline): a crash mid-loop leaves
            # index_dirty=1 so the next open reconciles
            store._mark_dirty()
            chunk = 256
            for start in range(0, len(rows), chunk):
                part = rows[start:start + chunk]
                vectors = target_embedder.encode([r["content"] for r in part])
                fresh.add([int(r["id"]) for r in part], vectors)
                print(
                    f"omnidb-server: re-embedded "
                    f"{min(start + chunk, len(rows))}/{len(rows)}",
                    file=sys.stderr, flush=True,
                )
            store.index = fresh
            store._persist_index()
            # AUDIT-I03: publish the identity and clear the migration marker
            # in ONE transaction. The home is never openable with the new
            # sidecar under the old identity (or the reverse); a crash here
            # leaves the marker set, so the next open rebuilds from rows.
            #
            # BUG-A1-04: the staged backups used to be removed *before* this
            # transaction, so a failure at the last fallible step left a home
            # whose vectors might not match its stored rows and no rollback
            # copy. Cleanup now runs only after the commit succeeds.
            try:
                store.conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('embedder', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (target_embedder.name,))
                store.conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('vec_dim', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(target_embedder.dim),))
                store.conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('vec_kind', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (fresh.kind,))
                store.conn.execute("DELETE FROM meta WHERE key = 'reembed_pending'")
                store.conn.commit()
            except sqlite3.Error as exc:
                raise _commit_or_rollback(store.conn, exc) from exc
            _remove_rebuild_artifacts(backup_paths)
            return {
                "migrated": len(rows),
                "from": stored_name,
                "to": target_embedder.name,
                "dim": target_embedder.dim,
                "index_kind": fresh.kind,
                "took_s": round(time.perf_counter() - t0, 1),
            }
        finally:
            store.close()
    finally:
        # AUDIT-I20: close the connection on every path that failed before
        # the store took ownership, then release the lock in its own scope.
        if conn is not None:
            conn.close()
        _release_home_lock(lock_fd)


def _atomic_write_text(path: Path, writer) -> None:
    """Write the file atomically: write to a sibling temp, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _build_memory_where(filters: dict) -> tuple[str, list]:
    """Translate export filters into a WHERE clause and bound params.

    Returns a clause plus a small helper that rewrites it for use in a JOIN
    (by aliasing each column reference to a specific edge endpoint alias).
    """
    clauses = ["deleted_ms IS NULL", "superseded_by IS NULL", "tenant_id = ?"]
    params: list = [filters["tenant"]]
    if filters.get("since_ms") is not None:
        clauses.append("created_ms >= ?")
        params.append(int(filters["since_ms"]))
    if filters.get("agent_id") is not None:
        clauses.append("agent_id = ?")
        params.append(filters["agent_id"])
    if filters.get("kind") is not None:
        if filters["kind"] not in _KINDS:
            raise StoreError(
                f"unknown kind {filters['kind']!r}; expected one of {_KINDS}", E_TYPE
            )
        clauses.append("kind = ?")
        params.append(filters["kind"])
    return " AND ".join(clauses), params


def _aliased_where(clause: str, alias: str) -> str:
    """Prefix the bare column names that appear in a memory WHERE clause
    with an alias so it can be used in an edge JOIN."""
    out = clause
    for column in ("deleted_ms", "superseded_by", "created_ms", "agent_id",
                   "tenant_id", "kind"):
        out = out.replace(column, f"{alias}.{column}")
    return out


def export_jsonl(store, path: str | Path, *,
                 since_ms: int | None = None,
                 agent_id: str | None = None,
                   kind: str | None = None,
                   tenant: str = DEFAULT_TENANT) -> dict:
    """Dump live memories then their edges as JSONL, one record per line.

    Atomic: writes to a sibling temp, fsyncs, renames — a crash mid-write
    leaves either the old file intact or the new file fully written.

    Filters: since_ms, agent_id, kind. Soft-deleted and superseded rows are
    always skipped; edges are only written when both endpoints match the
    memory filter (so an import can satisfy the FK constraints).
    """
    path = Path(path)
    filters = {"since_ms": since_ms, "agent_id": agent_id, "kind": kind,
               "tenant": tenant}
    mem_where, mem_params = _build_memory_where(filters)
    memories = 0
    edges = 0

    def _write(handle):
        nonlocal memories, edges
        for row in store.conn.execute(
            "SELECT id, kind, content, metadata, created_ms, access_count, "
            "agent_id, trigger_patterns FROM memories "
            f"WHERE {mem_where} ORDER BY id",
            mem_params,
        ):
            handle.write(json.dumps({
                "type": "memory",
                "id": int(row["id"]),
                "kind": row["kind"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_ms": int(row["created_ms"]),
                "access_count": int(row["access_count"]),
                "agent_id": row["agent_id"],
                "trigger_patterns": (
                    json.loads(row["trigger_patterns"])
                    if row["trigger_patterns"] is not None else None
                ),
            }) + "\n")
            memories += 1
        for row in store.conn.execute(
            "SELECT e.src, e.dst, e.label FROM edges e "
            f"JOIN memories s ON s.id = e.src AND {_aliased_where(mem_where, 's')} "
            f"JOIN memories d ON d.id = e.dst AND {_aliased_where(mem_where, 'd')} "
            "ORDER BY e.src, e.dst, e.label",
            mem_params + mem_params,
        ):
            handle.write(json.dumps({
                "type": "edge",
                "src": int(row["src"]),
                "dst": int(row["dst"]),
                "label": row["label"],
            }) + "\n")
            edges += 1

    _atomic_write_text(path, _write)
    return {"memories": memories, "edges": edges, "path": str(path),
            "filters": filters}


def import_jsonl(store, path: str | Path, *,
                  tenant: str = DEFAULT_TENANT) -> dict:
    """Restore an export into an EMPTY home.

    AUDIT-I04: original memory ids are preserved explicitly, every record
    and edge endpoint is validated BEFORE the first write, and the whole
    import runs in one transaction that rolls back completely on failure,
    so a rejected import leaves the home unchanged.
    """
    path = Path(path)
    if store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]:
        raise StoreError(
            "refusing import into a non-empty home (explicit ids would collide); "
            "point --home at an empty directory",
            E_CONSTRAINT,
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StoreError(f"cannot read export file: {exc}", E_NOT_FOUND) from exc

    memories: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[int] = set()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError as exc:
            raise StoreError(f"line {lineno}: invalid JSON ({exc})", E_TYPE) from exc
        kind = record.get("type") if isinstance(record, dict) else None
        if kind == "memory":
            if not isinstance(record.get("content"), str) or "id" not in record:
                raise StoreError(
                    f"line {lineno}: memory record needs an id and string content", E_TYPE
                )
            if record.get("kind", "semantic") not in _KINDS:
                raise StoreError(f"line {lineno}: unknown kind {record.get('kind')!r}", E_TYPE)
            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise StoreError(f"line {lineno}: metadata must be an object", E_TYPE)
            patterns = record.get("trigger_patterns")
            if patterns is not None and not isinstance(patterns, list):
                raise StoreError(f"line {lineno}: trigger_patterns must be a list", E_TYPE)
            try:
                mid = int(record["id"])
            except (TypeError, ValueError) as exc:
                raise StoreError(f"line {lineno}: memory id must be an integer", E_TYPE) from exc
            if mid in seen_ids:
                raise StoreError(f"line {lineno}: duplicate memory id {mid}", E_CONSTRAINT)
            # AUDIT-I21: the same content/pattern policy as creation, applied
            # before any database or index mutation.
            try:
                validate_content(record["content"], store.cfg.max_content_chars)
                if patterns is not None:
                    validate_trigger_patterns(patterns)
            except TypeError as exc:
                raise StoreError(f"line {lineno}: {exc}", E_TYPE) from exc
            except ValueError as exc:
                raise StoreError(f"line {lineno}: {exc}", E_CONSTRAINT) from exc
            seen_ids.add(mid)
            memories.append(record)
        elif kind == "edge":
            if not all(field in record for field in ("src", "dst", "label")):
                raise StoreError(f"line {lineno}: edge record needs src, dst and label", E_TYPE)
            try:
                src = int(record["src"])
                dst = int(record["dst"])
            except (TypeError, ValueError) as exc:
                raise StoreError(f"line {lineno}: edge endpoints must be integers", E_TYPE) from exc
            # AUDIT-I04: unresolved endpoints are a hard error, not a silent
            # skip — the report must reflect what was actually applied.
            for endpoint in (src, dst):
                if endpoint not in seen_ids:
                    raise StoreError(
                        f"line {lineno}: edge references unknown memory id {endpoint}",
                        E_CONSTRAINT,
                    )
            edges.append(record)
        else:
            raise StoreError(
                f'line {lineno}: record must carry "type": "memory" or "edge"', E_TYPE
            )

    if not memories:
        return {"memories": 0, "edges": 0, "path": str(path)}

    # Encode before any mutation so a failed import leaves the home untouched.
    contents = [r["content"] for r in memories]
    vectors = store.embedder.encode(contents)
    new_ids = [int(r["id"]) for r in memories]
    now_ms = int(store.now_fn() * 1000)

    try:
        for record in memories:
            patterns = record.get("trigger_patterns")
            mid = int(record["id"])
            content = record["content"]
            store.conn.execute(
                "INSERT INTO memories(id, kind, content, metadata, created_ms, "
                "access_count, agent_id, trigger_patterns, tenant_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mid,
                    record.get("kind", "semantic"),
                    content,
                    json.dumps(record.get("metadata") or {}),
                    int(record.get("created_ms") or now_ms),
                    int(record.get("access_count") or 0),
                    record.get("agent_id"),
                    json.dumps(patterns) if patterns is not None else None,
                        tenant,
                ),
            )
            # ADR-08 / W3-3: import writes memories directly, so it must sync
            # FTS itself — the triggers that used to do this are gone.
            store.conn.execute(
                "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                (mid, content),
            )
        applied_edges = 0
        for record in edges:
            cur = store.conn.execute(
                "INSERT OR IGNORE INTO edges(src, dst, label, created_ms) "
                "VALUES (?, ?, ?, ?)",
                (int(record["src"]), int(record["dst"]),
                 str(record["label"]), now_ms),
            )
            applied_edges += max(int(cur.rowcount), 0)
        store.conn.commit()
    except sqlite3.Error as exc:
        # AUDIT-I04: any sqlite failure rolls the whole import back so the
        # next unrelated commit cannot persist partial rows.
        raise _commit_or_rollback(store.conn, exc) from exc
    store._after_commit(lambda: store.index.add(new_ids, vectors), len(new_ids))
    return {"memories": len(memories), "edges": applied_edges, "path": str(path)}
