from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .config import Config
from .db import SCHEMA_VERSION, get_meta, set_meta
from .embeddings import Embedder
from .tenancy import DEFAULT_TENANT, validate_tenant_id
from .validation import validate_content, validate_trigger_patterns

if TYPE_CHECKING:
    from .vector_index import BruteForceIndex, HnswlibIndex

_KINDS = ("episodic", "semantic", "procedural")
_SELECTIVE = "deleted_ms IS NULL AND superseded_by IS NULL"
_SQL_FIRST_WORD = re.compile(r"^\s*([a-zA-Z]+)")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# AUDIT-I05: wall-clock ceiling for one `search` execution and the byte cap
# on materialized result cells (str cells measured in characters; other
# cells charged a fixed 8-byte approximation).
_QUERY_WALL_CLOCK_LIMIT_S = 5.0
_QUERY_RESULT_BYTES_LIMIT = 32 * 1024 * 1024

# AUDIT-I13: ceiling on the ids fetched/encoded per recovery batch; the
# effective batch is min(this, the connection's variable limit - 2).
_RECONCILE_BATCH_CAP = 500

# BUG-A1-08: one batching policy for every `id IN (...)` site. The cap bounds
# the work per statement; the reserve covers parameters a caller binds *beside*
# the id list (remember_many binds a timestamp, forget_many binds one
# deleted_ms, reconcile binds none).
_ID_BATCH_CAP = 500
_ID_BIND_RESERVE = 4


def _id_chunks(ids: list[int], batch: int) -> Iterator[list[int]]:
    """Split `ids` into chunks of at most `batch` elements.

    BUG-A1-08 (checklist 10): SQLite caps bound parameters per statement
    (SQLITE_LIMIT_VARIABLE_NUMBER, 32766 on stock builds), so every
    ``id IN (...)`` site must batch. This is the one place the slicing
    happens; ``MemoryStore._id_batch`` is the one place the chunk size is
    derived from the connection's limit. Extracting both keeps the
    arithmetic in a single helper instead of re-derived per call site.

    A ``batch`` of 0 (or less) is floored to 1 rather than raising or
    looping forever: a caller computing the size must never be able to turn
    a large delete into a silent no-op via ``range(0, n, 0)``. An empty
    input yields nothing, which is what every call site already assumed.
    """
    size = max(int(batch), 1)
    for start in range(0, len(ids), size):
        yield ids[start:start + size]

# Error codes follow the design doc's OMN-* table (extended where the spec was silent)
E_INTERNAL = "OMN-000"
E_TYPE = "OMN-002"
E_TABLE_NOT_FOUND = "OMN-003"
E_COLUMN_NOT_FOUND = "OMN-004"
E_CONSTRAINT = "OMN-005"
E_VECTOR_DIM_MISMATCH = "OMN-006"
E_TIMEOUT = "OMN-011"
E_CONSOLIDATION_FAILED = "OMN-013"
E_READ_ONLY = "OMN-020"
E_NOT_FOUND = "OMN-021"


class StoreError(RuntimeError):
    def __init__(self, message: str, code: str = E_INTERNAL):
        super().__init__(f"[{code}] {message}")
        self.code = code


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def _with_took(result: dict, t0: float) -> dict:
    result["took_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


def _commit_or_rollback(conn: sqlite3.Connection, exc: sqlite3.Error) -> StoreError:
    """Roll back the open transaction and translate the sqlite error onto the
    OMN-* taxonomy. A failed write must never leave partial rows for a later,
    unrelated commit to pick up (BUG-03)."""
    conn.rollback()
    code = E_CONSTRAINT if isinstance(exc, sqlite3.IntegrityError) else E_INTERNAL
    return StoreError(f"write failed: {exc}", code)


def _serialized(method):
    """Run a store operation under the store-wide reentrant lock.

    The MCP server executes tool bodies concurrently on anyio worker threads
    against one shared sqlite3.Connection (BUG-02): `search` mutates
    connection-global state (authorizer, progress handler), and the batch id
    read-back assumes no interleaved inserts. Serializing at the store boundary
    makes every tool atomic with respect to the others; the reentrancy lets
    delegating methods (remember -> remember_many, flush -> checkpoint) nest.

    This is also where a `tenant` keyword argument is validated (audit C3).
    `_serialized` is the single choke point every public store method passes
    through, so validating here means the guarantee cannot be forgotten by the
    next method someone adds — whereas validating inside `remember_many` alone
    would leave `forget`, `get` and the rest trusting their callers. Nested calls
    re-validate, which is cheap and harmless: `validate_tenant_id` is a length
    check and a set difference.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        tenant = kwargs.get("tenant")
        if tenant is not None:
            validate_tenant_id(tenant)
        with self._op_lock:
            return method(self, *args, **kwargs)
    return wrapper


def _record_query(store: "MemoryStore", kind: str, query: str,
                  result_count: int, took_ms: float, agent_id: str | None,
                  tenant: str = "default", commit: bool = True) -> None:
    """Insert a query-log row; opt-in via cfg.query_log (ARCH-03: the flag is
    plain instance state, not a module-level id(conn) set). Every
    ``query_retention_rows`` inserts the table is pruned to the newest rows
    (PERF-05) so a busy logged server cannot grow the home without bound.

    `tenant` is written explicitly. The column was added by migration 4 but
    this INSERT omitted it, so every logged row silently took the 'default'
    value: a recall issued by `acme` was recorded as `default`'s, which made
    per-tenant query attribution impossible, left `query-log` dumping every
    tenant's verbatim query text, and meant no tenant's query history could
    ever be erased (audit C15/R8)."""
    if not store._query_log_enabled:
        return
    try:
        store.conn.execute(
            "INSERT INTO queries(agent_id, kind, query, result_count, took_ms, "
            "created_ms, tenant_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (agent_id, kind, query, result_count, took_ms,
             int(time.time() * 1000), tenant),
        )
        store._query_inserts += 1
        max_rows = store.cfg.query_retention_rows
        if max_rows > 0 and store._query_inserts % max(64, max_rows // 4) == 0:
            store.conn.execute(
                "DELETE FROM queries WHERE id NOT IN "
                "(SELECT id FROM queries ORDER BY id DESC LIMIT ?)",
                (max_rows,),
            )
        if commit:
            store.conn.commit()
    except sqlite3.Error:
        pass


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _check_cold_start_budget(cold_start_ms: float) -> None:
    """One-shot env gate: warn to stderr if cold start exceeds OMNIDB_COLD_START_BUDGET_MS.

    No fail mode — this is an observability tripwire, not a hard limit. The
    warning prints exactly once per process (the caller invokes this from
    MemoryStore.__init__ only).
    """
    raw = os.environ.get("OMNIDB_COLD_START_BUDGET_MS")
    if not raw:
        return
    try:
        budget = int(raw)
    except ValueError:
        return
    if cold_start_ms > budget:
        print(
            f"omnidb-server: cold start {cold_start_ms:.2f}ms exceeds budget "
            f"OMNIDB_COLD_START_BUDGET_MS={budget}ms",
            file=sys.stderr,
        )


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder,
                 index: "HnswlibIndex | BruteForceIndex", cfg: Config,
                 now_fn=time.time, read_only: bool = False):
        # t0 must precede any IO so cold_start_ms reflects the full open path
        # (consistency check, index recovery, meta writes) the first time stats()
        # is queried. Subsequent stats() calls return the cached value.
        t0 = time.perf_counter()
        self._op_lock = threading.RLock()
        self.conn = conn
        self.embedder = embedder
        self.index: HnswlibIndex | BruteForceIndex = index
        self.cfg = cfg
        self.now_fn = now_fn
        # AUDIT-I02: a read-only diagnostic store performs zero persistence:
        # no recovery, no meta writes, no flush on close.
        self._read_only = read_only
        # Why this store's index is not the persisted one, when it is not.
        # The read-only opener falls back to a freshly built empty index after a
        # corrupt sidecar, which used to report itself as a healthy count-0
        # index against a non-zero live count — dense recall then silently
        # returned nothing (audit C6, the ARCH-04 false-clean class). Carrying
        # the reason here lets `stats` state the divergence in the payload
        # instead of only on stderr, so a caller that cannot see stderr can
        # still tell an empty home from a broken index.
        self.index_degraded_reason: str | None = None
        # AUDIT-I05: the budget is VM instructions, not progress callbacks;
        # the handler divides it by its callback interval below.
        self.max_query_steps = 5_000_000
        self._pending_ops = 0
        self._last_persist_at = now_fn()
        # AUDIT-I16: monotonic clock for the idle persistence deadline;
        # now_fn() is wall time and is also used by scoring.
        self._last_persist_mono = time.monotonic()
        self._last_flush_s = 0.0  # PERF-01: scales the time-trigger interval
        self._query_log_enabled = bool(cfg.query_log)
        self._query_inserts = 0
        # AUDIT-I01: ids whose committed content may not match the indexed
        # vector (crash between the DB commit and the sidecar persist).
        # Persisted in `meta` inside the same transaction as the content
        # UPDATE, so any interruption boundary re-embeds the committed row.
        self._pending_reembed: set[int] = set(
            int(i) for i in json.loads(get_meta(conn, "pending_reembed") or "[]")
        )
        self._check_consistency()
        if not read_only:
            self._recover_index_state()
            set_meta(conn, "schema_version", str(SCHEMA_VERSION))
            set_meta(conn, "embedder", embedder.name)
            set_meta(conn, "vec_dim", str(embedder.dim))
            set_meta(conn, "vec_kind", index.kind)
        self._cold_start_ms = round((time.perf_counter() - t0) * 1000, 2)
        _check_cold_start_budget(self._cold_start_ms)

    # -- consistency / recovery -------------------------------------------

    def _check_consistency(self) -> None:
        stored_name = get_meta(self.conn, "embedder")
        stored_dim = get_meta(self.conn, "vec_dim")
        if stored_name is not None and stored_name != self.embedder.name:
            raise StoreError(
                f'database was built with embedder "{stored_name}" but '
                f'"{self.embedder.name}" is configured; set OMNIDB_EMBEDDER='
                f"{stored_name} (or start a fresh home directory)",
                E_VECTOR_DIM_MISMATCH,
            )
        if stored_dim is not None and int(stored_dim) != self.embedder.dim:
            raise StoreError(
                f"vector dimension mismatch: database has {stored_dim}, "
                f"embedder produces {self.embedder.dim}",
                E_VECTOR_DIM_MISMATCH,
            )

    def _live_count(self) -> int:
        return int(self.conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {_SELECTIVE}"
        ).fetchone()[0])

    def _recover_index_state(self) -> None:
        """Cheap open path: full reconciliation only when a crash window is
        suspected (dirty flag set, or sidecar/DB counts disagree), and a full
        rebuild only when scheduled compaction is due."""
        if get_meta(self.conn, "compaction_due") == "1":
            self.rebuild_index(reason="scheduled compaction")
            return
        # AUDIT-I03: an interrupted re-embed leaves the sidecar's generation
        # unattributable (a same-dimension model swap is undetectable by the
        # consistency checks), so rebuild vectors from the rows — the source
        # of truth — under the stored identity before serving.
        if get_meta(self.conn, "reembed_pending") is not None:
            self.rebuild_index(reason="interrupted re-embed recovery")
            # The rebuilt generation now matches the stored identity; the
            # marker must not force a rebuild on every subsequent open.
            self.conn.execute("DELETE FROM meta WHERE key = 'reembed_pending'")
            self.conn.commit()
            return
        dirty = get_meta(self.conn, "index_dirty") == "1"
        if not dirty and not self._pending_reembed \
                and self.index.active_count() == self._live_count():
            return
        t0 = time.perf_counter()
        self._reconcile_index()
        set_meta(self.conn, "reconciled_at", _iso(int(time.time() * 1000)) or "")
        set_meta(self.conn, "reconciled_ms", f"{(time.perf_counter() - t0) * 1000:.1f}")

    def _reconcile_index(self) -> None:
        """Self-heal crash windows between the DB commit and index persist:
        re-embed live rows missing from the index, drop entries whose rows are
        gone. Also rebuilds fully if the sidecar file was deleted.

        AUDIT-I01: membership alone cannot see a same-ID content revision,
        so ids recorded in `pending_reembed` are re-embedded too.
        AUDIT-I13: fetch/encode/remove run in bounded batches so a lost
        sidecar cannot materialize the whole home in one query."""
        self._mark_dirty()
        live = {
            int(r["id"])
            for r in self.conn.execute(f"SELECT id FROM memories WHERE {_SELECTIVE}")
        }
        indexed = self.index.indexed_ids()
        missing = sorted((live - indexed) | (self._pending_reembed & live))
        stale = sorted(indexed - live)
        for part in _id_chunks(missing, self._id_batch()):
            placeholders = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
                part,
            ).fetchall()
            vectors = self.embedder.encode([r["content"] for r in rows])
            self.index.add([int(r["id"]) for r in rows], vectors)
        for part in _id_chunks(stale, self._id_batch()):
            self.index.remove(part)
        self._persist_index()

    def _id_batch(self) -> int:
        """The largest id list this connection can bind in one statement.

        SQLite caps bound parameters per statement (SQLITE_LIMIT_VARIABLE_NUMBER,
        32766 on stock builds), so every ``id IN (...)`` site must batch. The
        reserve covers the extra parameters each caller binds alongside the id
        list; the cap bounds the work per statement.

        BUG-A1-08: forget_many and purge built an unbounded placeholder list, so
        a home with more than the limit's worth of rows could not be purged at
        all and forget_many failed for any caller passing more ids than the
        limit. One helper, one place to get the arithmetic right.

        The ceiling is ``min(_ID_BATCH_CAP, _RECONCILE_BATCH_CAP)`` so the
        existing ``_RECONCILE_BATCH_CAP`` seam keeps working: AUDIT-I13's own
        test monkeypatches that constant to 2 to prove recovery batches, and
        introducing a second, independent cap silently disabled it.
        """
        return max(1, min(_ID_BATCH_CAP, _RECONCILE_BATCH_CAP,
                          self.conn.getlimit(
                              sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
                          - _ID_BIND_RESERVE))

    # -- persistence plumbing ----------------------------------------------

    def _mark_dirty(self) -> None:
        if self._read_only:
            return
        set_meta(self.conn, "index_dirty", "1")

    def _persist_index(self) -> None:
        if self._read_only:
            return
        self.index.persist()
        set_meta(self.conn, "index_dirty", "0")
        # AUDIT-I01: the revision queue clears only after vectors are durable.
        self._pending_reembed.clear()
        set_meta(self.conn, "pending_reembed", "[]")
        self._pending_ops = 0
        self._last_persist_at = self.now_fn()
        self._last_persist_mono = time.monotonic()

    def _maybe_persist(self) -> None:
        if self._pending_ops == 0:
            return
        elapsed = self.now_fn() - self._last_persist_at
        # PERF-01: a sidecar persist is a full O(N) dump, so the time trigger
        # must not fire per slow write. It now needs persist_min_ops pending
        # ops AND an interval that scales with the last flush's own duration.
        # AUDIT-I18: this cost control applies only to the elapsed-time
        # trigger; the operation-count and maximum-age limits deliberately
        # override it to bound reconstruction work and dirty-vector age.
        interval = max(self.cfg.persist_every_s,
                       self._FLUSH_COST_MULTIPLIER * self._last_flush_s)
        if (self._pending_ops >= self.cfg.persist_every_ops
                or (self._pending_ops >= self.cfg.persist_min_ops
                    and elapsed >= interval)
                or elapsed >= self.cfg.persist_max_s):
            self.flush()

    _FLUSH_COST_MULTIPLIER = 4.0

    def _after_commit(self, apply_index, n_ops: int,
                      compaction: bool = False) -> None:
        """The post-commit half of every write path, in exactly one place
        (ARCH-02): the dirty flag is set FIRST — a crash during the index
        mutation must leave `index_dirty=1` so the open path reconciles —
        then the index mutation, then optional compaction scheduling, then
        the debounced sidecar persist. Callers do SQL + commit (via
        `_commit_or_rollback`) and hand this the index mutation."""
        self._mark_dirty()
        apply_index()
        if compaction:
            self._schedule_compaction_if_needed()
        self._pending_ops += n_ops
        self._maybe_persist()

    @_serialized
    def flush(self) -> None:
        """Force-persist the vector sidecar and checkpoint the WAL."""
        t0 = time.perf_counter()
        self._persist_index()
        self.checkpoint()
        self._last_flush_s = time.perf_counter() - t0

    def checkpoint(self) -> None:
        """PASSIVE WAL checkpoint; tolerated when a concurrent reader blocks it."""
        with self._op_lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.OperationalError:
                pass  # a concurrent reader can legitimately block a checkpoint

    @_serialized
    def maintenance_tick(self) -> None:
        """AUDIT-I16: enforce the idle persistence deadline.

        The write-path `_maybe_persist` only runs after an indexed mutation,
        so a dirty sidecar could sit indefinitely after the final operation.
        The server's background loop calls this on a fixed cadence; it flushes
        when the durability ceiling (`persist_max_s`) has elapsed on monotonic
        time and does nothing when the store is already clean."""
        if self._pending_ops == 0 and not self._pending_reembed:
            return
        if time.monotonic() - self._last_persist_mono >= self.cfg.persist_max_s:
            self.flush()

    @contextmanager
    def locked(self):
        """Serialize a compound multi-step operation against concurrent tool
        calls. Public methods already lock individually; use this only for
        sequences that must not interleave (e.g. consolidation's snapshot and
        mark phases, which touch conn outside store methods)."""
        with self._op_lock:
            yield

    # -- writes ------------------------------------------------------------

    @_serialized
    def remember(self, content: str, kind: str = "semantic",
                 metadata: dict | None = None,
                 links: list[dict] | None = None,
                 agent_id: str | None = None,
                 trigger_patterns: list[str] | None = None,
                 created_ms: int | None = None,
                 access_count: int | None = None,
                 _explicit_id: int | None = None,
                 tenant: str = "default") -> dict:
        # Single-item path is a one-liner on top of the batch path. The
        # batch path is what makes bulk import (and high-throughput
        # ingestion) viable — one commit, one embed call, one persist.
        results = self.remember_many([{
            "content": content,
            "kind": kind,
            "metadata": metadata,
            "links": links,
            "agent_id": agent_id,
            "trigger_patterns": trigger_patterns,
            "created_ms": created_ms,
            "access_count": access_count,
            "_explicit_id": _explicit_id,
            "tenant": tenant,
        }])
        return results[0]

    @_serialized
    def remember_many(self, items: list[dict]) -> list[dict]:
        """Persist many memories in a single transaction (J-09).

        Each item is a dict with the same kwargs remember() accepts. The
        whole batch shares one embedder call and one SQL transaction, with
        per-row INSERTs capturing their assigned ids. Sidecar persistence is
        debounced. Returns the new ids in the same order.
        """
        if not items:
            return []
        t0 = time.perf_counter()
        now_ms = int(self.now_fn() * 1000)
        kinds: list[str] = []
        contents: list[str] = []
        metadatas: list[str] = []
        agent_ids: list[str | None] = []
        trigger_patterns_list: list[str | None] = []
        created_mses: list[int] = []
        access_counts: list[int] = []
        explicit_ids: list[int | None] = []
        tenants: list[str] = []
        prepared_links: list[list[tuple[int, str]]] = []

        for item in items:
            kind = item.get("kind", "semantic")
            if kind not in _KINDS:
                raise StoreError(f"kind must be one of {_KINDS}", E_TYPE)
            content = item["content"]
            # AUDIT-I21: one shared content policy across create/update/import.
            try:
                validate_content(content, self.cfg.max_content_chars)
            except TypeError as exc:
                raise StoreError(str(exc), E_TYPE) from exc
            except ValueError as exc:
                # SEC-04: bound one call's blast radius on DB, FTS and embedder
                raise StoreError(str(exc), E_CONSTRAINT) from exc
            kinds.append(kind)
            contents.append(content)
            try:
                metadatas.append(json.dumps(item.get("metadata") or {}))
            except TypeError as exc:
                # QA-02: metadata must be JSON-serializable; fail typed, early
                raise StoreError(
                    f"metadata must be JSON-serializable: {exc}", E_TYPE
                ) from exc
            agent_ids.append(item.get("agent_id"))
            patterns = item.get("trigger_patterns")
            if patterns is not None:
                # AUDIT-I06: admission bounds — a stored pattern must be a
                # list of bounded strings before it can ever reach the matcher.
                try:
                    validate_trigger_patterns(patterns)
                except TypeError as exc:
                    raise StoreError(str(exc), E_TYPE) from exc
                except ValueError as exc:
                    raise StoreError(str(exc), E_CONSTRAINT) from exc
            trigger_patterns_list.append(
                json.dumps(patterns) if patterns is not None else None
            )
            created_mses.append(int(item.get("created_ms") or now_ms))
            access_counts.append(int(item.get("access_count") or 0))
            explicit_ids.append(item.get("_explicit_id"))
            # ADR-08: the tenant is validated below, at the store boundary. The
            # auth layer validates too, but the store is what every other caller
            # (CLI --tenant, skills, the public Python API) reaches first, so it
            # validates rather than assuming (audit C3). The default keeps
            # single-tenant deployments and every pre-tenant caller working
            # unchanged.
            tenant = str(item.get("tenant") or "default")
            validate_tenant_id(tenant)
            tenants.append(tenant)
            prepared_links.append(self._validate_links(
                item.get("links") or [], tenant=tenants[-1]))

        vectors = self.embedder.encode(contents)

        # Build the per-row INSERT statement. Always-on columns first
        # (kind, content, metadata, created_ms, access_count, tenant_id); v2
        # columns (agent_id, trigger_patterns) and explicit id are per row.
        base_columns = ["kind", "content", "metadata", "created_ms",
                        "access_count", "tenant_id"]
        per_row_extra: list[list[tuple[str, object]]] = []
        for i in range(len(items)):
            extras: list[tuple[str, object]] = []
            agent_id_i = agent_ids[i]
            if agent_id_i is not None:
                extras.append(("agent_id", agent_id_i))
            patterns_i = trigger_patterns_list[i]
            if patterns_i is not None:
                extras.append(("trigger_patterns", patterns_i))
            explicit_i = explicit_ids[i]
            if explicit_i is not None:
                extras.append(("id", int(explicit_i)))
            per_row_extra.append(extras)

        # Reuse one SQL template for the per-row INSERTs. Optional columns
        # omitted by a row receive NULL; a NULL integer primary key lets
        # SQLite assign that row's id automatically.
        all_columns: list[str] = list(base_columns)
        seen: set[str] = set(base_columns)
        for extras in per_row_extra:
            for col, _ in extras:
                if col not in seen:
                    seen.add(col)
                    all_columns.append(col)
        placeholders = ", ".join("?" for _ in all_columns)
        sql = f"INSERT INTO memories({', '.join(all_columns)}) VALUES ({placeholders})"

        # An explicit id with NULL for unspecified fields is fine because
        # we built the row parameter set to match the column list. Cells
        # where a column was never set get NULL.
        rows = []
        for i in range(len(items)):
            row: list[object] = [
                kinds[i], contents[i], metadatas[i], created_mses[i],
                access_counts[i], tenants[i],
            ]
            row_dict = dict(zip(all_columns, row))
            for col, val in per_row_extra[i]:
                row_dict[col] = val
            rows.append([row_dict.get(col) for col in all_columns])

        try:
            # AUDIT-I07: capture each assigned id, including interleaved
            # explicit ids, without another read or transaction boundary.
            new_ids: list[int] = []
            for row in rows:
                cur = self.conn.execute(sql, row)
                if cur.lastrowid is None:
                    raise StoreError("insert did not report an assigned id",
                                     E_INTERNAL)
                new_ids.append(int(cur.lastrowid))

            # Inline prepared links in the same transaction.
            for src, item_links in zip(new_ids, prepared_links):
                for dst, label in item_links:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, label, created_ms) "
                        "VALUES (?, ?, ?, ?)",
                        (src, dst, label, now_ms),
                    )
            # ADR-08 / W3-3: FTS sync is in-transaction, exactly as the removed
            # triggers did. A crash between this and the commit cannot leave FTS
            # and rows divergent, because they commit together.
            for mid, content in zip(new_ids, contents):
                self.conn.execute(
                    "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                    (int(mid), content),
                )
            self.conn.commit()
        except sqlite3.Error as exc:
            # A failed batch must not leave partial rows behind: the next
            # unrelated commit would otherwise persist them with no vectors
            # in the index (BUG-03).
            raise _commit_or_rollback(self.conn, exc) from exc
        self._after_commit(lambda: self.index.add(new_ids, vectors), len(new_ids))
        return [
            _with_took({"id": mid, "kind": k}, t0)
            for mid, k in zip(new_ids, kinds)
        ]

    @_serialized
    def update_memory(self, memory_id: int, content: str | None = None,
                      metadata: dict | None = None,
                      tenant: str = "default") -> dict:
        t0 = time.perf_counter()
        row = self.get(memory_id, tenant)
        if row is None or row["deleted_ms"] is not None:
            raise StoreError(f"memory {memory_id} does not exist or is deleted", E_NOT_FOUND)
        updated: list[str] = []
        new_content = None
        if content is not None and content != row["content"]:
            new_content = content
            updated.append("content")
        new_metadata_json = None
        if metadata is not None:
            merged = json.loads(row["metadata"])
            merged.update(metadata)
            try:
                new_metadata_json = json.dumps(merged)
            except TypeError as exc:
                raise StoreError(
                    f"metadata must be JSON-serializable: {exc}", E_TYPE
                ) from exc
            updated.append("metadata")
        if not updated:
            return _with_took({"id": memory_id, "updated": []}, t0)
        if new_content is not None:
            # AUDIT-I21: the creation-time content policy also guards updates,
            # so the configured size bound is one contract across write paths.
            try:
                validate_content(new_content, self.cfg.max_content_chars)
            except TypeError as exc:
                raise StoreError(str(exc), E_TYPE) from exc
            except ValueError as exc:
                raise StoreError(str(exc), E_CONSTRAINT) from exc
            # AUDIT-I01: encode BEFORE any SQL mutation — an encoder failure
            # leaves content, metadata and vectors exactly as they were.
            vec = self.embedder.encode([new_content])[0]
        sets: list[str] = []
        params: list[str | int] = []
        if new_content is not None:
            sets.append("content = ?")
            params.append(new_content)
        if new_metadata_json is not None:
            sets.append("metadata = ?")
            params.append(new_metadata_json)
        params.append(memory_id)
        # ADR-08: the tenant check is a hard equality, not just a pre-check via
        # get() — a foreign row cannot be edited even if the id is guessed.
        params.append(tenant)
        try:
            self.conn.execute(
                f"UPDATE memories SET {', '.join(sets)} "
                f"WHERE id = ? AND tenant_id = ?", params
            )
            if new_content is not None:
                # AUDIT-I01: record the content revision in the SAME
                # transaction as the row update. A crash before the sidecar
                # persist re-embeds this id on the next open, so a committed
                # revision can never coexist with a stale vector.
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('pending_reembed', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (json.dumps(sorted(self._pending_reembed | {int(memory_id)})),),
                )
                # ADR-08 / W3-3: replace the FTS posting in the same
                # transaction, matching what the removed `memories_au` trigger
                # did (delete old content, insert new).
                self.conn.execute(
                    "DELETE FROM memory_fts WHERE rowid = ?", (int(memory_id),)
                )
                self.conn.execute(
                    "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                    (int(memory_id), new_content),
                )
            self.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(self.conn, exc) from exc
        if new_content is not None:
            self._pending_reembed.add(int(memory_id))
            self._after_commit(lambda: self.index.update(memory_id, vec), 1)
        return _with_took({"id": memory_id, "updated": updated}, t0)

    @_serialized
    def forget_many(self, memory_ids: list[int],
                    tenant: str = "default") -> dict:
        t0 = time.perf_counter()
        wanted = [int(m) for m in dict.fromkeys(memory_ids)]
        if not wanted:
            return _with_took({"forgotten": [], "missing": []}, t0)
        # BUG-A1-08: batch both statements. A caller passing more ids than the
        # connection's variable limit previously raised outright; a home with
        # more tombstoned rows than the limit could not be purged at all.
        batch = self._id_batch()
        forgotten: list[int] = []
        for part in _id_chunks(wanted, batch):
            placeholders = ",".join("?" * len(part))
            # ADR-08: id-addressed mutation is tenant-scoped — a caller cannot
            # tombstone another tenant's rows, no matter how many ids it names.
            live_rows = self.conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) "
                f"AND {_SELECTIVE} AND tenant_id = ?",
                [*part, tenant],
            ).fetchall()
            forgotten.extend(int(r["id"]) for r in live_rows)
        # AUDIT-I19: build the membership set once, not per wanted id.
        forgotten_set = set(forgotten)
        missing = [m for m in wanted if m not in forgotten_set]
        if not forgotten:
            return _with_took({"forgotten": [], "missing": missing}, t0)
        stamp = int(self.now_fn() * 1000)
        try:
            for part in _id_chunks(forgotten, batch):
                ph = ",".join("?" * len(part))
                self.conn.execute(
                    f"UPDATE memories SET deleted_ms = ? "
                    f"WHERE id IN ({ph}) AND {_SELECTIVE}",
                    [stamp, *part],
                )
            self.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(self.conn, exc) from exc
        self._after_commit(lambda: self.index.remove(forgotten), len(forgotten),
                           compaction=True)
        return _with_took({"forgotten": forgotten, "missing": missing}, t0)

    @_serialized
    def link(self, src: int, dst: int, label: str = "related",
             tenant: str = "default") -> None:
        try:
            src, dst = int(src), int(dst)
        except (TypeError, ValueError) as exc:
            raise StoreError("link endpoints must be integer memory ids", E_TYPE) from exc
        # ADR-08: links only bind rows within one tenant, so a forged edge can
        # never bridge two tenants (the door `related()` would otherwise open).
        for mid in (src, dst):
            target = self.get(mid, tenant)
            if target is None or target["deleted_ms"] is not None:
                raise StoreError(
                    f"memory {mid} does not exist or is deleted", E_CONSTRAINT
                )
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, ?)",
                (src, dst, label, int(self.now_fn() * 1000)),
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(self.conn, exc) from exc

    @_serialized
    def forget(self, memory_id: int, tenant: str = "default") -> dict:
        t0 = time.perf_counter()
        result = self.forget_many([memory_id], tenant)
        if not result["forgotten"]:
            raise StoreError(
                f"memory {memory_id} not found or already deleted", E_NOT_FOUND
            )
        return _with_took({"forgotten": memory_id}, t0)

    @_serialized
    def supersede(self, old_id: int, new_id: int,
                  tenant: str = "default") -> dict:
        old_id, new_id = int(old_id), int(new_id)
        if old_id == new_id:
            raise StoreError("supersede endpoints must differ", E_TYPE)
        # ADR-08: supersession, like linking, is within-tenant only.
        for mid in (old_id, new_id):
            row = self.get(mid, tenant)
            if row is None or row["deleted_ms"] is not None:
                raise StoreError(
                    f"memory {mid} does not exist or is deleted", E_NOT_FOUND
                )
            # BUG-09: superseding by an already-superseded row would close a
            # cycle (A->B then B->A) and drop both rows from the live set.
            if mid == new_id and row["superseded_by"] is not None:
                raise StoreError(f"memory {new_id} is already superseded", E_CONSTRAINT)
        try:
            cur = self.conn.execute(
                f"UPDATE memories SET superseded_by = ? "
                f"WHERE id = ? AND tenant_id = ? "
                f"AND superseded_by IS NULL AND {_SELECTIVE}",
                (new_id, old_id, tenant),
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(self.conn, exc) from exc
        if cur.rowcount == 0:
            raise StoreError(f"memory {old_id} is already superseded", E_CONSTRAINT)
        # the old row leaves the live set, so treat it like a soft delete
        self._after_commit(lambda: self.index.remove([old_id]), 1, compaction=True)
        return {"superseded": old_id, "by": new_id}

    def _validate_links(self, links: list[dict],
                        tenant: str = "default") -> list[tuple[int, str]]:
        prepared: list[tuple[int, str]] = []
        for link in links:
            if not isinstance(link, dict) or "to" not in link:
                raise StoreError(
                    "each link must be {'to': <existing memory id>, 'label': optional str}",
                    E_CONSTRAINT,
                )
            try:
                dst = int(link["to"])
            except (TypeError, ValueError) as exc:
                raise StoreError(
                    f"link target {link['to']!r} is not an integer id", E_TYPE
                ) from exc
            # ADR-08: a link target must exist AND belong to the same tenant as
            # the row being written, so a batch can never forge a cross-tenant
            # edge through remember().
            target = self.get(dst, tenant)
            if target is None or target["deleted_ms"] is not None:
                raise StoreError(
                    f"link target {dst} does not exist or is deleted", E_CONSTRAINT
                )
            prepared.append((dst, str(link.get("label", "related"))))
        return prepared

    # -- reads -------------------------------------------------------------

    @_serialized
    def get(self, memory_id: int, tenant: str = "default") -> dict | None:
        # ADR-08: id-addressed reads are tenant-scoped, so a caller can never
        # read another tenant's row by guessing or enumerating ids.
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ? AND tenant_id = ?",
            (memory_id, tenant),
        ).fetchone()
        return _row_to_dict(row) if row else None

    @_serialized
    def recall(self, query: str, k: int = 5, kind: str | None = None,
               k_fetch: int | None = None, agent_id: str | None = None,
               reranker: str | None = None, tenant: str = "default") -> list[dict]:
        from .recall import run_recall
        return run_recall(self, query, k, kind, k_fetch, agent_id, reranker,
                          log_query=self._query_log_enabled, tenant=tenant)

    @_serialized
    def search(self, sql: str, tenant: str = "default", *,
               unscoped: bool = False) -> dict:
        # ADR-08: raw SQL cannot be tenant-scoped safely — SQLite's authorizer
        # sees column names, never predicate values, so a `WHERE` is invisible
        # to it and `SELECT * FROM memories` would cross tenants.
        #
        # The gate is the `unscoped` CAPABILITY, set only by a transport that
        # knows the whole home is one trust domain (the CLI, and stdio via
        # attach_tools). It used to be `tenant != "default"` — a comparison
        # against an operator-controlled *name*. `default` is a legal tenant id
        # in a key map, so a credential registered under the most natural name
        # (the same one every single-tenant deployment already uses) unlocked
        # an unscoped read of every tenant's rows. The name is data; the
        # capability is not, so the capability is what we branch on.
        if not unscoped:
            raise StoreError(
                "search is unavailable to scoped tenants; run it via the "
                "single-tenant CLI where the whole home is one trust domain",
                E_READ_ONLY,
            )
        t0 = time.perf_counter()
        statement = " ".join(sql.split()).rstrip(";").strip()
        match = _SQL_FIRST_WORD.match(statement)
        first = (match.group(1).lower() if match else "")
        if first not in ("select", "with"):
            raise StoreError(
                "only read-only SELECT / WITH statements are allowed", E_READ_ONLY
            )

        # The allowlist alone cannot stop SQLite's data-modifying CTEs
        # ("WITH t AS (...) DELETE FROM ..."); the authorizer is enforcement.
        def _readonly(action, arg1, arg2, dbname, source):
            if action in (
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            ):
                return sqlite3.SQLITE_OK
            # FTS5 internals read PRAGMA data_version; that single pragma is
            # safe — every other one stays denied
            if action == sqlite3.SQLITE_PRAGMA and arg1 == "data_version":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        # AUDIT-I05: the progress handler fires once per N SQLite VM
        # instructions, so the callback count must be budget / N — counting
        # callbacks against the instruction budget under-counted ~10,000x.
        # A monotonic deadline bounds wall-clock work, and a byte cap bounds
        # the materialized result. A VM-callback deadline cannot interrupt a
        # long-running scalar function mid-call; hard wall-clock guarantees
        # would need process isolation, which this local channel does not do.
        interval = 1_000
        max_callbacks = max(1, self.max_query_steps // interval)
        deadline = time.monotonic() + _QUERY_WALL_CLOCK_LIMIT_S
        state = {"steps": 0}

        def _progress():
            state["steps"] += 1
            if state["steps"] > max_callbacks or time.monotonic() > deadline:
                return 1
            return 0

        self.conn.set_authorizer(_readonly)
        self.conn.set_progress_handler(_progress, interval)
        rows: list = []
        bytes_seen = 0
        over_bytes = False
        try:
            cursor = self.conn.execute(statement)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            while True:
                batch = cursor.fetchmany(256)
                if not batch:
                    break
                for row in batch:
                    bytes_seen += sum(len(c) if isinstance(c, str) else 8
                                      for c in row)
                    rows.append(row)
                    if bytes_seen > _QUERY_RESULT_BYTES_LIMIT:
                        over_bytes = True
                        break
                if over_bytes or len(rows) > self.cfg.max_search_rows:
                    break
        except sqlite3.DatabaseError as exc:
            raise StoreError(*_classify_sqlite_error(exc)) from exc
        finally:
            self.conn.set_authorizer(None)
            self.conn.set_progress_handler(None, 0)
        truncated = over_bytes or len(rows) > self.cfg.max_search_rows
        result = _with_took({
            "columns": columns,
            "rows": [list(r) for r in rows[: self.cfg.max_search_rows]],
            "truncated": truncated,
        }, t0)
        _record_query(self, "search", sql, len(rows), result["took_ms"], None,
                       tenant=tenant)
        return result

    @_serialized
    def related(self, memory_id: int, max_hops: int = 2,
                label: str | None = None,
                tenant: str = "default") -> list[dict]:
        # client-supplied hops are clamped: undirected recursion without visit
        # tracking is exponential even with modest depth
        hops = max(1, min(int(max_hops), 8))
        join_cond = "(e.src = walk.node OR e.dst = walk.node)"
        # placeholder order: start node, seen-init, [label], depth limit,
        # tenant (fence), exclude-start, tenant (final)
        params: list = [memory_id, memory_id]
        if label is not None:
            join_cond += " AND e.label = ?"
            params.append(label)
        params.extend([hops, tenant])
        rows = self.conn.execute(
            f"""
            WITH RECURSIVE walk(node, depth, seen) AS (
                SELECT ?, 0, ',' || CAST(? AS TEXT) || ','
                UNION ALL
                SELECT n.id,
                       walk.depth + 1,
                       walk.seen || CAST(n.id AS TEXT) || ','
                FROM walk
                JOIN edges e ON {join_cond}
                JOIN memories n
                  ON n.id = CASE WHEN e.src = walk.node THEN e.dst ELSE e.src END
                WHERE walk.depth < ?
                  -- ADR-08: the walk never leaves the requesting tenant — a
                  -- forged cross-tenant edge is refused at link() time, and
                  -- even a historical one stops here because the neighbour is
                  -- only followed when it belongs to this tenant.
                  AND n.tenant_id = ?
                  AND instr(walk.seen, ',' || CAST(n.id AS TEXT) || ',') = 0
            )
            SELECT m.id, m.kind, m.content, MIN(walk.depth) AS depth
            FROM walk JOIN memories m ON m.id = walk.node
            WHERE m.deleted_ms IS NULL AND m.superseded_by IS NULL AND m.id != ?
              AND m.tenant_id = ?
            GROUP BY m.id, m.kind, m.content
            ORDER BY depth, m.id
            LIMIT 200
            """,
            [*params, memory_id, tenant],
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # -- maintenance -------------------------------------------------------

    @_serialized
    def check(self) -> dict:
        try:
            self.conn.execute(
                "INSERT INTO memory_fts(memory_fts) VALUES('integrity-check')"
            )
            self.conn.commit()
            fts_integrity = "ok"
        except sqlite3.DatabaseError as exc:
            self.conn.rollback()
            fts_integrity = str(exc)
        live = int(self.conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {_SELECTIVE}"
        ).fetchone()[0])
        orphan_edges = int(self.conn.execute(
            "SELECT COUNT(*) FROM edges e WHERE NOT EXISTS "
            "(SELECT 1 FROM memories m WHERE m.id = e.src) OR NOT EXISTS "
            "(SELECT 1 FROM memories m WHERE m.id = e.dst)"
        ).fetchone()[0])
        wal_path = self._home() / "omnidb.db-wal"
        return {
            "fts_integrity": fts_integrity,
            "index_drift": abs(self.index.active_count() - live),
            "orphan_edges": orphan_edges,
            "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            "soft_deleted": int(self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted_ms IS NOT NULL"
            ).fetchone()[0]),
            "superseded": int(self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]),
        }

    @_serialized
    def purge(self, do_vacuum: bool = False,
              tenant: str = "default") -> dict:
        # DEBT-01: superseded-only rows were never purgeable and appeared in
        # no report. Capture the id set first, null their superseded_by
        # references (the FK has no cascade and purgees may reference each
        # other), then hard-delete by id.
        try:
            # ADR-08: purge is tenant-scoped — a caller can only reclaim space
            # for its own soft-deleted rows, never another tenant's.
            ids = [int(r[0]) for r in self.conn.execute(
                "SELECT id FROM memories "
                "WHERE (deleted_ms IS NOT NULL OR superseded_by IS NOT NULL) "
                "AND tenant_id = ?",
                (tenant,),
            ).fetchall()]
            purged = 0
            # BUG-A1-08: both statements batch. A home holding more purgeable
            # rows than the connection's variable limit could not be purged at
            # all — the placeholder list was built unbounded, so the statement
            # raised and the operator had no way to reclaim the space.
            batch = self._id_batch()
            for part in _id_chunks(ids, batch):
                ph = ",".join("?" * len(part))
                self.conn.execute(
                    f"UPDATE memories SET superseded_by = NULL WHERE id IN ({ph})",
                    part,
                )
                # ADR-08 / W3-3: drop the FTS postings in the same transaction,
                # matching what the removed `memories_ad` trigger did.
                self.conn.execute(
                    f"DELETE FROM memory_fts WHERE rowid IN ({ph})", part
                )
                cur = self.conn.execute(
                    f"DELETE FROM memories WHERE id IN ({ph})", part
                )
                purged += max(int(cur.rowcount), 0)
            if not ids:
                cur = self.conn.execute("SELECT 1 WHERE 0")
            self.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(self.conn, exc) from exc
        if do_vacuum:
            self.conn.execute("VACUUM")
            self.conn.commit()
        return {"purged": purged}

    def _schedule_compaction_if_needed(self) -> None:
        if self.index.kind == "hnsw" and self.index.dead_ratio >= self.cfg.dead_compact_ratio:
            set_meta(self.conn, "compaction_due", "1")

    @_serialized
    def rebuild_index(self, reason: str = "manual") -> dict:
        """Full vector rebuild from DB contents (compaction / sidecar loss)."""
        from .vector_index import BruteForceIndex, HnswlibIndex

        rows = self.conn.execute(
            f"SELECT id, content FROM memories WHERE {_SELECTIVE} ORDER BY id"
        ).fetchall()
        dim = self.embedder.dim
        home = self._home()

        fresh: HnswlibIndex | BruteForceIndex
        # BUG-A1-12: construct first, unlink after. The old order deleted the
        # sidecar and then constructed its replacement, so a constructor
        # failure (dim mismatch, hnswlib error) destroyed the only vector copy
        # and forced a full re-embed. Building into a scratch path and
        # publishing with os.replace makes a failed rebuild a no-op.
        if isinstance(self.index, HnswlibIndex):
            scratch_path = home / "index.hnsw.rebuilding"
            scratch_path.unlink(missing_ok=True)
            (home / "index.hnsw.rebuilding.deleted.json").unlink(missing_ok=True)
            fresh = HnswlibIndex(scratch_path, dim, count_hint=len(rows),
                                 ef_search=self.cfg.ef_search)
        else:
            scratch_path = home / "vectors.npz.rebuilding"
            scratch_path.unlink(missing_ok=True)
            fresh = BruteForceIndex(scratch_path, dim)
        # remember where the scratch artifacts live so the swap can publish
        # them under the canonical names once the build has succeeded
        scratch_deleted = scratch_path.with_name(
            scratch_path.name + ".deleted.json")

        self._mark_dirty()
        chunk = 512
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            vectors = self.embedder.encode([r["content"] for r in part])
            fresh.add([int(r["id"]) for r in part], vectors)
        # Materialise the scratch artifacts BEFORE publishing. An index object
        # writes nothing to disk until persist(), so replacing first would
        # rename a file that does not exist yet.
        fresh.persist()
        # The build succeeded, so the scratch artifacts can be published under
        # the canonical names. Until this point the previous sidecar was
        # untouched and the home remained loadable.
        canonical = (home / "index.hnsw") if isinstance(
            self.index, HnswlibIndex) else (home / "vectors.npz")
        os.replace(scratch_path, canonical)
        if scratch_deleted.exists():
            os.replace(scratch_deleted, canonical.with_name(
                canonical.name + ".deleted.json"))
        # the index object keeps writing to self.path, so repoint it at the
        # published name before it becomes the live index. The tombstone
        # sidecar must be repointed under its REAL attribute name: guarded by
        # `hasattr(fresh, "deleted_path")` (no such attribute) the repoint was
        # silently skipped, so every later persist() wrote tombstones to the
        # scratch file while the canonical sidecar stayed empty — and the open
        # path, which reads only the canonical name, resurrected every deleted
        # row into dense results after a reopen.
        fresh.path = canonical
        if hasattr(fresh, "_deleted_path"):
            fresh._deleted_path = canonical.with_name(
                canonical.name + ".deleted.json")
        self.index = fresh
        self._persist_index()
        set_meta(self.conn, "compaction_due", "0")
        set_meta(self.conn, "last_rebuild_reason", reason)
        return {"rebuilt": len(rows), "reason": reason,
                "index_kind": self.index.kind}

    # -- introspection -----------------------------------------------------

    @_serialized
    def stats(self, tenant: str = "default") -> dict:
        """Tenant-scoped database statistics.

        Every count in this payload is fenced by `tenant`. `edges` is counted
        through its source memory because the table has no `tenant_id` column,
        and `files` lists only the database artefacts rather than the whole home
        (audit C7) — a home-wide listing disclosed `tenants.json` and per-file
        sizes, which vary with other tenants' activity.
        """
        t0 = time.perf_counter()
        # ADR-08: every tenant-scoped count is fenced — the audit's MT-02
        # bypass exposed `agents` (and, by the same shape, `live`/`by_kind`) to
        # any caller, so these now take an explicit tenant.
        kinds = {
            r["kind"]: r["n"]
            for r in self.conn.execute(
                f"SELECT kind, COUNT(*) AS n FROM memories "
                f"WHERE {_SELECTIVE} AND tenant_id = ? GROUP BY kind",
                (tenant,),
            )
        }
        deleted = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_ms IS NOT NULL "
            "AND tenant_id = ?", (tenant,)
        ).fetchone()[0]
        agents = {
            r["agent_id"]: r["n"]
            for r in self.conn.execute(
                f"SELECT agent_id, COUNT(*) AS n FROM memories "
                f"WHERE agent_id IS NOT NULL AND {_SELECTIVE} "
                f"AND tenant_id = ? "
                f"GROUP BY agent_id ORDER BY n DESC, agent_id LIMIT 20",
                (tenant,),
            )
        }
        # `edges` has no tenant_id column (db.py:_MIGRATION_1) — migration 4 adds
        # one to memories and queries only. Counting it directly therefore
        # returned the whole home's total to every tenant, which is why this
        # counts through the src memory instead of assuming a fence that the
        # schema cannot express (audit C7).
        edges = self.conn.execute(
            "SELECT COUNT(*) FROM edges e "
            "JOIN memories m ON m.id = e.src "
            "WHERE m.tenant_id = ?",
            (tenant,),
        ).fetchone()[0]
        # A home-wide listing disclosed tenants.json, the lock file and per-file
        # sizes — all of which vary with another tenant's activity (audit C7).
        # Scoped tenants get only the database artefacts, which are deployment
        # facts; the single-tenant caller owns the whole home, so it still gets
        # the sidecars an operator needs for disk accounting (I-05).
        if tenant == DEFAULT_TENANT:
            files = {
                p.name: p.stat().st_size
                for p in sorted(self._home().iterdir())
                if p.is_file()
            }
        else:
            files = {
                p.name: p.stat().st_size
                for p in sorted(self._home().iterdir())
                if p.is_file() and p.name.startswith("omnidb.db")
            }
        index_count = self.index.active_count()
        # An index holding fewer vectors than there are live rows is the
        # divergence that matters: dense recall will miss rows silently. Stated
        # in the payload, not just on stderr, because an MCP caller reading
        # `stats` cannot see the opener's stderr (audit C6).
        index_drift = self.index_degraded_reason or (
            f"index holds {index_count} vectors for {sum(kinds.values())} live rows"
            if index_count != sum(kinds.values()) else None
        )
        payload = {
            "tenant": tenant,
            "live": sum(kinds.values()),
            "by_kind": kinds,
            "soft_deleted": deleted,
            "agents": agents,
            "edges": edges,
            "embedder": {"name": self.embedder.name, "dim": self.embedder.dim},
            "index": {
                "kind": self.index.kind,
                "count": index_count,
                "dead_ratio": round(getattr(self.index, "dead_ratio", 0.0), 4),
            },
            "index_drift": index_drift,
            "pending_compaction": get_meta(self.conn, "compaction_due") == "1",
            "reconciled_at": get_meta(self.conn, "reconciled_at"),
            "wal_bytes": files.get("omnidb.db-wal", 0),
            "files": files,
            "cold_start_ms": self._cold_start_ms,
            "took_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        return payload

    @_serialized
    def tenants(self) -> list[dict]:
        """Every tenant that owns at least one row in this home, with live and
        soft-deleted counts and a per-kind breakdown (ADR-08 observability).

        This is the single-tenant-operator view — the CLI and a deployment owner
        inspecting a home before they wire `serve-http`. It is deliberately NOT
        exposed as an MCP tool: an authenticated tenant may only see its own rows
        through `stats(tenant)`, never the roster of other tenants. It is pure
        read-only SQL, so it takes no write lock and persists nothing.
        """
        rows = self.conn.execute(
            "SELECT tenant_id, "
            f"SUM(CASE WHEN {_SELECTIVE} THEN 1 ELSE 0 END) AS live, "
            "SUM(CASE WHEN deleted_ms IS NOT NULL THEN 1 ELSE 0 END) AS soft_deleted, "
            "SUM(CASE WHEN deleted_ms IS NULL AND superseded_by IS NOT NULL "
            "THEN 1 ELSE 0 END) AS superseded, "
            "COUNT(*) AS total "
            "FROM memories GROUP BY tenant_id ORDER BY live DESC, tenant_id"
        ).fetchall()
        by_kind: dict[str, dict] = {}
        for r in self.conn.execute(
            f"SELECT tenant_id, kind, COUNT(*) AS n FROM memories "
            f"WHERE {_SELECTIVE} GROUP BY tenant_id, kind"
        ):
            by_kind.setdefault(r["tenant_id"], {})[r["kind"]] = int(r["n"])
        return [
            {
                "tenant": r["tenant_id"],
                "live": int(r["live"]),
                "soft_deleted": int(r["soft_deleted"]),
                "superseded": int(r["superseded"]),
                "total": int(r["total"]),
                "by_kind": by_kind.get(r["tenant_id"], {}),
            }
            for r in rows
        ]

    @_serialized
    def query_stats(self, hours: float = 24.0,
                    tenant: str | None = None) -> dict:
        """Aggregate the query log over a trailing window (F-02).

        Read-only; an empty queries table (log disabled) yields empty
        structures rather than an error.

        `tenant` scopes the aggregation to one tenant's own queries. It is
        optional and defaults to the whole home because the single-tenant
        operator's view is the unscoped one; on a multi-tenant home the caller
        passes a tenant to avoid one tenant seeing another's query text.
        """
        window = max(float(hours), 0.0)
        cutoff = int((self.now_fn() - window * 3600.0) * 1000)
        # Scoped: AND tenant_id = ?. Unscoped: no predicate at all, which is the
        # single-tenant operator's view. Both are bound parameters — the SQL is
        # a fixed string, so the AST splice guard (SEC-A3-10) sees nothing to
        # flag and there is no string-concatenation shape to get wrong.
        params: list = [cutoff, tenant, tenant]
        by_kind = {
            r["kind"]: {
                "count": int(r["n"]),
                "avg_ms": round(r["avg_ms"], 2) if r["avg_ms"] is not None else None,
                "max_ms": round(r["max_ms"], 2) if r["max_ms"] is not None else None,
            }
            for r in self.conn.execute(
                "SELECT kind, COUNT(*) AS n, AVG(took_ms) AS avg_ms, MAX(took_ms) AS max_ms "
                "FROM queries WHERE created_ms >= ? "
                "AND (? IS NULL OR tenant_id = ?) GROUP BY kind",
                params,
            )
        }
        top_queries = [
            {"query": r["query"], "count": int(r["n"])}
            for r in self.conn.execute(
                "SELECT query, COUNT(*) AS n FROM queries "
                "WHERE created_ms >= ? AND (? IS NULL OR tenant_id = ?) "
                "GROUP BY query ORDER BY n DESC, query LIMIT 10",
                params,
            )
        ]
        # COUNT from the same predicate, so `total` cannot drift from by_kind.
        total = int(self.conn.execute(
            "SELECT COUNT(*) FROM queries WHERE created_ms >= ? "
            "AND (? IS NULL OR tenant_id = ?)",
            params,
        ).fetchone()[0])
        return {"window_hours": window, "tenant": tenant, "total": total,
                "by_kind": by_kind, "top_queries": top_queries}

    @_serialized
    def schema(self) -> dict:
        """Introspection for the `search` escape hatch (E-06): tables, notes,
        and example queries. Every example is a statement the read-only
        authorizer actually accepts, so agents can copy-run them."""
        tables = [
            {"name": str(r["name"]), "sql": str(r["sql"])}
            for r in self.conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT LIKE 'memory_fts_%' ORDER BY name"
            )
        ]
        live = "deleted_ms IS NULL AND superseded_by IS NULL"
        return {
            "schema_version": SCHEMA_VERSION,
            "tables": tables,
            "notes": {
                "memories": (
                    "one row per memory; kind is episodic|semantic|procedural; "
                    "metadata and trigger_patterns are JSON strings; soft "
                    f"delete via deleted_ms, supersession via superseded_by; "
                    f"live rows satisfy: {live}"
                ),
                "memory_fts": (
                    "FTS5 index over memories.content (porter unicode61); "
                    "query with memory_fts MATCH ?"
                ),
                "edges": (
                    "graph links between memory ids; PK (src, dst, label); "
                    "FK cascade on memory delete"
                ),
                "meta": "internal key/value state (embedder identity, index flags)",
                "queries": (
                    "opt-in query log (config query_log); aggregate with "
                    "`omnidb-server query-stats`"
                ),
            },
            "examples": [
                f"SELECT id, kind, substr(content, 1, 80) AS preview FROM memories "
                f"WHERE {live} ORDER BY created_ms DESC LIMIT 10",
                f"SELECT kind, COUNT(*) AS n FROM memories WHERE {live} GROUP BY kind",
                "SELECT rowid, bm25(memory_fts) AS rank FROM memory_fts "
                "WHERE memory_fts MATCH 'query' ORDER BY rank LIMIT 10",
                "SELECT e.src, e.dst, e.label FROM edges e LIMIT 20",
                "SELECT key, value FROM meta LIMIT 20",
                "SELECT kind, COUNT(*) AS n, AVG(took_ms) AS avg_ms FROM queries "
                "GROUP BY kind",
            ],
        }

    def _home(self) -> Path:
        return Path(self.conn.execute("PRAGMA database_list").fetchone()[2]).parent

    @_serialized
    def close(self) -> None:
        try:
            # AUDIT-I02: a read-only diagnostic store must never flush.
            if not self._read_only:
                self.flush()
                # A PASSIVE checkpoint is permitted to no-op and never
                # truncates, so the WAL keeps a complete plaintext copy of
                # everything written this session — including rows an operator
                # purged and believes are erased. TRUNCATE resets the file to
                # zero length; it fails harmlessly if another connection is
                # still reading the home, which is the same case PASSIVE
                # tolerated (audit R6/erasure).
                try:
                    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.OperationalError:
                    pass  # a concurrent reader blocks it; nothing to do
        finally:
            self.conn.close()


def _classify_sqlite_error(exc: sqlite3.DatabaseError) -> tuple[str, str]:
    msg = str(exc)
    if "not authorized" in msg:
        return "statement rejected: only read-only queries are allowed", E_READ_ONLY
    if "interrupted" in msg:
        return f"query aborted: {msg}", E_TIMEOUT
    if "no such table" in msg:
        return f"query aborted: {msg}", E_TABLE_NOT_FOUND
    if "no such column" in msg:
        return f"query aborted: {msg}", E_COLUMN_NOT_FOUND
    return f"query aborted: {msg}", E_INTERNAL
