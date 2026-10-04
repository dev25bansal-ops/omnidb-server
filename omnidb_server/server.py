from __future__ import annotations

import functools
from contextlib import contextmanager
from pathlib import Path
import sys

# Tools run on anyio worker threads: sync store operations would otherwise
# execute on the event loop and block pings/other requests for their full
# duration (a slow LLM consolidation could stall the whole server).
import anyio

from .store import E_CONSTRAINT, E_NOT_FOUND, MemoryStore, StoreError
from .tenancy import DEFAULT_TENANT


async def _call(fn, /, *args, **kwargs):
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def _stdio_tenant() -> str:
    """The tenant for a stdio request (ADR-08).

    stdio carries no headers: one process, one pipe, one identity, so there is no
    token to read and every call belongs to the deployment's single tenant. This is
    the only path that may resolve without a credential.

    The streamable-HTTP transport passes `http_server.current_tenant` instead, which
    raises rather than defaulting. The distinction is load-bearing: a shared
    fail-open default here would hand every HTTP request the default tenant.
    """
    return DEFAULT_TENANT


def attach_tools(mcp, store, tenant_resolver=None, *, unscoped_search=False):
    """Register OmniDB's tool surface on an MCP server.

    Shared by the stdio transport (`create_mcp`) and the streamable-HTTP
    transport (`http_server.run_http`). Keeping one definition is load-bearing:
    a tool added to one transport but not the other is a tool whose tenant
    check is missing on that transport.
    """
    resolve_tenant = tenant_resolver or _stdio_tenant
    # No tenant_resolver means the stdio transport: one process, one pipe, one
    # identity, so the whole home really is a single trust domain and the raw-SQL
    # escape hatch is available. A resolver means a credentialed transport
    # (streamable-HTTP) and the default is fail-closed.
    unscoped_search = unscoped_search or tenant_resolver is None


    @mcp.tool()
    async def remember(content: str, kind: str = "semantic",
                       metadata: dict | None = None,
                       links: list[dict] | None = None,
                       agent_id: str | None = None,
                       trigger_patterns: list[str] | None = None) -> dict:
        """Persist a memory (kind: episodic | semantic | procedural), optionally linking to other memory ids.

        agent_id scopes the memory to one agent (recall with the same agent_id only
        sees it); trigger_patterns are regexes that surface procedural memories as
        skill triggers when a recall query matches.
        """
        return await _call(store.remember, content, kind=kind, metadata=metadata,
                           links=links, agent_id=agent_id,
                           trigger_patterns=trigger_patterns,
                           tenant=resolve_tenant())

    @mcp.tool()
    async def recall(query: str, k: int = 5, kind: str | None = None,
                     k_fetch: int | None = None, agent_id: str | None = None,
                     reranker: str | None = None) -> list[dict]:
        """Hybrid recall over memories: BM25 + dense-vector fusion with recency/frequency rerank.

        agent_id restricts results to memories saved by that agent (None searches all);
        reranker optionally reorders results via an omnidb_server.rerank REGISTRY name.
        """
        return await _call(store.recall, query, k=k, kind=kind, k_fetch=k_fetch,
                           agent_id=agent_id, reranker=reranker,
                           tenant=resolve_tenant())

    @mcp.tool()
    async def search(sql: str) -> dict:
        """Run a read-only SELECT/WITH statement against SQLite (tables: memories, edges, meta)."""
        # `unscoped_search` is a transport capability, not a tenant name: only a
        # caller that declares the whole home one trust domain gets raw SQL. The
        # HTTP transport never sets it, so an authenticated tenant cannot reach
        # an unscoped read regardless of what its id is called.
        return await _call(store.search, sql, tenant=resolve_tenant(),
                           unscoped=unscoped_search)

    @mcp.tool()
    async def schema() -> dict:
        """Introspect the `search` surface: tables, column notes, and copy-runnable example queries."""
        return await _call(store.schema)

    @mcp.tool()
    async def related(memory_id: int, max_hops: int = 2,
                      label: str | None = None) -> list[dict]:
        """Graph walk from a memory through its links (recursive CTE, undirected)."""
        return await _call(store.related, memory_id, max_hops=max_hops,
                           label=label, tenant=resolve_tenant())

    @mcp.tool()
    async def update_memory(memory_id: int, content: str | None = None,
                            metadata: dict | None = None) -> dict:
        """Edit a memory's content (re-embeds and reindexes) and/or merge metadata."""
        return await _call(store.update_memory, memory_id, content=content,
                           metadata=metadata, tenant=resolve_tenant())

    @mcp.tool()
    async def forget(memory_id: int) -> dict:
        """Soft-delete a memory so it never appears in recall again."""
        return await _call(store.forget, memory_id, tenant=resolve_tenant())

    @mcp.tool()
    async def forget_many(memory_ids: list[int]) -> dict:
        """Soft-delete several memories in one call; reports ids that were missing."""
        return await _call(store.forget_many, memory_ids,
                           tenant=resolve_tenant())

    @mcp.tool()
    async def consolidate(hours: float = 24.0, dry_run: bool = False) -> dict:
        """Distill recent episodic memories into semantic ones (uses configured LLM, else offline stub).

        Episodes are extracted in chunks; a failed chunk retries alone on the
        next run while the others commit. dry_run=True reports what would be
        written without storing anything (still spends LLM calls when one is
        configured).

        Scoped to the calling tenant: the episodes read, the facts written and
        the rows marked consumed all carry that tenant, so a run never distils
        one tenant into another or sends another tenant's transcript to the
        LLM endpoint."""
        from .consolidate import run_consolidation

        return await _call(run_consolidation, store, hours, dry_run,
                           tenant=resolve_tenant())

    @mcp.tool()
    async def stats() -> dict:
        """Database statistics: counts by kind, edges, embedder/index info, file sizes."""
        return await _call(store.stats, tenant=resolve_tenant())


def create_mcp(store):
    """Build the stdio MCP server.

    Single-tenant: the store runs as one implicit tenant (see
    `omnidb_server.tenancy.single_tenant`), so no credential resolution happens
    on this transport. The tool surface is identical to the HTTP transport.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("omnidb")
    attach_tools(mcp, store, unscoped_search=True)
    return mcp


def build_stack(cfg=None):
    """Open config -> connection -> embedder -> index -> MemoryStore.

    AUDIT-I02: callers that serve or mutate a home must go through
    `open_owned_store` so the home lock is held BEFORE this function runs;
    a writable stack must never be built without ownership."""
    from . import db as dbmod
    from .config import Config
    from .db import get_meta
    from .embeddings import make_embedder
    from .vector_index import open_vector_index

    cfg = cfg or Config.load()
    conn = dbmod.connect(cfg.home)
    embedder = make_embedder(cfg)
    stored_kind = get_meta(conn, "vec_kind")
    if cfg.vector_index == "auto":
        index_kind = stored_kind or "auto"
    else:
        index_kind = cfg.vector_index
    count_hint = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    index = open_vector_index(cfg.home, index_kind, embedder.dim, count_hint,
                              ef_search=cfg.ef_search)
    return MemoryStore(conn, embedder, index, cfg)


@contextmanager
def open_owned_store(cfg=None):
    """AUDIT-I02/I20: the single ownership entry point for every writable
    opener (server and CLI).

    The home lock is acquired BEFORE any writable initialization (connection,
    index load, meta writes), so a conflicting opener is refused without ever
    touching mutable state. On every exit path — including a construction
    failure after the lock was taken — the acquired resources are released in
    their own finally blocks: lock release cannot be skipped by a store.close()
    error, and a failed build closes its connection before re-raising."""
    from .config import Config

    cfg = cfg or Config.load()
    lock_fd = _acquire_home_lock(cfg.home)
    if isinstance(lock_fd, OSError):
        raise StoreError(
            f"another process is using {cfg.home}: {lock_fd}", E_CONSTRAINT
        )
    store = None
    try:
        store = build_stack(cfg)
        yield store
    finally:
        try:
            if store is not None:
                store.close()
        finally:
            _release_home_lock(lock_fd)


def open_read_only_store(cfg=None):
    """AUDIT-I02: read-only diagnostics open the home WITHOUT the write lock
    and perform zero persistence — no migrations, no meta writes, no index
    recovery, no flush on close. WAL gives them a consistent snapshot while
    a server holds the home."""
    import sqlite3

    from .config import Config
    from .db import get_meta
    from .embeddings import make_embedder
    from .vector_index import (BruteForceIndex, VectorIndexError,
                            open_vector_index)

    cfg = cfg or Config.load()
    db_path = cfg.home / "omnidb.db"
    if not db_path.exists():
        raise StoreError(f"no OmniDB home at {cfg.home}", E_NOT_FOUND)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    embedder = make_embedder(cfg)
    stored_kind = get_meta(conn, "vec_kind")
    if cfg.vector_index == "auto":
        index_kind = stored_kind or "auto"
    else:
        index_kind = cfg.vector_index
    count_hint = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    try:
        index = open_vector_index(cfg.home, index_kind, embedder.dim, count_hint,
                                  ef_search=cfg.ef_search)
    except VectorIndexError as exc:
        # ARCH-04 made the corrupt-sidecar fallback say why on stderr. This is
        # the last resort ABOVE that, and it used to be a bare `except Exception`
        # that printed nothing - so a TypeError raised inside the index layer and a
        # genuinely unusable index were indistinguishable, and the operator saw a
        # store that looked healthy. Narrowed to the index layer own signal: a
        # programming error now propagates instead of being laundered into a silent
        # downgrade (F13).
        print(
            f"omnidb-server: read-only open could not use the {index_kind!r} "
  f"index ({exc}); falling back to brute force",
            file=sys.stderr,
        )
        index = BruteForceIndex(cfg.home / "vectors.npz", embedder.dim)
    return MemoryStore(conn, embedder, index, cfg, read_only=True)


def _acquire_home_lock(home: Path) -> "int | OSError":
    """Take an exclusive advisory lock on <home>/.lock for the process's
    lifetime: flock on POSIX, an msvcrt byte-range lock on Windows. The fd
    must stay open for the lock to hold; the OS releases it on process death,
    so a SIGKILLed server leaves no stale lock behind (ARCH-05 — the previous
    scheme was a no-op on Windows and bricked POSIX startups after a crash).

    Returns the fd, or the OSError on contention/cannot-open.
    """
    import os as _os
    import sys as _sys

    lock_path = home / ".lock"
    try:
        # AUDIT-I02: the lock is taken before db.connect would create the
        # home, so ensure the directory exists first (creating an empty
        # directory is not mutable-store state and is race-safe).
        home.mkdir(parents=True, exist_ok=True)
        fd = _os.open(str(lock_path), _os.O_RDWR | _os.O_CREAT, 0o600)
    except OSError as exc:
        return exc
    try:
        _os.write(fd, b"0")  # give the locked region a byte on all platforms
        _os.lseek(fd, 0, _os.SEEK_SET)
        if _sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]  # POSIX only
    except OSError as exc:
        _os.close(fd)
        return exc
    return fd


def _release_home_lock(fd: int) -> None:
    import os as _os

    try:
        _os.close(fd)  # closing the fd drops the flock / byte-range lock
    except OSError:
        pass


def serve(home=None) -> int:
    import os as _os
    import signal
    import sys as _sys
    import threading as _threading

    from .config import Config

    cfg = Config.from_env(home)
    # AUDIT-I02: ownership is acquired BEFORE any writable initialization.
    # A rejected second server never builds a stack, so it performs zero
    # persistence — no flush, no sidecar or meta writes on the reject path.
    lock_fd = _acquire_home_lock(cfg.home)
    if isinstance(lock_fd, OSError):
        print(
            f"omnidb-server: error: another process is using {cfg.home}: {lock_fd}",
            file=_sys.stderr,
        )
        return 3
    try:
        store = build_stack(cfg)
    except BaseException:
        _release_home_lock(lock_fd)
        raise
    home = store._home()

    # one-line startup banner on stderr (E-07): the most useful support
    # artifact; stderr is legal on the stdio transport (protocol is stdout)
    print(
        f"omnidb-server: serving home={home} memories={store._live_count()} "
        f"embedder={store.embedder.name} index={store.index.kind} "
        f"cold_start={store._cold_start_ms}ms pid={_os.getpid()}",
        file=_sys.stderr, flush=True,
    )

    # Background maintenance (ARCH-05, AUDIT-I16): checkpoint the WAL and
    # enforce the idle persistence deadline so dirty vectors flush within
    # persist_max_s even after the final client operation.
    _stop_event = _threading.Event()

    def _maintenance_loop() -> None:
        while not _stop_event.wait(1.0):
            store.maintenance_tick()
            store.checkpoint()

    _maintenance_thread = _threading.Thread(
        target=_maintenance_loop, name="store-maintenance", daemon=True)
    _maintenance_thread.start()

    def _graceful_shutdown(signum, frame):  # noqa: ARG001
        _stop_event.set()
        store.flush()
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _graceful_shutdown)
        except (ValueError, OSError):
            # not the main thread (rare on Windows) or handler not allowed
            pass

    try:
        create_mcp(store).run()
    finally:
        _stop_event.set()
        # AUDIT-I20: lock release lives in its own finally so a raising
        # store.close() can never strand the ownership lock.
        try:
            store.close()
        finally:
            _release_home_lock(lock_fd)
    return 0
