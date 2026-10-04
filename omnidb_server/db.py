from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 6

# The shadow tables FTS5 creates alongside every virtual table. DROP TABLE on
# the virtual table removes only the table itself and orphans these, after which
# CREATE VIRTUAL TABLE with the same name fails — so v6 drops them explicitly
# before recreating a tenant's index.
#
# Module-level and literal so the AST splice guard (SEC-A3-10) can verify every
# identifier built from it; a typo here would otherwise silently orphan a shadow
# table and only surface as an unopenable home.
_FTS_SHADOW_SUFFIXES = ("_content", "_data", "_idx", "_docsize", "_config")

# Migration 1 is the base schema. Later schema changes append here as
# (version, ddl) pairs; PRAGMA user_version tracks the applied head.
_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY,
    kind             TEXT NOT NULL DEFAULT 'semantic'
                     CHECK (kind IN ('episodic', 'semantic', 'procedural')),
    content          TEXT NOT NULL,
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_ms       INTEGER NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_ms INTEGER,
    deleted_ms       INTEGER,
    superseded_by    INTEGER REFERENCES memories(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS edges (
    src        INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst        INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (src, dst, label)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
"""

_MIGRATION_2 = """
ALTER TABLE memories ADD COLUMN agent_id TEXT;
ALTER TABLE memories ADD COLUMN trigger_patterns TEXT;
CREATE INDEX IF NOT EXISTS memories_agent_idx ON memories(agent_id);
"""

_MIGRATION_3 = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT,
    kind TEXT NOT NULL,
    query TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    took_ms REAL NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS queries_created_idx ON queries(created_ms);
"""

# ADR-08: every row is owned by a tenant. The column defaults to the single
# tenant so pre-tenant homes keep working unchanged; the boundary is added by
# W3-4, which makes reads and writes tenant-scoped. NOT NULL because a row
# with no owner is the current "visible to everyone" defect wearing a new
# column name.
_MIGRATION_4 = """
ALTER TABLE memories ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE queries ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS memories_tenant_idx ON memories(tenant_id);
"""

# ADR-08 / W3-3: FTS moves out of the trigger/external-content mechanism into
# explicit, in-transaction sync by the store write path. The three triggers are
# dropped, the external-content table (which cannot be manually INSERTed into)
# is replaced by a standalone FTS5 table, and every row is back-filled. The
# back-fill covers soft-deleted rows too, matching the trigger semantics: the
# triggers only removed rows on hard DELETE, and soft-deleted rows stayed
# searchable and were filtered by the `_SELECTIVE` join.
_MIGRATION_5 = """
DROP TRIGGER IF EXISTS memories_ai;
DROP TRIGGER IF EXISTS memories_ad;
DROP TRIGGER IF EXISTS memories_au;
DROP TABLE IF EXISTS memory_fts;
CREATE VIRTUAL TABLE memory_fts USING fts5(content, tokenize='porter unicode61');
INSERT INTO memory_fts(rowid, content) SELECT id, content FROM memories;
"""

# ADR-08 / W3-3: one FTS5 table per tenant. The data movement is NOT in this
# string — it needs one statement per tenant, with a runtime table name, which
# is exactly the interpolation surface the v4 comment warns about. So the DDL
# here only drops the shared table, and `_migrate` calls
# `_populate_per_tenant_fts` (below) inside the same transaction to create and
# fill each tenant's table, then verifies before publishing v6.
_MIGRATION_6 = """
DROP TABLE IF EXISTS memory_fts;
"""


def _populate_per_tenant_fts(conn: sqlite3.Connection) -> None:
    """Create and fill one FTS table per tenant. Runs inside v6's transaction.

    Every tenant present in `memories` gets a table, including `default` — which
    holds every pre-tenant row and every `tenant="default"` default argument, so
    a home that predates tenancy still migrates into something searchable.

    Soft-deleted rows are indexed too, deliberately and matching migration 5:
    `forget` replaces a posting (delete old, insert new) so the vector sidecar
    stays consistent, and dropping soft-deleted rows from FTS would change what
    that does silently.
    """
    from .tenant_schema import fts_table_name

    tenants = [r[0] for r in conn.execute(
        "SELECT DISTINCT tenant_id FROM memories ORDER BY tenant_id")]
    for tenant in tenants:
        # Deliberately not register_and_create: it opens its own transaction and
        # commits, which would end the migration's. Its atomicity (marker and
        # CREATE together) is supplied by the caller's transaction instead.
        #
        # DROP before CREATE, not IF NOT EXISTS, because this migration must be
        # re-runnable. A home can arrive at v6 with per-tenant tables already
        # present — a torn upgrade, a `user_version` rewound for recovery, or a
        # build that already shipped v6 — and inserting the same rowids into a
        # populated FTS table raises a constraint failure partway through, so
        # the whole migration rolls back and the home cannot open. Rebuilding
        # makes the result a function of the rows rather than of the history,
        # which is the same property migration 5 has.
        bare = f"fts_{tenant}"  # fts_table_name returns this quoted
        table = fts_table_name(tenant)
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        # Shadow tables are dropped explicitly: DROP TABLE on the virtual table
        # removes only the table itself, and a later CREATE VIRTUAL TABLE with
        # the same name fails on the orphans — which is how a home reaching v6
        # with a leftover index becomes unopenable rather than merely stale.
        #
        # Written out one statement per suffix rather than looped, so every
        # identifier spliced into SQL is literal at the point of use and the
        # AST splice guard (SEC-A3-10) can verify each one. A typo here would
        # otherwise orphan a shadow table silently.
        conn.execute(f'DROP TABLE IF EXISTS "{bare}_content"')
        conn.execute(f'DROP TABLE IF EXISTS "{bare}_data"')
        conn.execute(f'DROP TABLE IF EXISTS "{bare}_idx"')
        conn.execute(f'DROP TABLE IF EXISTS "{bare}_docsize"')
        conn.execute(f'DROP TABLE IF EXISTS "{bare}_config"')
        conn.execute(
            f"CREATE VIRTUAL TABLE {table} USING fts5("
            f"content, tokenize='porter unicode61')"
        )
        conn.execute(
            f"INSERT INTO {table}(rowid, content) "
            f"SELECT id, content FROM memories WHERE tenant_id = ?",
            (tenant,),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, '1') "
            "ON CONFLICT(key) DO NOTHING",
            (f"tenant_fts:{tenant}",),
        )


def _verify_per_tenant_fts(conn: sqlite3.Connection) -> None:
    """Never publish v6 with a tenant's postings missing or short.

    Same contract as the v2 and v4 postconditions: the version bump is the
    moment the schema is declared good, so it must not happen until every
    tenant with rows has a table holding exactly those rows.
    """
    from .tenant_schema import fts_table_name

    for tenant, expected in conn.execute(
        "SELECT tenant_id, COUNT(*) FROM memories GROUP BY tenant_id"
    ):
        table = fts_table_name(tenant)
        try:
            actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError as exc:
            raise DbError(
                f"v6 schema missing FTS table for tenant {tenant!r}: {exc}"
            ) from exc
        if actual != expected:
            raise DbError(
                f"v6 FTS table for tenant {tenant!r} holds {actual} rows, "
                f"expected {expected}"
            )
        marker = conn.execute(
            "SELECT 1 FROM meta WHERE key = ?", (f"tenant_fts:{tenant}",)
        ).fetchone()
        if marker is None:
            raise DbError(f"v6 schema missing tenant_fts marker for {tenant!r}")

    # The whole point: the shared table must be gone, or two lexical indexes
    # exist and only one is written.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'memory_fts'"
    ).fetchone() is not None:
        raise DbError("v6 schema still contains the shared memory_fts table")


MIGRATIONS: list[tuple[int, str]] = [
    (1, _MIGRATION_1), (2, _MIGRATION_2), (3, _MIGRATION_3), (4, _MIGRATION_4),
    (5, _MIGRATION_5), (6, _MIGRATION_6),
]


class DbError(RuntimeError):
    pass


def connect(home: Path) -> sqlite3.Connection:
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "omnidb.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        # AUDIT-I17: reject unsupported homes before persistent PRAGMA changes.
        _checked_schema_version(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        # Zero freed pages instead of leaving the old bytes in place until the
        # page is reused. `purge` is the operation that promises erasure, and
        # without this a freed page keeps the deleted content readable to
        # anything that opens the file. Costs a memset per deleted page, which
        # is negligible against the embeddings a memory carries.
        conn.execute("PRAGMA secure_delete=ON")
        _migrate(conn)
        try:
            # Probe FTS5 support against a throwaway table: the shipped indexes
            # are per-tenant (v6) and there may be no tenant yet on a fresh
            # home, so `memory_fts` no longer exists to probe.
            conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(content)")
            conn.execute("DROP TABLE _fts5_probe")
        except sqlite3.OperationalError as exc:
            raise DbError(f"FTS5 is not available in this SQLite build: {exc}") from exc
        return conn
    except BaseException:
        conn.close()
        raise


def _checked_schema_version(conn: sqlite3.Connection) -> int:
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    supported = max(version for version, _ in MIGRATIONS)
    if current > supported:
        raise DbError(
            f"unsupported database schema v{current}; this OmniDB supports up to "
            f"v{supported}. Upgrade OmniDB before opening this home."
        )
    return current


def _migrate(conn: sqlite3.Connection) -> None:
    current = _checked_schema_version(conn)
    for version, ddl in MIGRATIONS:
        if version <= current:
            continue
        if version != current + 1:
            raise DbError(
                f"migration gap: database at v{current}, next migration is v{version}"
            )
        if version == 2:
            ddl = _heal_torn_v2(conn, ddl)
        elif version == 4:
            ddl = _heal_torn_v4(conn, ddl)
        # Keep DDL, verified postconditions, and the version bump in one
        # transaction. executescript supplies BEGIN but leaves COMMIT to us.
        try:
            conn.executescript(f"BEGIN;{ddl}")
            if version == 2:
                # AUDIT-I10: never publish v2 while either column is missing.
                columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
                missing = {"agent_id", "trigger_patterns"} - columns
                if missing:
                    raise DbError(f"v2 schema missing required columns: {sorted(missing)}")
            elif version == 4:
                # ADR-08: never publish v4 while tenant_id is missing on either
                # tenant-owned table. Table names are literals, not loop
                # variables — a runtime table name here would be an injection
                # surface and is provably unnecessary with two tables.
                mem_cols = {row[1] for row in
                            conn.execute("PRAGMA table_info(memories)")}
                qry_cols = {row[1] for row in
                            conn.execute("PRAGMA table_info(queries)")}
                if "tenant_id" not in mem_cols:
                    raise DbError("v4 schema missing tenant_id on memories")
                if "tenant_id" not in qry_cols:
                    raise DbError("v4 schema missing tenant_id on queries")
            elif version == 6:
                # Per-tenant FTS needs a runtime table name per tenant, so the
                # data movement is Python rather than a DDL string. It runs
                # inside this transaction: parameterized inserts between the
                # BEGIN executescript opened and the commit below are part of it.
                _populate_per_tenant_fts(conn)
                _verify_per_tenant_fts(conn)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()
        except (sqlite3.Error, DbError) as exc:
            conn.rollback()
            raise DbError(f"migration to v{version} failed: {exc}") from exc
        current = version


def _heal_torn_v2(conn: sqlite3.Connection, ddl: str) -> str:
    """AUDIT-I10: repair each independently applied v2 column addition.

    Pre-atomic migrations can leave either or both columns present while
    user_version remains 1. Skip only additions that would duplicate a column.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    for column in ("agent_id", "trigger_patterns"):
        if column in columns:
            ddl = ddl.replace(f"ALTER TABLE memories ADD COLUMN {column} TEXT;", "")
    return ddl


def _heal_torn_v4(conn: sqlite3.Connection, ddl: str) -> str:
    """ADR-08: repair an independently applied v4 tenant_id addition.

    The same tear class as v2, but the column is added to two tables, so an
    interrupted v4 can be missing the column on either. Each already-present
    column's ADD is skipped; the postcondition in _migrate then verifies both
    are present before publishing user_version=4.
    """
    mem_cols = {row[1] for row in
                conn.execute("PRAGMA table_info(memories)")}
    qry_cols = {row[1] for row in
                conn.execute("PRAGMA table_info(queries)")}
    if "tenant_id" in mem_cols:
        ddl = ddl.replace(
            "ALTER TABLE memories ADD COLUMN tenant_id "
            "TEXT NOT NULL DEFAULT 'default';", "")
    if "tenant_id" in qry_cols:
        ddl = ddl.replace(
            "ALTER TABLE queries ADD COLUMN tenant_id "
            "TEXT NOT NULL DEFAULT 'default';", "")
    return ddl


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()
