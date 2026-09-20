from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 3

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

MIGRATIONS: list[tuple[int, str]] = [(1, _MIGRATION_1), (2, _MIGRATION_2), (3, _MIGRATION_3)]


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
        _migrate(conn)
        try:
            conn.execute("SELECT 1 FROM memory_fts LIMIT 0").fetchone()
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
