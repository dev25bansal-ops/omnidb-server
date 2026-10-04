"""AUDIT-I10/I17 regression tests: torn-v2 repair and future-schema rejection.

All fixtures are disposable homes built by hand (no shared state); unsupported
homes are never modified destructively — each test asserts the failed open
leaves the marker and file untouched.
"""

import sqlite3

import pytest

from omnidb_server import db as dbmod

_V2_COLUMNS = {"agent_id", "trigger_patterns"}


def _make_v1_home(home, extra_columns=()):
    """Hand-build a version-1 database with one data row, then add any
    subset of the v2 columns to simulate a torn pre-atomic upgrade."""
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "omnidb.db")
    conn.executescript(dbmod.MIGRATIONS[0][1])
    conn.execute(
        "INSERT INTO memories(id, kind, content, metadata, created_ms) "
        "VALUES (4, 'semantic', 'torn fixture row', '{}', 12345)"
    )
    for column in extra_columns:
        conn.execute(f"ALTER TABLE memories ADD COLUMN {column} TEXT")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def _columns(conn):
    return {row[1] for row in conn.execute("PRAGMA table_info(memories)")}


# --- AUDIT-I10: each torn v2 column is repaired independently -----------------

def test_torn_v2_agent_id_only_still_adds_trigger_patterns(tmp_path):
    # The audit's core case: agent_id exists, trigger_patterns does not.
    # The old all-or-nothing heal left trigger_patterns missing forever.
    _make_v1_home(tmp_path, extra_columns=("agent_id",))
    conn = dbmod.connect(tmp_path)
    assert _V2_COLUMNS <= _columns(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    row = conn.execute(
        "SELECT id, content, agent_id FROM memories WHERE id = 4"
    ).fetchone()
    assert tuple(row) == (4, "torn fixture row", None)
    conn.close()


def test_torn_v2_trigger_patterns_only_still_adds_agent_id(tmp_path):
    _make_v1_home(tmp_path, extra_columns=("trigger_patterns",))
    conn = dbmod.connect(tmp_path)
    assert _V2_COLUMNS <= _columns(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    conn.close()


def test_torn_v2_both_columns_present_heals_without_duplicate(tmp_path):
    _make_v1_home(tmp_path, extra_columns=_V2_COLUMNS)
    conn = dbmod.connect(tmp_path)
    assert _V2_COLUMNS <= _columns(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    conn.close()


def test_normal_v1_upgrade_adds_both_columns(tmp_path):
    _make_v1_home(tmp_path)
    conn = dbmod.connect(tmp_path)
    assert _V2_COLUMNS <= _columns(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    # the pre-migration row survived and FTS stayed synced via the v1 triggers
    hit = conn.execute(
        'SELECT rowid FROM "fts_default" WHERE "fts_default" MATCH ?',
        ("fixture",),
    ).fetchone()
    assert hit is not None and hit[0] == 4
    conn.close()


def test_missing_v2_postcondition_blocks_version_bump(tmp_path, monkeypatch):
    # A heal that only restores one column must fail the postcondition check
    # BEFORE the version bump: DDL and marker roll back, nothing published.
    _make_v1_home(tmp_path)
    monkeypatch.setattr(
        dbmod, "_heal_torn_v2",
        lambda conn, ddl: "ALTER TABLE memories ADD COLUMN agent_id TEXT;",
    )
    with pytest.raises(dbmod.DbError, match="trigger_patterns"):
        dbmod.connect(tmp_path)
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "agent_id" not in _columns(conn)  # transaction rolled back
    conn.close()


# --- AUDIT-I17: future schema versions are rejected before initialization -----

def test_current_version_home_opens(tmp_path):
    _make_v1_home(tmp_path)
    conn = dbmod.connect(tmp_path)
    conn.close()
    conn = dbmod.connect(tmp_path)  # reopen at head: idempotent, no error
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    conn.close()


def test_future_schema_version_rejected_before_initialization(tmp_path):
    _make_v1_home(tmp_path)
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    conn.execute(f"PRAGMA user_version = {dbmod.SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(dbmod.DbError, match="unsupported database schema"):
        dbmod.connect(tmp_path)

    # the unsupported home was not modified destructively: marker unchanged
    # and journal_mode was never flipped to WAL (rejection precedes PRAGMAs)
    probe = sqlite3.connect(tmp_path / "omnidb.db")
    try:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == (
            dbmod.SCHEMA_VERSION + 1
        )
        assert probe.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    finally:
        probe.close()
