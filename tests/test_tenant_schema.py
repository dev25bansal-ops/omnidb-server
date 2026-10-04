"""Per-tenant FTS5 table lifecycle (ADR-08 storage appendix).

The property under test is the crash-safety contract: a tenant's FTS table and
its registration marker exist or are absent **together** — never one without the
other — and the marker, not the table, is the reachability criterion.
"""

from __future__ import annotations

import pytest

from omnidb_server.db import connect
from omnidb_server.tenancy import TenantError
from omnidb_server.tenant_schema import (
    drop_tenant,
    fts_table_name,
    is_registered,
    list_registered,
    parse_marker,
    register_and_create,
)


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path)
    yield c
    c.close()


# --- creation is idempotent and atomic ------------------------------------

def test_first_create_returns_true_and_registers(conn):
    assert register_and_create(conn, "acme") is True
    assert is_registered(conn, "acme")
    assert list_registered(conn) == ["acme"]


def test_second_create_returns_false(conn):
    register_and_create(conn, "acme")
    assert register_and_create(conn, "acme") is False


def test_registered_table_is_usable_for_matching(conn):
    register_and_create(conn, "acme")
    table = fts_table_name("acme")
    conn.execute(f"INSERT INTO {table}(rowid, content) VALUES (1, 'alpha beta')")
    row = conn.execute(
        f"SELECT rowid FROM {table} WHERE {table} MATCH ?", ('"alpha"',)
    ).fetchone()
    assert row is not None and row[0] == 1


def test_table_name_is_quoted_and_validated():
    assert fts_table_name("acme") == '"fts_acme"'
    # a hyphenated tenant is a legal identifier only under double quotes
    assert fts_table_name("acme-prod") == '"fts_acme-prod"'
    with pytest.raises(TenantError):
        fts_table_name("acme/prod")  # traversal
    with pytest.raises(TenantError):
        fts_table_name('acme"prod')  # would break the quoting


# --- the crash-safety contract --------------------------------------------

def test_dropping_removes_table_and_marker_together(conn):
    register_and_create(conn, "acme")
    drop_tenant(conn, "acme")
    assert not is_registered(conn, "acme")
    assert list_registered(conn) == []
    # the table name is gone too
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'fts_acme'"
    ).fetchone() is None


def test_dropping_an_unknown_tenant_is_a_no_op(conn):
    drop_tenant(conn, "never-existed")  # must not raise
    assert list_registered(conn) == []


def test_manually_torn_marker_without_table_is_detectable(conn):
    """Simulate the impossible state to prove the reachability distinction.

    If a marker existed without its table, the write path would submit to a
    missing table. The contract is that is_registered() is the criterion, so
    this test documents what a torn marker looks like and that a repair path
    (drop + recreate) restores consistency.
    """
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('tenant_fts:acme', '1')"
    )
    conn.commit()
    assert is_registered(conn, "acme")          # marker present
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'fts_acme'"
    ).fetchone() is None                        # but no table — the torn state

    # repair: drop removes the marker, create re-establishes both atomically
    drop_tenant(conn, "acme")
    register_and_create(conn, "acme")
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'fts_acme'"
    ).fetchone() is not None


def test_parse_marker_round_trips():
    assert parse_marker("tenant_fts:acme") == "acme"
    assert parse_marker("other_key") is None
    assert parse_marker("tenant_fts:a-b_c") == "a-b_c"


def test_list_registered_is_empty_on_a_fresh_home(conn):
    assert list_registered(conn) == []


def test_multiple_tenants_do_not_cross(conn):
    register_and_create(conn, "acme")
    register_and_create(conn, "globex")
    assert sorted(list_registered(conn)) == ["acme", "globex"]
    # matches in one tenant's table are invisible in the other
    conn.execute('INSERT INTO "fts_acme"(rowid, content) VALUES (1, "alpha")')
    assert conn.execute(
        'SELECT rowid FROM "fts_globex" WHERE "fts_globex" MATCH ?', ('"alpha"',)
    ).fetchone() is None