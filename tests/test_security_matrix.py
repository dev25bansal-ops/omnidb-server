import sqlite3

import pytest

from omnidb_server.store import StoreError

# --- analysis 6.5: authorizer deny matrix ----------------------------------------
# Every entry must raise StoreError. Most rejections surface as OMN-020 whether
# caught by the SELECT/WITH prefix gate or by the authorizer underneath; the
# multi-statement entry dies earlier inside sqlite3 itself ("You can only execute
# one statement at a time"), which classifies as OMN-000.
_DENY_CASES = [
    pytest.param("PRAGMA journal_mode", "OMN-020", id="pragma_journal_mode"),
    pytest.param("PRAGMA wal_checkpoint", "OMN-020", id="pragma_wal_checkpoint"),
    pytest.param("ATTACH DATABASE ':memory:' AS aux", "OMN-020", id="attach_memory_db"),
    pytest.param("VACUUM", "OMN-020", id="vacuum"),
    pytest.param("VACUUM INTO 'x.db'", "OMN-020", id="vacuum_into"),
    pytest.param("INSERT INTO meta VALUES('k','v')", "OMN-020", id="insert_into_meta"),
    pytest.param(
        "WITH t AS (SELECT 1) DELETE FROM memories", "OMN-020",
        id="data_modifying_cte_delete",
    ),
    pytest.param(
        "WITH t AS (SELECT 1) INSERT INTO edges SELECT 1,1,'x'", "OMN-020",
        id="data_modifying_cte_insert",
    ),
    pytest.param(
        "-- SELECT 1;\nDELETE FROM memories", "OMN-020",
        id="leading_comment_disguise",
    ),
    pytest.param(
        "SELECT 1; DROP TABLE memories", "OMN-000",
        id="multi_statement_trailing_drop",
    ),
    pytest.param("CREATE TABLE t(x)", "OMN-020", id="create_table"),
    pytest.param("DROP TABLE memories", "OMN-020", id="drop_table"),
    pytest.param("REPLACE INTO meta VALUES('a','b')", "OMN-020", id="replace_into_meta"),
]


@pytest.mark.parametrize(("sql", "expected_code"), _DENY_CASES)
def test_denied_sql_rejected_with_code(store, sql, expected_code):
    store.remember("sentinel row that must survive")
    with pytest.raises(StoreError) as exc:
        store.search(sql)
    assert exc.value.code == expected_code
    assert store.search("SELECT COUNT(*) FROM memories")["rows"] == [[1]]


# --- analysis 6.5: authorizer allow matrix ---------------------------------------
# A seeded memory ("alpha beta gamma payload") becomes id 1 in the fresh fixture,
# so the FTS leg can pin its rowid; first_row=None entries only assert that the
# statement executes cleanly.
_ALLOW_CASES = [
    pytest.param("SELECT 1", [[1]], id="constant_select"),
    pytest.param("SELECT * FROM memories LIMIT 1", None, id="memory_table_select"),
    pytest.param(
        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'alpha'", [[1]],
        id="fts_match_query",
    ),
    pytest.param("SELECT * FROM pragma_data_version", None, id="pragma_data_version_via_tvf"),
    pytest.param(
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5) "
        "SELECT sum(x) FROM n",
        [[15]],
        id="recursive_cte_select",
    ),
]


@pytest.mark.parametrize(("sql", "first_row"), _ALLOW_CASES)
def test_allowed_sql_executes(store, sql, first_row):
    store.remember("alpha beta gamma payload")
    out = store.search(sql)
    assert out["truncated"] is False
    assert out["rows"]
    if first_row is not None:
        assert out["rows"] == first_row


def test_pragma_data_version_allowance_scope(store):
    # The authorizer's lone PRAGMA carve-out (data_version) exists because FTS5
    # internals issue that pragma during MATCH queries — the fts_match_query
    # entry proving that path stays open is the real regression guard. Bare
    # "PRAGMA data_version" is still stopped by the statement-type gate, so the
    # read-only surface for it is the equivalent table-valued form below, which
    # must yield exactly one row.
    store.remember("alpha beta gamma payload")
    matched = store.search("SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'alpha'")
    assert matched["rows"] == [[1]]
    out = store.search("SELECT * FROM pragma_data_version")
    assert out["columns"] == ["data_version"]
    assert len(out["rows"]) == 1


def test_extension_loading_fails_closed(store):
    # The authorizer permits SQLITE_FUNCTION by design, so blocking
    # load_extension() relies on SQLite's runtime disablement, not the action
    # table. CPython exposes the enable_load_extension toggle hook (some builds
    # omit it entirely); either state must fail closed.
    conn = store.conn
    if getattr(conn, "enable_load_extension", None) is None:
        pytest.skip("this sqlite3 build offers no extension-loading hook")
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute("SELECT load_extension('omnidb_no_such_extension')")
    with pytest.raises(StoreError) as exc:
        store.search("SELECT load_extension('omnidb_no_such_extension')")
    assert exc.value.code == "OMN-020"


def test_denials_leave_transaction_clean(store):
    store.remember("keeper one")
    store.remember("keeper two")
    attempts = (
        "VACUUM",
        "ATTACH DATABASE ':memory:' AS aux",
        "WITH t AS (SELECT 1) DELETE FROM memories",
        "SELECT 1; DROP TABLE memories",
    )
    for sql in attempts:
        with pytest.raises(StoreError):
            store.search(sql)
    assert store.conn.in_transaction is False
    assert store.search("SELECT COUNT(*) FROM memories")["rows"] == [[2]]
    hits = store.recall("keeper two", k=2)
    assert hits[0]["content"] == "keeper two"
