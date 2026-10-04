"""BUG-A1-08: every `id IN (...)` statement batches on the connection's
variable limit.

The defect: `forget_many` and `purge` built an unbounded placeholder list, so
- a caller passing more ids than SQLITE_LIMIT_VARIABLE_NUMBER (32766 on stock
  builds) got a sqlite3.ProgrammingError instead of a forget, and
- a home holding more purgeable rows than that limit could **never** be purged,
  at any setting, with no diagnostic.

The fix routes both through one `MemoryStore._id_batch()` helper.

These tests lower the connection's variable limit with monkeypatch rather than
creating 32k rows, so the batch boundary is exercised exactly and the suite
stays fast. The large-count behaviour is asserted through the helper's own
arithmetic, which is the part that was wrong.
"""

from __future__ import annotations

import sqlite3

from conftest import make_store

from omnidb_server.store import _ID_BATCH_CAP, _ID_BIND_RESERVE


def _set_var_limit(conn, value: int) -> None:
    """Lower SQLITE_LIMIT_VARIABLE_NUMBER so a handful of ids spans batches."""
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, value)


# --- the helper's arithmetic ------------------------------------------------

def test_id_batch_is_bounded_by_the_cap_and_the_limit(store):
    """The batch never exceeds the cap, and never exceeds limit - reserve."""
    assert store._id_batch() <= _ID_BATCH_CAP
    limit = store.conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    assert store._id_batch() <= limit - _ID_BIND_RESERVE


def test_id_batch_tracks_a_lowered_connection_limit(store, monkeypatch):
    """Lowering the connection's limit lowers the batch, so the helper is
    actually reading the connection and not a hardcoded number."""
    baseline = store._id_batch()
    assert baseline > 1

    # Re-open a store on a connection whose variable limit is tiny.
    def _limited(home, **_kwargs):
        s = make_store(home)
        _set_var_limit(s.conn, _ID_BIND_RESERVE + 3)
        return s

    small = _limited(store.cfg.home)
    try:
        # floor of 1: never returns 0, which would loop forever
        assert small._id_batch() == 3
    finally:
        small.close()


def test_id_batch_never_returns_zero(store, monkeypatch):
    """A limit at or below the reserve must floor at 1, not 0.

    Returning 0 would make `range(0, n, 0)` raise and, worse, a negative or
    zero batch from a future edit would silently skip work instead of failing.
    """
    st = make_store(store.cfg.home)
    try:
        _set_var_limit(st.conn, _ID_BIND_RESERVE)  # limit - reserve == 0
        assert st._id_batch() == 1
    finally:
        st.close()


# --- forget_many batches ----------------------------------------------------

def test_forget_many_spans_multiple_batches(tmp_path, monkeypatch):
    """More ids than one batch still all get forgotten."""
    st = make_store(tmp_path)
    try:
        ids = [r["id"] for r in st.remember_many(
            [{"content": f"batching probe {i}"} for i in range(12)])]
        assert len(ids) == 12

        # Force 3 ids per batch.
        _set_var_limit(st.conn, _ID_BIND_RESERVE + 3)
        assert st._id_batch() == 3

        result = st.forget_many(ids)
        assert sorted(result["forgotten"]) == sorted(ids)
        assert result["missing"] == []
        assert st.stats()["live"] == 0
    finally:
        st.close()


def test_forget_many_reports_missing_ids_across_batches(tmp_path):
    """The missing set is computed against the WHOLE input, not one batch.

    This is the regression a naive batching fix introduces: if `missing` is
    derived per batch and only the last batch's contribution is returned, a
    caller silently loses the not-found report for every earlier batch.
    """
    st = make_store(tmp_path)
    try:
        real = [r["id"] for r in st.remember_many(
            [{"content": f"present {i}"} for i in range(5)])]
        absent = [999001, 999002, 999003]
        _set_var_limit(st.conn, _ID_BIND_RESERVE + 2)

        result = st.forget_many(real + absent)
        assert sorted(result["forgotten"]) == sorted(real)
        assert sorted(result["missing"]) == sorted(absent)
    finally:
        st.close()


def test_forget_many_stamps_every_row_with_one_timestamp(tmp_path):
    """All rows in a batched purge share one deleted_ms value.

    Batching must not call now_fn() per batch: a caller that forgets 40k rows
    would otherwise get 80 distinct timestamps and a stat() report that cannot
    be reasoned about.
    """
    st = make_store(tmp_path)
    try:
        ids = [r["id"] for r in st.remember_many(
            [{"content": f"stamp probe {i}"} for i in range(8)])]
        _set_var_limit(st.conn, _ID_BIND_RESERVE + 2)
        st.forget_many(ids)

        stamps = {r[0] for r in st.conn.execute(
            "SELECT DISTINCT deleted_ms FROM memories "
            "WHERE deleted_ms IS NOT NULL").fetchall()}
        assert len(stamps) == 1, stamps
    finally:
        st.close()


# --- purge batches ----------------------------------------------------------

def test_purge_spans_multiple_batches(tmp_path):
    """A home with more purgeable rows than one batch still purges fully.

    Before the fix this raised and the space was unreclaimable.
    """
    st = make_store(tmp_path)
    try:
        ids = [r["id"] for r in st.remember_many(
            [{"content": f"purge probe {i}"} for i in range(10)])]
        st.forget_many(ids)
        assert st.stats()["live"] == 0

        _set_var_limit(st.conn, _ID_BIND_RESERVE + 2)
        assert st._id_batch() == 2

        result = st.purge()
        assert result["purged"] == 10
        remaining = st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert remaining == 0
    finally:
        st.close()


def test_purge_counts_across_batches(tmp_path):
    """The returned count is the SUM over batches, not the last batch's rowcount.

    A batching fix that leaves `purged = cur.rowcount` outside the loop reports
    2 when it deleted 10, and the operator cannot tell a partial purge from a
    complete one.
    """
    st = make_store(tmp_path)
    try:
        ids = [r["id"] for r in st.remember_many(
            [{"content": f"count probe {i}"} for i in range(9)])]
        st.forget_many(ids)
        _set_var_limit(st.conn, _ID_BIND_RESERVE + 4)  # 4 per batch, 3 batches

        assert st.purge()["purged"] == 9
    finally:
        st.close()


def test_purge_on_an_empty_home_is_a_no_op(tmp_path):
    """Control: nothing to purge is not an error and is not a crash."""
    st = make_store(tmp_path)
    try:
        _set_var_limit(st.conn, _ID_BIND_RESERVE + 1)
        assert st.purge()["purged"] == 0
    finally:
        st.close()


def test_purge_still_nulls_superseded_by_before_deleting(tmp_path):
    """DEBT-01's invariant survives batching.

    superseded_by has no ON DELETE CASCADE, so the reference must be nulled in
    the same transaction as the delete or the FK check fails. Batching must not
    reorder those two operations.
    """
    st = make_store(tmp_path)
    try:
        a = st.remember("superseded target", kind="semantic")["id"]
        b = st.remember("superseding record", kind="semantic")["id"]
        st.supersede(a, b)
        st.forget_many([a, b])

        _set_var_limit(st.conn, _ID_BIND_RESERVE + 1)
        result = st.purge()
        assert result["purged"] == 2
        assert st.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        st.close()
