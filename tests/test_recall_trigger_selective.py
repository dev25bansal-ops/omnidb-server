"""F24 - the trigger detail fetch dropped _SELECTIVE, resurrecting deleted rows.

A row soft-deleted between the trigger scan and the detail fetch was pulled
back into results: a deleted memory returned by recall. The two sibling queries
carried _SELECTIVE; this one did not.

It is a time-of-check/time-of-use race, so the test injects it rather than hoping
for it: the connection is wrapped so the row is soft-deleted the instant the
trigger scan returns, which is exactly the window the missing predicate left open.

ISOLATION. The row is inserted with raw SQL, so it is absent from the vector
index and from the FTS table. The trigger scan reads memories directly, so it is
the ONLY leg that can return this row - without that, the dense leg returns it
before the race window even opens and the test would prove nothing.
"""

from __future__ import annotations

import json

from conftest import make_store

from omnidb_server.recall import run_recall

_CONTENT = "restore the previous release when the canary fails"
_PATTERNS = json.dumps([r"emergency rollback runbook"])
_QUERY = "walk me through the emergency rollback runbook"


def _run(st, kind):
    return run_recall(st, _QUERY, 5, kind, 5, None, None)


def _insert_trigger_only(st):
    """A procedural row with a trigger that NO other leg can reach."""
    cur = st.conn.execute(
        "INSERT INTO memories (kind, content, metadata, created_ms, "
        "access_count, trigger_patterns, tenant_id) "
        "VALUES ('procedural', ?, '{}', ?, 0, ?, 'default')",
        (_CONTENT, int(st.now_fn() * 1000), _PATTERNS),
    )
    st.conn.commit()
    return int(cur.lastrowid)


class _RaceConn:
    """Delegates everything; fires the callback right after the trigger scan."""

    def __init__(self, real, after_trigger_scan):
        self._real = real
        self._after = after_trigger_scan
        self._fired = False

    def execute(self, sql, *a, **k):
        res = self._real.execute(sql, *a, **k)
        if not self._fired and "SELECT id, trigger_patterns" in sql:
            self._fired = True
            self._after()
        return res

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_a_row_deleted_after_the_trigger_scan_is_not_returned(tmp_path):
    """THE F24 REGRESSION. Without _SELECTIVE on the detail fetch the
    soft-deleted row is scored into the results anyway."""
    st = make_store(tmp_path)
    mid = _insert_trigger_only(st)
    real = st.conn
    fired = []

    def soft_delete_it():
        real.execute(
            "UPDATE memories SET deleted_ms = ? WHERE id = ?",
            (int(st.now_fn() * 1000) + 1, mid),
        )
        real.commit()
        fired.append(True)

    st.conn = _RaceConn(real, soft_delete_it)
    ids = [r["id"] for r in _run(st, "procedural")]

    assert fired, "the race was never injected, so this proves nothing"
    assert mid not in ids, "a soft-deleted memory was returned by recall"
    st.close()


def test_the_live_triggered_row_is_still_returned(tmp_path):
    """The positive control: the fence must not have made the path dead."""
    st = make_store(tmp_path)
    mid = _insert_trigger_only(st)
    ids = [r["id"] for r in _run(st, "procedural")]
    assert mid in ids
    st.close()


def test_a_trigger_only_row_is_marked_as_a_skill_trigger(tmp_path):
    st = make_store(tmp_path)
    mid = _insert_trigger_only(st)
    row = [r for r in _run(st, "procedural") if r["id"] == mid][0]
    assert row["metadata"].get("skill_trigger") is True
    st.close()


def test_triggers_stay_procedural_only(tmp_path):
    """The audit called the two kind predicates redundant. They are not: they are
    a conjunction, and the hardcoded one is what stops a kind=semantic recall
    from returning procedural trigger rows. This must hold identically before and
    after, so it is the guard on the conjunction."""
    st = make_store(tmp_path)
    mid = _insert_trigger_only(st)
    assert mid in [r["id"] for r in _run(st, "procedural")]
    assert mid in [r["id"] for r in _run(st, None)]
    assert mid not in [r["id"] for r in _run(st, "semantic")]
    assert mid not in [r["id"] for r in _run(st, "episodic")]
    st.close()
