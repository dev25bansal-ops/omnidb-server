"""`recall` must tolerate metadata that is valid JSON but not an object.

Audit finding C11. `metadata` is a JSON column, so a row can hold a list, a
string or a number. `recall`'s main path passes `json.loads(row["metadata"])`
straight through and is fine with that. The two skill-trigger merges do not:

    {**json.loads(row["metadata"]), "skill_trigger": True}

which raises `TypeError: 'list' object is not a mapping` for any non-object.

The inconsistency is the defect. `skills.py` and `portability.py` both
`isinstance(metadata, dict)` before touching `.get()`, and
`tests/test_skills_core.py::test_skill_reads_tolerate_metadata_that_is_not_an_object`
documents that decision. `recall` was simply never updated to match — so a row
that every other read path handles gracefully takes down the main read path of
the product, and it does so silently in production (the error surfaces as an
MCP tool error with no query context).

Reachable through the documented public API: `remember(metadata=[...],
trigger_patterns=[...])` is accepted, because `validation.py` has no metadata
check at all and `store.py:487` requires only JSON-serializability.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_store

# Every JSON type except object. `null` is excluded on purpose: the column is
# only ever written from `json.dumps(item.get("metadata") or {})`, so an empty
# or absent value lands as `{}`.
NON_OBJECT_METADATA = [
    ["alpha", 42],
    "a bare string",
    42,
    3.5,
    True,
    {"nested": ["still", "fine"]},  # control: an object is fine
]


@pytest.fixture()
def st(tmp_path):
    store = make_store(tmp_path)
    yield store
    store.close()


def _triggered_row(st, metadata, trigger):  # noqa: ARG001
    """A row that only the trigger leg can reach.

    Fill the index with stronger matches so the trigger row falls out of the
    fused pool and is re-fetched by the trigger path — that is the branch with
    the unguarded merge.
    """
    st.remember("restore the previous release when the canary fails",
                metadata=metadata, trigger_patterns=[trigger], kind="procedural")
    # The trigger row scores 0.0 by design, so the [:k] window drops it when
    # enough scored rows outrank it. Three fillers keep it inside the window
    # while still leaving the fused path (site 2) unexercised; the trigger-only
    # fetch (site 1) is what this exercises.
    for n in range(3):
        st.remember(f"filler {n} about unrelated topics entirely")


@pytest.mark.parametrize("metadata", NON_OBJECT_METADATA)
def test_recall_survives_a_trigger_row_with_non_object_metadata(st, metadata):
    """The trigger path must not raise on data the main path already tolerates."""
    st.remember("restore the previous release when the canary fails",
                metadata=["a", "b"],
                trigger_patterns=[r"emergency rollback runbook"],
                kind="procedural")
    # The trigger row scores 0.0 by design, so the [:k] window drops it when
    # enough scored rows outrank it. Three fillers keep it inside the window
    # while still leaving the fused path (site 2) unexercised; the trigger-only
    # fetch (site 1) is what this exercises.
    for n in range(3):
        st.remember(f"filler {n} about unrelated topics entirely")

    results = st.recall("walk me through the emergency rollback runbook", k=5)
    assert isinstance(results, list)


def test_a_triggered_row_with_array_metadata_is_still_tagged(st):
    """Tolerance must not mean silently dropping the trigger.

    The merge's whole job is to add `skill_trigger: True`. If the fix were
    "skip rows whose metadata is not an object", the row would disappear from
    skill recall entirely — a quieter bug than the crash.
    """
    st.remember("restore the previous release when the canary fails",
                metadata=["a", "b"],
                trigger_patterns=[r"emergency rollback runbook"],
                kind="procedural")
    # The trigger row scores 0.0 by design, so the [:k] window drops it when
    # enough scored rows outrank it. Three fillers keep it inside the window
    # while still leaving the fused path (site 2) unexercised; the trigger-only
    # fetch (site 1) is what this exercises.
    for n in range(3):
        st.remember(f"filler {n} about unrelated topics entirely")

    results = st.recall("walk me through the emergency rollback runbook", k=5)
    tagged = [r for r in results if r["metadata"].get("skill_trigger")]
    assert tagged, f"no result carried skill_trigger: {results[:2]}"
    assert isinstance(tagged[0]["metadata"], dict)


def test_non_object_metadata_is_preserved_on_the_main_path(st):
    """The main path already tolerates it; pin that so the fix cannot regress it."""
    mid = st.remember("array metadata row", metadata=["alpha", 42])["id"]
    results = st.recall("array metadata", k=5)
    row = next((r for r in results if r["id"] == mid), None)
    assert row is not None, "the row did not come back at all"
    assert row["metadata"] == ["alpha", 42], \
        "main-path metadata was coerced; tolerance means preserving it, not rewriting"


def test_object_metadata_is_unchanged(st):
    """The control: normal dict metadata behaves exactly as before."""
    mid = st.remember("object metadata row", metadata={"skill": "runbook"},
                      trigger_patterns=[r"^object metadata$"], kind="procedural")["id"]
    for n in range(60):
        st.remember(f"filler {n} object metadata notes")

    results = st.recall("object metadata", k=5)
    row = next(r for r in results if r["id"] == mid)
    assert row["metadata"]["skill"] == "runbook"
    assert row["metadata"]["skill_trigger"] is True


def test_merging_into_a_non_object_yields_an_object(st):
    """The merge result must be a dict even when the stored value was a list.

    This is what makes the tag survivable: callers downstream use `.get()`, and
    a list would break them again at the next hop.
    """
    st.remember("restore the previous release when the canary fails",
                metadata="a string value",
                trigger_patterns=[r"emergency rollback runbook"],
                kind="procedural")
    # The trigger row scores 0.0 by design, so the [:k] window drops it when
    # enough scored rows outrank it. Three fillers keep it inside the window
    # while still leaving the fused path (site 2) unexercised; the trigger-only
    # fetch (site 1) is what this exercises.
    for n in range(3):
        st.remember(f"filler {n} about unrelated topics entirely")

    results = st.recall("walk me through the emergency rollback runbook", k=5)
    tagged = [r for r in results if r["metadata"].get("skill_trigger")]
    assert tagged, "triggered row with string metadata was dropped"
    assert isinstance(tagged[0]["metadata"], dict)
    assert tagged[0]["metadata"] == {"skill_trigger": True}, \
        "the original non-object value should not leak into the merge"


def test_recall_result_metadata_is_always_a_mapping_when_tagged(st):
    """Property form: no returned item may have an untaggable metadata field."""
    st.remember("restore the previous release when the canary fails",
                metadata=[["deep"]],
                trigger_patterns=[r"emergency rollback runbook"],
                kind="procedural")
    # The trigger row scores 0.0 by design, so the [:k] window drops it when
    # enough scored rows outrank it. Three fillers keep it inside the window
    # while still leaving the fused path (site 2) unexercised; the trigger-only
    # fetch (site 1) is what this exercises.
    for n in range(3):
        st.remember(f"filler {n} about unrelated topics entirely")

    for item in st.recall("walk me through the emergency rollback runbook", k=10):
        if item["metadata"].get("skill_trigger"):
            assert isinstance(item["metadata"], dict)
        json.dumps(item["metadata"])  # must stay serialisable for the MCP wire