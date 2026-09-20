"""Consolidation deduplication and supersession semantics.

Dedup: a validated fact that is a near-duplicate (cosine >= 0.92) of a LIVE
semantic memory updates the existing row instead of creating a new one.
Supersede stays a manual store API; recall must hide superseded rows.
"""

import pytest
from conftest import make_store

from omnidb_server.consolidate import run_consolidation
from omnidb_server.store import StoreError


def _seed_batch(st, template: str, n: int = 3):
    for i in range(n):
        st.remember(template.format(i=i), kind="episodic")


def test_second_run_dedups_near_identical_episodes(tmp_path):
    clock = {"t": 1_700_000_000.0}
    st = make_store(tmp_path, now=lambda: clock["t"])
    _seed_batch(st, "deploy runbook {i}: bump the version tag and cut the release branch")
    report1 = run_consolidation(st, hours=24)
    assert report1["facts_created"] == 3
    assert report1["facts_deduplicated"] == 0

    clock["t"] += 48 * 3600  # first batch marked, so this batch is all "fresh"
    # same facts, lightly reworded: cosine ~0.99 through the hashing embedder
    _seed_batch(st, "deploy runbook {i}: bump the version tag and cut the release branch ")
    report2 = run_consolidation(st, hours=24)
    assert report2["facts_created"] == 0
    assert report2["facts_deduplicated"] == 3

    rows = st.search("SELECT COUNT(*) FROM memories WHERE kind = 'semantic'")["rows"]
    assert rows[0][0] == 3  # no new semantic rows were created
    hits = st.recall("deploy runbook version tag release branch", k=5, kind="semantic")
    assert len(hits) == 3
    st.close()


def test_supersede_hides_from_recall_and_blocks_double_supersede(tmp_path):
    clock = {"t": 1_700_000_000.0}
    st = make_store(tmp_path, now=lambda: clock["t"])
    old_id = st.remember("user prefers dark mode for terminal themes")["id"]
    new_id = st.remember("user switched to light mode for terminal themes")["id"]

    result = st.supersede(old_id, new_id)
    assert result == {"superseded": old_id, "by": new_id}

    hits = st.recall("preferred terminal theme mode", k=5)
    assert [h["id"] for h in hits] == [new_id]

    with pytest.raises(StoreError, match="already superseded"):
        st.supersede(old_id, new_id)

    stats = st.stats()
    assert stats["by_kind"]["semantic"] == 1
    st.close()


def test_no_dedup_below_threshold(tmp_path):
    clock = {"t": 1_700_000_000.0}
    st = make_store(tmp_path, now=lambda: clock["t"])
    _seed_batch(st, "quarterly budget review {i}: marketing spend grew twelve percent")
    run_consolidation(st, hours=24)

    clock["t"] += 48 * 3600
    _seed_batch(st, "kitchen trial {i}: simmer tomatoes garlic basil into a smooth sauce")
    report = run_consolidation(st, hours=24)
    assert report["facts_created"] == 3
    assert report["facts_deduplicated"] == 0

    rows = st.search("SELECT COUNT(*) FROM memories WHERE kind = 'semantic'")["rows"]
    assert rows[0][0] == 6
    st.close()
