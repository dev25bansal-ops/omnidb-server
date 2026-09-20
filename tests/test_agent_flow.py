"""Whole-product integration test: an agent's first day of use.

Session flow: episodic chatter -> consolidation distills semantics -> topical
recall surfaces distilled knowledge -> graph linking -> selective forgetting ->
stats agree. Exercises the tools the way Claude Desktop will, in one story.
"""

import pytest
from conftest import make_store

from omnidb_server.consolidate import run_consolidation
from omnidb_server.store import StoreError


def test_agent_first_day(tmp_path):
    st = make_store(tmp_path)

    # --- session 1: raw episodic chatter -------------------------------
    st.remember("User said they prefer Rust for systems work", kind="episodic")
    st.remember("User asked how SQLite WAL checkpoint tuning works", kind="episodic")
    st.remember("User complained the old CLI felt slow yesterday", kind="episodic")

    # --- consolidation distills episodes into semantic knowledge -------
    report = run_consolidation(st, hours=1)
    assert report["facts_created"] == 3
    assert report["llm_used"] is False

    kinds = dict(
        (r[0], r[1]) for r in st.search(
            "SELECT kind, COUNT(*) FROM memories GROUP BY kind"
        )["rows"]
    )
    assert kinds == {"episodic": 3, "semantic": 3}

    # re-running consolidates nothing new (episodes are marked)
    assert run_consolidation(st, hours=1)["facts_created"] == 0

    # --- recall by paraphrase finds the distilled preference -----------
    hits = st.recall("what language does the user like for systems programming?", k=5)
    assert any("Rust" in h["content"] for h in hits)

    # --- agent links two related memories, then walks the graph --------
    rust_id = hits[0]["id"]
    wal = st.recall("WAL checkpointing", k=1)[0]["id"]
    st.link(rust_id, wal, label="context")
    neighbors = {r["id"] for r in st.related(rust_id, max_hops=1)}
    assert wal in neighbors

    # --- selective forgetting ------------------------------------------
    doomed = st.remember("temporary scratch note about nothing", kind="semantic")["id"]
    st.forget(doomed)
    assert all(h["id"] != doomed for h in st.recall("scratch note nothing", k=5))
    with pytest.raises(StoreError):
        st.link(doomed, rust_id)  # cannot link from a deleted memory

    # --- stats agree with everything above ------------------------------
    stats = st.stats()
    assert stats["by_kind"]["semantic"] == 3          # scratch was soft-deleted
    assert stats["soft_deleted"] == 1
    assert stats["edges"] == 1
    st.close()
