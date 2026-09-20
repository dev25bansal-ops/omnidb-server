"""Recall-pipeline split (E-01): the in-store body moved into
omnidb_server.recall.run_recall. These tests prove the public
MemoryStore.recall contract is unchanged after the extraction."""

import pytest

from omnidb_server.recall import run_recall
from omnidb_server.store import StoreError


def _ids(results):
    return [r["id"] for r in results]


def test_recall_thin_forwarder_returns_same_results_as_run_recall(store):
    """store.recall must be a forwarder onto run_recall; identity of results
    is the strongest proof the split changed nothing visible to callers."""
    a = store.remember("python decorators guide")["id"]
    b = store.remember("jazz improvisation chords")["id"]
    store.remember("history of jazz chords evolution")["id"]

    via_store = store.recall("jazz chords", k=5)
    via_helper = run_recall(store, "jazz chords", 5, None, None, None, None)

    assert _ids(via_store) == _ids(via_helper)
    pos_store = {r["id"]: i for i, r in enumerate(via_store)}
    pos_helper = {r["id"]: i for i, r in enumerate(via_helper)}
    assert pos_store == pos_helper
    if a in pos_store:
        assert pos_store[b] < pos_store[a]


def test_fusion_ordering_preserved_after_split(store):
    """Dense+sparse RRF ordering must match pre-split semantics: a row that
    matches only the sparse leg still appears, and the dual-match row outranks
    it."""
    strong = store.remember("kubernetes ingress troubleshooting")["id"]
    weak = store.remember("random musings about cats and dogs")["id"]
    hits = store.recall("kubernetes ingress", k=5)
    pos = {r["id"]: i for i, r in enumerate(hits)}
    assert strong in pos
    assert pos[strong] < pos.get(weak, 10**9)


def test_agent_id_filtering_still_works(store):
    a = store.remember("deploy runbook for payments", agent_id="alpha")["id"]
    b = store.remember("deploy runbook for billing", agent_id="beta")["id"]

    alpha_hits = store.recall("deploy runbook", k=10, agent_id="alpha")
    beta_hits = store.recall("deploy runbook", k=10, agent_id="beta")
    unscoped = store.recall("deploy runbook", k=10)

    assert _ids(alpha_hits) == [a]
    assert _ids(beta_hits) == [b]
    assert {h["id"] for h in unscoped} == {a, b}


def test_reranker_still_flips_order_after_split(tmp_path):
    import conftest

    st = conftest.make_store(tmp_path)
    strong_rrf = st.remember("harbor seal migration")["id"]
    weak_rrf = st.remember("harbor seal migration routes and pupping grounds")["id"]

    baseline = st.recall("harbor seal migration routes", k=2)
    reranked = st.recall(
        "harbor seal migration routes", k=2, reranker="lexical_overlap"
    )
    assert _ids(reranked) == [weak_rrf, strong_rrf]
    assert _ids(baseline)[0] == strong_rrf
    assert _ids(reranked)[0] != _ids(baseline)[0]
    st.close()


def test_skill_trigger_and_k_clamp_still_apply_after_split(store):
    """The procedural-trigger merge and final [:k] clamp live inside
    run_recall; both must survive the extraction."""
    skill = store.remember(
        "run kubectl get events first",
        kind="procedural",
        trigger_patterns=["^deploy\\b", "kubectl"],
    )["id"]
    for i in range(4):
        store.remember(f"kayak storage tip number {i}")

    triggered = store.recall("how do I use kubectl here?", k=5)
    by_id = {h["id"]: h for h in triggered}
    assert skill in by_id
    assert by_id[skill]["metadata"]["skill_trigger"] is True

    no_trigger = store.recall("gardening tips for spring", k=5)
    assert all(not h["metadata"].get("skill_trigger") for h in no_trigger)

    # k clamp still enforced
    clipped = store.recall("kayak storage tips", k=2)
    assert len(clipped) == 2
    with pytest.raises(StoreError):
        store.recall("anything", k=0)
