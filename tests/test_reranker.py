"""Reranker hook (analysis §4.2): registry, built-ins, and store integration.

Direct apply_reranker tests cover ordering semantics on synthetic result
lists; the store-backed tests prove recall(reranker=...) actually flips order
relative to the un-reranked baseline.
"""

import pytest

from omnidb_server.rerank import REGISTRY, apply_reranker, register


def _item(mid, score=1.0, recency=None, content=""):
    return {
        "id": mid,
        "score": score,
        "content": content,
        "recency": 0.5 if recency is None else recency,
    }


def _ids(results):
    return [r["id"] for r in results]


# --- registry -------------------------------------------------------------------

def test_builtin_names_registered():
    assert {"none", "lexical_overlap", "recency_first"} <= set(REGISTRY)


def test_unknown_name_raises_typed_error():
    from omnidb_server.store import StoreError

    with pytest.raises(StoreError) as exc:
        apply_reranker("does_not_exist", "query", [])
    assert "unknown reranker 'does_not_exist'" in str(exc.value)
    assert exc.value.code == "OMN-002"


def test_register_decorator_adds_callable():
    @register("test_probe_strategy")
    def _probe(query, results):
        return results

    try:
        assert REGISTRY["test_probe_strategy"] is _probe
    finally:
        del REGISTRY["test_probe_strategy"]


# --- none -----------------------------------------------------------------------

def test_none_identity_preserves_order():
    results = [_item(3, 0.9), _item(1, 2.0), _item(2, 1.0)]
    out = apply_reranker("none", "anything", results)
    assert out is results
    assert _ids(out) == [3, 1, 2]


# --- lexical_overlap --------------------------------------------------------------

def test_lexical_overlap_boosts_matching_content_to_front():
    results = [
        _item(10, 1.3, content="totally unrelated words here"),
        _item(11, 1.0, content="zebra habitat facts and zebra diet"),
    ]
    out = apply_reranker("lexical_overlap", "zebra habitat", results)
    # id 11 doubles (both tokens hit) while id 10 stays at its base score;
    # ties would fall back to id, but these scores are distinct
    assert out[0]["score"] > out[1]["score"]
    assert _ids(out) == [11, 10]


def test_lexical_overlap_tie_broken_by_id_ascending():
    results = [
        _item(42, 1.0, content="alpha"),
        _item(7, 1.0, content="beta gamma"),
        _item(19, 1.0, content="delta"),
    ]
    out = apply_reranker("lexical_overlap", "nothing matches xyzzy", results)
    assert _ids(out) == [7, 19, 42]


def test_lexical_overlap_query_tokens_lowercased_and_word_split():
    results = [
        _item(1, 1.0, content="Deploy Runbook!"),
        _item(2, 1.0, content="unrelated"),
    ]
    out = apply_reranker("lexical_overlap", "deploy runbook", results)
    assert out[0]["id"] == 1
    assert out[0]["score"] == pytest.approx(2.0)


def test_lexical_overlap_empty_query_leaves_relative_scores_intact():
    results = [_item(1, 4.0), _item(2, 9.0)]
    out = apply_reranker("lexical_overlap", "", results)
    assert _ids(out) == [2, 1]
    assert [r["score"] for r in out] == [9.0, 4.0]


def test_lexical_overlap_formula_matches_spec():
    # score *= 1 + |q ∩ c| / max(1, |q|); q={a,b,c}, content holds all three -> x2
    partial = apply_reranker("lexical_overlap", "a b c", [_item(1, 0.25, content="a b")])
    assert partial[0]["score"] == pytest.approx(0.25 * (1 + 2 / 3))
    full = apply_reranker("lexical_overlap", "a b c", [_item(2, 0.25, content="a b c")])
    assert full[0]["score"] == pytest.approx(0.5)


# --- recency_first ----------------------------------------------------------------

def test_recency_first_orders_by_recency_then_score_then_id():
    results = [
        _item(1, score=5.0, recency=0.1),
        _item(2, score=0.2, recency=0.9),
        _item(3, score=9.9, recency=0.9),
        _item(4, score=1.0, recency=0.9),
    ]
    out = apply_reranker("recency_first", "q", results)
    assert _ids(out) == [3, 4, 2, 1]
    # ranks are encoded into score so they survive store.recall's re-sort
    assert [r["score"] for r in out] == [4.0, 3.0, 2.0, 1.0]


def test_recency_first_equal_recency_falls_back_to_score():
    results = [
        _item(8, score=0.5, recency=1.0),
        _item(9, score=2.0, recency=1.0),
    ]
    out = apply_reranker("recency_first", "q", results)
    assert _ids(out) == [9, 8]


def test_recency_first_full_tie_stable_by_id():
    results = [_item(6, score=1.0, recency=1.0), _item(5, score=1.0, recency=1.0)]
    out = apply_reranker("recency_first", "q", results)
    assert _ids(out) == [5, 6]


# --- store integration ---------------------------------------------------------------

def test_recall_with_none_reranker_matches_baseline(store):
    for text in ("quartz crystal oscillator", "quartz mining history",
                 "unrelated pottery glaze"):
        store.remember(text)
    baseline = store.recall("quartz", k=3)
    with_none = store.recall("quartz", k=3, reranker="none")
    assert _ids(with_none) == _ids(baseline)


def test_recall_lexical_overlap_flips_order_vs_baseline(tmp_path):
    import conftest

    st = conftest.make_store(tmp_path)
    # the shorter content wins fusion outright, but the longer one matches
    # strictly more query tokens — lexical_overlap must invert the order
    strong_rrf = st.remember("harbor seal migration")["id"]
    weak_rrf = st.remember("harbor seal migration routes and pupping grounds")["id"]

    baseline_ids = _ids(st.recall("harbor seal migration routes", k=2))
    reranked_ids = _ids(st.recall(
        "harbor seal migration routes", k=2, reranker="lexical_overlap"))
    assert len(baseline_ids) == len(reranked_ids) == 2
    assert set(baseline_ids) == {strong_rrf, weak_rrf}
    assert reranked_ids[0] != baseline_ids[0]
    assert reranked_ids[0] == weak_rrf
    st.close()


def test_recall_recency_first_prefers_newest_memory(tmp_path):
    import conftest

    clock = {"t": 1_700_000_000.0}
    st = conftest.make_store(tmp_path, now=lambda: clock["t"])
    old = st.remember("identical payload body")["id"]
    clock["t"] += 500 * 3600
    new = st.remember("identical payload body")["id"]

    baseline = st.recall("identical payload", k=2)
    by_id = {r["id"]: r for r in baseline}
    assert by_id[new]["score"] > by_id[old]["score"]  # baseline favors new mildly

    flipped = st.recall("identical payload", k=2, reranker="recency_first")
    assert _ids(flipped)[0] == new
    assert _ids(flipped)[1] == old
    # ordinal rank survived store.recall's final (-score, id) sort
    assert flipped[0]["score"] > flipped[1]["score"]
    st.close()


def test_recall_k_still_applied_after_reranking(store):
    ids = [store.remember(f"kayak storage tip number {i}")["id"] for i in range(4)]
    hits = store.recall("kayak storage tips", k=2, reranker="recency_first")
    assert len(hits) == 2
    assert set(_ids(hits)) <= set(ids)


def test_recall_reranked_result_payload_shape_unchanged(store):
    store.remember("payload shape probe")
    base = store.recall("payload shape probe", k=1)[0]
    reranked = store.recall("payload shape probe", k=1, reranker="lexical_overlap")[0]
    assert set(reranked) == set(base)
