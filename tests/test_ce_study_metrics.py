"""Protocol-level tests for candidate pool and ranking utilities.

These tests do not load models or execute BEIR retrieval; they pin the contracts
used by the expensive study runner.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pytest
from ce_study_runner import oracle_ndcg, qrels_metrics, rank_rrf


def test_rank_rrf_endpoints_and_tie_order_are_deterministic():
    ids=["d3","d1","d2"]
    scores=[0.1,0.9,0.2]
    assert rank_rrf(ids,scores,0.0,60)==ids
    assert rank_rrf(ids,scores,1.0,60)==["d1","d2","d3"]
    assert rank_rrf(ids,[1,1,1],1.0,60)==["d3","d1","d2"]


def test_rank_rrf_rejects_misalignment_and_duplicate_candidates():
    with pytest.raises(ValueError): rank_rrf(["a"],[0.1,0.2],0.5,60)
    with pytest.raises(ValueError): rank_rrf(["a","a"],[0.1,0.2],0.5,60)


def test_judged_unjudged_and_oracle_pool_metrics():
    qrels={"a":2,"b":1,"judged-zero":0}
    ranking=["unjudged","a","judged-zero","b"]
    m=qrels_metrics(ranking,qrels,k=4)
    assert m["judged_at10"]==0.75
    assert m["unjudged_at10"]==0.25
    assert m["ndcg10"] < qrels_metrics(["a", "b"], qrels, k=4)["ndcg10"]
    assert oracle_ndcg(ranking, qrels, k=4) >= m["ndcg10"]


def test_condensed_qrels_metrics_not_a_replacement_for_standard_metric():
    qrels={"a":1,"b":1}
    ordinary=qrels_metrics(["x","x2","a","b"],qrels,k=4)["ndcg10"]
    condensed=qrels_metrics(["a","b"],qrels,k=4)["ndcg10"]
    assert condensed > ordinary
