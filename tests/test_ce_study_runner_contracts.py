"""Regression tests for the locked CE-study protocol and runner contracts.

No model weights or retrieval runs are touched; only the pure helpers that the
expensive harness depends on are tested.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pytest
from ce_study_runner import (K, oracle_ndcg, qrels_metrics, read_jsonl,
                             rank_rrf)


def test_metrics_handle_unjudged_documents_and_oracle(tmp_path):
    qrels = {"a": 2, "b": 1, "judged-zero": 0}
    ranking = ["unjudged", "a", "judged-zero", "b"]
    m = qrels_metrics(ranking, qrels, k=4)
    assert m["judged_at10"] == 0.75
    assert m["unjudged_at10"] == 0.25
    assert oracle_ndcg(ranking, qrels, k=4) >= m["ndcg10"]


def test_rank_rrf_endpoints_are_deterministic_and_reject_misalignment():
    ids = ["d3", "d1", "d2"]
    scores = [0.1, 0.9, 0.2]
    assert rank_rrf(ids, scores, 0.0, 60) == ids
    assert rank_rrf(ids, scores, 1.0, 60) == ["d1", "d2", "d3"]
    with pytest.raises(ValueError):
        rank_rrf(["a"], [0.1, 0.2], 0.5, 60)
    with pytest.raises(ValueError):
        rank_rrf(["a", "a"], [0.1, 0.2], 0.5, 60)


def test_jsonl_recovers_truncated_final_fragment(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n{"c":3', encoding="utf-8")
    rows = list(read_jsonl(p))
    assert rows == [{"a": 1}, {"b": 2}]
    # After recovery, only the final partial line is removed.
    assert p.read_text(encoding="utf-8") == '{"a":1}\n{"b":2}\n'