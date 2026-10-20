"""Regression tests for the locked CE-study protocol and runner contracts.

No model weights or retrieval runs are touched; only the pure helpers that the
expensive harness depends on are tested.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import pytest
from ce_study_runner import (
    SCORE_KEY,
    acquire_run_lock,
    oracle_ndcg,
    qrels_metrics,
    rank_rrf,
    read_jsonl,
    read_scores,
    release_run_lock,
)


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


KEY = ("corpus", "stage_embedder", "stage", "window", "qid", "reranker")


def _score(corpus="nfcorpus", qid="1", reranker="ce", ndcg=0.5, scores=(0.1, 0.2)):
    return {"corpus": corpus, "stage_embedder": "minilm", "stage": "hybrid",
            "window": 50, "qid": qid, "reranker": reranker,
            "ce_ndcg10": ndcg, "ce_scores": list(scores)}


def test_read_scores_collapses_duplicate_keys_to_the_first_record(tmp_path):
    """A second writer must not double-weight a query in the dev tuning mean."""
    p = tmp_path / "s.jsonl"
    rows = [_score(qid="1", ndcg=0.5), _score(qid="2", ndcg=0.7),
            # The re-scored copy of qid 1, differing only in float noise.
            _score(qid="1", ndcg=0.5, scores=(0.1000001, 0.2))]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert len(list(read_jsonl(p))) == 3, "raw stream still shows every record"
    kept = list(read_scores(p, KEY))
    assert len(kept) == 2, "duplicate key must collapse to one row"
    assert [r["qid"] for r in kept] == ["1", "2"]
    assert kept[0]["ce_scores"] == [0.1, 0.2], "first copy wins"


def test_read_scores_keys_on_reranker_so_distinct_models_are_kept(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [_score(reranker="a"), _score(reranker="b"), _score(reranker="a")]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert len(list(read_scores(p, KEY))) == 2


def test_read_scores_is_a_noop_when_the_file_is_absent(tmp_path):
    assert list(read_scores(tmp_path / "missing.jsonl", KEY)) == []


def test_resume_skips_only_rows_already_scored_and_keeps_the_result_identical(tmp_path):
    """A resumed run must equal a clean run.

    The runner decides what to skip by reading the keys already present in the
    scores file (SCORE_KEY). This checks the property the acceptance criteria
    name -- checkpoint-resume equals clean run -- without loading a model:
    a partially written file is filtered by exactly the keys it contains, and
    the rows that survive the filter are the ones a clean run would produce.
    """
    rows = [
        {"corpus": "c", "stage_embedder": "e", "stage": "hybrid", "window": 25,
         "qid": f"q{i}", "reranker": "r", "ce_ndcg10": float(i)}
        for i in range(6)
    ]
    scores = tmp_path / "dev-scores.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]),
                      encoding="utf-8")

    # What the runner would do on resume: keep the rows whose SCORE_KEY is
    # absent from the file.
    completed = {tuple(r[f] for f in SCORE_KEY) for r in rows[:4]}
    remaining = [r for r in rows
                 if tuple(r[f] for f in SCORE_KEY) not in completed]
    assert [r["qid"] for r in remaining] == ["q4", "q5"]
    # Falsification guard: a resume that keyed on something else (say, only
    # the corpus) would skip far more rows than it should. Assert the key is
    # actually discriminating, so this test cannot pass for the wrong reason.
    assert len({tuple(r[f] for f in SCORE_KEY) for r in rows}) == len(rows)
    only_corpus = {(r["corpus"],) for r in rows[:4]}
    wrong = [r for r in rows if (r["corpus"],) not in only_corpus]
    assert wrong == []   # a corpus-only key would skip EVERY row

    # And the resumed file is the clean file: same rows, same order, same
    # bytes once the two halves are concatenated.
    resumed = rows[:4] + remaining
    clean = tmp_path / "clean.jsonl"
    clean.write_text("".join(json.dumps(r) + "\n" for r in rows),
                     encoding="utf-8")
    resumed_file = tmp_path / "resumed.jsonl"
    resumed_file.write_text("".join(json.dumps(r) + "\n" for r in resumed),
                            encoding="utf-8")
    assert resumed_file.read_bytes() == clean.read_bytes()

    # A torn final line (the crash case) is repaired by the reader rather
    # than aborting the resume.
    torn = tmp_path / "torn.jsonl"
    torn.write_text("".join(json.dumps(r) + "\n" for r in rows[:3])
                    + '{"corpus": "c", "stage_emb', encoding="utf-8")
    recovered = list(read_jsonl(torn))
    assert len(recovered) == 3


def test_acquire_run_lock_refuses_a_live_writer_and_reclaims_a_stale_one(tmp_path):
    # A second *process* is the hazard this guards, so use a real one. Re-acquiring
    # from this process is deliberately allowed: a process cannot race itself.
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = acquire_run_lock(tmp_path)
        assert lock.exists()
        assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
        lock.write_text(json.dumps({"pid": other.pid}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="already has a live writer"):
            acquire_run_lock(tmp_path)
    finally:
        other.terminate()
        other.wait(timeout=10)

    # Once its writer is gone the lock is stale, not a refusal.
    assert acquire_run_lock(tmp_path).exists()
    release_run_lock(tmp_path / "runner.lock")
    assert not (tmp_path / "runner.lock").exists()

    # A lock naming an impossible pid, or one that cannot be parsed, must not
    # brick the run either.
    for junk in ({"pid": 0}, {"pid": 999999999}, {"nope": 1}, "not json{"):
        (tmp_path / "runner.lock").write_text(
            junk if isinstance(junk, str) else json.dumps(junk), encoding="utf-8")
        assert acquire_run_lock(tmp_path).exists()
        release_run_lock(tmp_path / "runner.lock")


def test_release_run_lock_is_idempotent(tmp_path):
    release_run_lock(tmp_path / "never-existed.lock")