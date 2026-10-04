"""F15 - a score-0 BEIR judgment is NOT a relevant document.

`load_beir` yields relevance as a `{doc_id: score}` dict read straight from the BEIR
qrels, which grade some documents 0. "Assessed and judged not relevant" is not the same
as "never assessed", so those documents must not count toward recall.

The old code did `set(rel)`, which keeps every dict key - so the denominator counted
score-0 documents and recall came out systematically low, by a corpus-dependent amount
proportional to how many negative judgments that corpus has.

nDCG is unaffected: it needs the graded scores and reads the dict directly
(`research_bench_v3.ndcg_at_k`), so the headline metric was never wrong. The damage was
confined to `stage_recall_at_w`, the x-axis of the paper's regime law.

This module pins the corrected semantics so the old form cannot come back.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "bench"
# The bench scripts import each other by bare module name, so bench/ has to be
# importable before any of them is loaded.
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_f15_{name}", _BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rb():
    return _load("research_bench_v3")


@pytest.fixture(scope="module")
def prw():
    return _load("probe_rerank_window")


# --- the helper -------------------------------------------------------------


def test_score_zero_documents_are_excluded(rb):
    rel = {"d1": 1, "d2": 2, "d3": 0, "d4": 0, "d5": 0}
    assert rb.relevant_docs(rel) == {"d1", "d2"}


def test_binary_qrels_behave_identically_to_the_old_form(rb):
    """A 0/1 corpus still drops the negatives - and that is the point."""
    rel = {"d1": 1, "d2": 0, "d3": 0}
    assert rb.relevant_docs(rel) == {"d1"}


def test_all_zero_relevance_yields_an_empty_set_not_every_doc(rb):
    assert rb.relevant_docs({"d1": 0, "d2": 0}) == set()


def test_the_buggy_helper_is_gone(rb):
    """`recall_at_k` divided by len(dict) and had zero callers. It must not return."""
    assert not hasattr(rb, "recall_at_k")


# --- the metric that was actually wrong --------------------------------------


def test_recall_is_deflated_by_the_old_form(rb, prw):
    """Fixing the denominator LOWERS recall. That is the expected direction.

    A system that retrieved one of the two genuinely-relevant documents scored 0.60
    under `set(rel)` (1 hit over 5 judged docs) and scores 0.50 now (1 hit over 2
    relevant docs). Reporting the old number flattered the system on any corpus with
    negative judgments.
    """
    rel = {"d1": 1, "d2": 2, "d3": 0, "d4": 0, "d5": 0}
    ranked = ["d1", "d3", "d5"]
    fixed = prw.recall_at(ranked, rb.relevant_docs(rel), 3)
    old = prw.recall_at(ranked, set(rel), 3)
    assert fixed == pytest.approx(0.5)
    assert old == pytest.approx(0.6)
    assert fixed < old


def test_recall_with_no_relevant_documents_is_zero_not_a_crash(rb, prw):
    """A query whose every judgment is 0 must not divide by zero."""
    rel = {"d1": 0, "d2": 0}
    assert prw.recall_at(["d1", "d2"], rb.relevant_docs(rel), 2) == 0.0


def test_ndcg_is_unchanged(rb):
    """The headline metric reads scores, not the set, so it must not move."""
    rel = {"d1": 1, "d2": 2, "d3": 0}
    assert rb.ndcg_at_k(["d1", "d3"], rel, 2) == pytest.approx(rb.ndcg_at_k(["d1", "d3"], rel, 2))
    assert rb.ndcg_at_k(["d2", "d1"], rel, 2) > rb.ndcg_at_k(["d1", "d3"], rel, 2)
