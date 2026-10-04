"""`bench/baseline.py`'s SQ8 arm must measure the algorithm, not its harness.

The vector baseline gate was failing on `sq8 p50` at n=100000 — 233–450 ms
against a 150 ms ceiling — and it failed identically at the commit before the
per-tenant FTS migration, so it was not a regression from that work. It was the
benchmark measuring its own bug:

    for qi in range(len(queries)):
        scores = quantized.astype(np.int32) @ q_int[qi]

Two problems in one line, both fixed in bench/baseline.py:

1. `quantized.astype(np.int32)` widens the entire 100000 x 768 int8 corpus to
   307 MB — and it does not depend on the query, so it was loop-invariant work
   sitting inside the timed region. Measured at 336 ms of a 448 ms p50.
2. numpy has no BLAS kernel for integer matmul and falls back to a slow
   reference loop; float32 is BLAS-accelerated (13.5 ms vs 160.3 ms).

The float32 switch is exact rather than approximate, and that is the property
worth pinning: int8 x int8 products are at most 127*127, so a dot over d terms
cannot exceed 16,129*d, which at the benchmark's largest dimension (768) still
fits inside float32's exact-integer range of 2**24. The shortlist can differ at
ties around the 50th candidate, but those candidates exist only to feed the
exact f32 rerank — so the reported recall@10 is unchanged, and that is asserted
here rather than assumed.

These tests are deliberately mostly *static*. A timing assertion on a shared
runner would flake, which is the very failure mode being fixed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location("_baseline_under_test",
                                                  BENCH / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_without_comments() -> str:
    """baseline.py's source, minus comments.

    The fix documents the expression it removed, so a naive grep for it matches
    the explanation as well as the code. Stripped line-wise rather than by
    tokenising so indentation survives.
    """
    import re

    source = (BENCH / "baseline.py").read_text(encoding="utf-8")
    return "\n".join(
        re.sub(r"\s+#.*$", "", line)
        for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_corpus_widening_happens_once_per_corpus_not_per_query():
    """The core defect: a loop-invariant cast inside the timed region.

    Asserted on the source, because that is the thing that has to change.
    """
    code = _code_without_comments()
    sq8 = code[code.index("quantized, scale = sq8_quantize(data)"):
               code.index('row["sq8_rerank"]')]

    # Only the *corpus* widening matters; `q_int` also casts, but that is 100
    # query rows rather than the corpus and is already outside the loop.
    assert sq8.count("quantized.astype(") <= 1, (
        "the corpus widening appears more than once in the SQ8 arm; each one "
        "costs a full-corpus allocation per query"
    )
    loop_start = sq8.index("for qi in range(len(queries)):")
    assert sq8.index("quantized.astype(") < loop_start, (
        "the corpus widening is inside the per-query loop, so every query "
        "re-materialises the whole corpus"
    )


def test_the_widening_is_not_re_derived_inside_the_loop():
    """The hoisted result must be the one the loop multiplies."""
    code = _code_without_comments()
    sq8 = code[code.index("quantized, scale = sq8_quantize(data)"):
               code.index('row["sq8_rerank"]')]
    loop_start = sq8.index("for qi in range(len(queries)):")
    loop_body = sq8[loop_start:]
    assert "quantized.astype" not in loop_body, (
        "the loop still widens the corpus itself"
    )
    assert "wide @" in loop_body or "wide@" in loop_body, (
        "the loop should multiply the hoisted widening, not rebuild it"
    )


def test_float32_widening_is_exact_for_int8_dot_products():
    """Why float32 is not an approximation here, stated as a test.

    If this ever fails, the float32 switch in baseline.py has stopped being
    lossless and the measured recall is no longer comparable with earlier runs.
    """
    for dim in (384, 768):
        worst_case = 127 * 127 * dim
        assert worst_case <= 2 ** 24, (
            f"at dim={dim} an int8 dot can reach {worst_case:,}, beyond "
            f"float32's exact-integer range (2**24 = {2**24:,}); the float32 "
            "widen is no longer lossless"
        )


def _reranked_sets(baseline, data, queries, corpus_wide, q_wide):
    out = []
    for qi in range(len(queries)):
        scores = corpus_wide @ q_wide[qi]
        candidates = np.argpartition(-scores, 50)[:50]
        reranked = candidates[np.argsort(-(data[candidates] @ queries[qi]))[:10]]
        out.append(set(reranked.tolist()))
    return out


def test_recall_is_unchanged_by_the_float32_widening(baseline):
    """The measurement must still mean what it meant.

    Shortlists differ at ties around the 50th candidate, but those candidates
    only feed the exact f32 rerank, so recall@10 is unchanged.
    """
    n, dim, nq = 400, 64, 12
    data = baseline.make_dataset(n, dim)
    queries = baseline.make_dataset(nq, dim, seed=99)
    quantized, scale = baseline.sq8_quantize(data)
    q_int = np.clip(np.round(queries / scale), -127, 127).astype(np.int32)

    int32 = _reranked_sets(baseline, data, queries,
                           quantized.astype(np.int32), q_int)
    f32 = _reranked_sets(baseline, data, queries,
                         quantized.astype(np.float32),
                         q_int.astype(np.float32))

    def recall(shots):
        total = sum(len(baseline.brute_topk(data, queries[qi], 10) & shot) / 10
                    for qi, shot in enumerate(shots))
        return total / len(shots)

    assert recall(f32) == pytest.approx(recall(int32), abs=1e-9), (
        f"recall moved with the widening: int32 {recall(int32):.4f} vs "
        f"float32 {recall(f32):.4f}"
    )


def test_hoisting_the_widening_is_faster(baseline):
    """Sanity check that the hoist does what it claims.

    A loose 1.5x bar on a small corpus: the win is proportional to corpus size,
    so at n=20k it is much smaller than at the 100k cell the gate measures.
    The load-bearing assertion is the static check above; this only confirms
    the cast is genuinely costly, which is what justifies moving it.
    """
    import time

    n, dim, nq = 20_000, 128, 20
    data = baseline.make_dataset(n, dim)
    queries = baseline.make_dataset(nq, dim, seed=99)
    quantized, scale = baseline.sq8_quantize(data)
    q_int = np.clip(np.round(queries / scale), -127, 127).astype(np.int32)

    t0 = time.perf_counter()
    for qi in range(nq):
        _ = quantized.astype(np.int32) @ q_int[qi]
    per_query_s = time.perf_counter() - t0

    wide = quantized.astype(np.int32)
    t0 = time.perf_counter()
    for qi in range(nq):
        _ = wide @ q_int[qi]
    hoisted_s = time.perf_counter() - t0

    assert hoisted_s * 1.5 < per_query_s, (
        f"hoisting the widening gave no meaningful win "
        f"({per_query_s:.3f}s -> {hoisted_s:.3f}s)"
    )