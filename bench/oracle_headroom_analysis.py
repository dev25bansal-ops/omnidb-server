"""X2: does the pool-oracle headroom predict the re-ranker's effect?

Section V-D of the manuscript states that the proposed mechanism (the first
stage's bound on what a re-ranker can recover) is NOT directly confirmed,
because the rank displacement of relevant documents, the score/order rank
correlation, and the up-vs-down displacement counts are not measured.  It
further says the "pool-oracle nDCG" is "reported per cell in Table VI" and
that a formal query-clustered predictor analysis is left to future work.

That leaves a claim and a measurement gap: the paper asserts the mechanism is
plausible on the basis of a "bound" argument, and separately reports a quantity
(the pool-oracle headroom) that bounds the same thing.  This script tests
whether the headroom actually orders the observed effect.

Two questions, both answerable from the committed held-out artifacts:

  Q1 (does headroom order the effect?)  For every held-out cell, compute
      headroom = oracle_mean - base_mean (the maximum nDCG@10 any re-ranker
      could have achieved by reordering that pool) and compare it against the
      observed re-ranking effect.  Rank-correlate within corpus and overall.

  Q2 (is the effect bounded by the headroom?)  A re-ranker cannot exceed the
      pool oracle, so for a CE to beat the base it needs headroom > 0.  Count
      the cells where the observed effect has the opposite sign to the
      headroom, i.e. cases where the mechanism's bound is not merely loose but
      is violated in sign.

The per-query arrays are not in test-summary.json, so Q1/Q2 are computed over
the 660 cell means.  With m=660 the analysis is descriptive, not confirmatory,
and the manuscript must say so.

Usage:
    python bench/oracle_headroom_analysis.py [--run-id ce-study-v1-20261009]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def spearman(x: np.ndarray, y: np.ndarray):
    """Spearman rho with an exact-ish two-sided p by permutation."""
    from scipy import stats
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="ce-study-v1-20261009")
    args = ap.parse_args()

    run = HERE / "ce-study-runs" / args.run_id
    summary = json.loads((run / "test-summary.json").read_text(encoding="utf-8"))
    cells = summary["family_a"]

    print(f"run: {args.run_id}   cells: {len(cells)}")
    print(f"declared family size: {summary['metadata']['family_size_predeclared']}")
    print()

    for c in cells:
        c["_headroom"] = c["oracle_mean"] - c["base_mean"]
        c["_effect"] = c["delta_mean"]          # blend - base, dev-selected fusion
        c["_ce_effect"] = c["ce_only_mean"] - c["base_mean"]  # pure replacement

    # ---- Q1: does the headroom order the effect? -------------------------
    print("=" * 78)
    print("Q1  Does the pool-oracle headroom order the observed effect?")
    print("=" * 78)

    for label, key in (("dev-selected fusion (blend - base)", "_effect"),
                       ("pure replacement (CE-alone - base)", "_ce_effect")):
        h = np.array([c["_headroom"] for c in cells])
        e = np.array([c[key] for c in cells])
        rho, p = spearman(h, e)
        print(f"\n  all 660 cells, {label}:")
        print(f"    Spearman rho = {rho:+.4f}   p = {p:.4f}")

    # within corpus (removes the between-corpus scale differences)
    print("\n  within corpus (hybrid stage only, one corpus at a time):")
    by_corpus = defaultdict(list)
    for c in cells:
        if c["stage"] == "hybrid":
            by_corpus[c["corpus"]].append(c)
    rhos = []
    for corpus in sorted(by_corpus):
        grp = by_corpus[corpus]
        h = np.array([c["_headroom"] for c in grp])
        e = np.array([c["_effect"] for c in grp])
        rho, p = spearman(h, e)
        rhos.append(rho)
        print(f"    {corpus:<10} n={len(grp):>3}  rho = {rho:+.4f}  p = {p:.4f}")
    print(f"    mean within-corpus rho = {np.mean(rhos):+.4f}")

    # across corpora, one point per corpus (hybrid, averaged over the matrix)
    print("\n  across corpora (hybrid, averaged over rerankers x windows, n=4):")
    for corpus in sorted(by_corpus):
        grp = by_corpus[corpus]
        h = float(np.mean([c["_headroom"] for c in grp]))
        e = float(np.mean([c["_effect"] for c in grp]))
        ece = float(np.mean([c["_ce_effect"] for c in grp]))
        print(f"    {corpus:<10} headroom={h:+.4f}  fusion effect={e:+.4f}  "
              f"CE-alone effect={ece:+.4f}")
    corpus_h = np.array([np.mean([c["_headroom"] for c in by_corpus[cp]]) for cp in sorted(by_corpus)])
    corpus_e = np.array([np.mean([c["_effect"] for c in by_corpus[cp]]) for cp in sorted(by_corpus)])
    corpus_ce = np.array([np.mean([c["_ce_effect"] for c in by_corpus[cp]]) for cp in sorted(by_corpus)])
    rho, p = spearman(corpus_h, corpus_e)
    print(f"    Spearman rho (fusion)  = {rho:+.4f}  p = {p:.4f}")
    rho, p = spearman(corpus_h, corpus_ce)
    print(f"    Spearman rho (CE-alone) = {rho:+.4f}  p = {p:.4f}")

    # ---- Q2: is the effect bounded by the headroom in sign? --------------
    print()
    print("=" * 78)
    print("Q2  Is the observed effect consistent with the headroom's sign?")
    print("=" * 78)
    print("  (a negative headroom means the base ranking is already better than")
    print("   any reordering of the pool, so a re-ranker cannot help)")
    print()

    for label, key in (("fusion", "_effect"), ("CE-alone", "_ce_effect")):
        viol = [c for c in cells
                if c["_headroom"] < 0 and c[key] > 0]
        opp = [c for c in cells
               if c["_headroom"] > 0 and c[key] < 0]
        print(f"  {label}:")
        print(f"    cells with headroom < 0 but effect > 0 (bound violated): {len(viol)}")
        if viol:
            for c in viol[:8]:
                print(f"       {c['corpus']}/{c['stage']}/{c['reranker'].split('/')[-1]}"
                      f"/w{c['window']}: headroom={c['_headroom']:+.4f} "
                      f"effect={c[key]:+.4f}")
        print(f"    cells with headroom > 0 but effect < 0 (headroom unused): {len(opp)}")
        print()

    # headroom availability: how much room is there at all
    hs = np.array([c["_headroom"] for c in cells])
    print(f"  headroom distribution over all 660 cells:")
    print(f"    min={hs.min():+.4f}  median={np.median(hs):+.4f}  max={hs.max():+.4f}")
    print(f"    cells with headroom <= 0: {int((hs <= 0).sum())}")
    print(f"    cells with headroom > 0:  {int((hs > 0).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
