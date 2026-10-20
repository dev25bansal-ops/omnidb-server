"""Query-clustered analysis of the pool-oracle headroom as a predictor.

This is the analysis the manuscript currently defers to future work
("conduct a formal query-clustered predictor analysis using the per-cell
pool-oracle and base-nDCG quantities reported in Table VI").

It reads the per-query test results (test-results.jsonl) and answers two
questions that the Mechanism section currently asserts it cannot answer:

  A. Does the pool-oracle headroom predict the re-ranker's per-query effect,
     once the dependence between a query's observations across cells is
     handled by clustering on the query?

  B. Is the sign of the headroom a valid bound?  A re-ranker can only beat
     the base if the pool contains a better ordering than the base chose.
     This is a per-query statement, so it can be checked exactly.

The query-clustered inference uses a cluster bootstrap: queries are resampled
with replacement, and all of a query's observations move with it.  The
statistic is the Spearman rho between a query's headroom and its effect,
computed within each cell and then averaged over cells (so that the
correlation is not driven by between-corpus scale differences), with the
resampling done at the query level.

Implementation note: the data are held as flat numpy arrays with a
query-index vector, and the bootstrap resamples query indices rather than
rebuilding per-cell dictionaries, so 2,000 resamples over 660 cells is a
matter of seconds rather than hours.

Usage:
    python bench/oracle_headroom_clustered.py [--run-id ce-study-v1-20261009]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="ce-study-v1-20261009")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="limit rows read (0 = all); for smoke testing")
    args = ap.parse_args()

    run = HERE / "ce-study-runs" / args.run_id
    src = run / "test-results.jsonl"

    print(f"reading {src} ...")
    cell_of: list[str] = []
    qid_of: list[str] = []
    corpus_of: list[str] = []
    headroom: list[float] = []
    neg_base: list[float] = []
    fusion: list[float] = []
    replacement: list[float] = []
    qid_index: dict[str, int] = {}

    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("partition") != "test":
                continue
            cell_of.append(
                f"{r['corpus']}|{r['stage']}|{r['stage_embedder']}|"
                f"{r['reranker']}|w{r['window']}")
            qid = r["qid"]
            qi = qid_index.get(qid)
            if qi is None:
                qi = len(qid_index)
                qid_index[qid] = qi
            qid_of.append(qid)
            corpus_of.append(r["corpus"])
            headroom.append(r["oracle_ndcg10"] - r["base_ndcg10"])
            neg_base.append(-r["base_ndcg10"])
            fusion.append(r["blend_ndcg10"] - r["base_ndcg10"])
            replacement.append(r["ce_ndcg10"] - r["base_ndcg10"])
            if args.max_rows and len(cell_of) >= args.max_rows:
                break

    cell_arr = np.array(cell_of)
    qidx = np.array([qid_index[q] for q in qid_of])
    H = np.array(headroom)
    NB = np.array(neg_base)
    F = np.array(fusion)
    R = np.array(replacement)
    cells = np.unique(cell_arr)
    n_q = len(qid_index)

    print(f"  {len(cell_arr)} test rows -> {len(cells)} cells, "
          f"{n_q} distinct queries")

    meta_path = run / "test-summary.json"
    if meta_path.exists():
        declared = json.loads(meta_path.read_text(encoding="utf-8"))[
            "metadata"]["family_size_predeclared"]
        if len(cells) != declared:
            print(f"  FAIL: {len(cells)} cells present but the declared family "
                  f"is {declared}")
            return 1
        print(f"  cell count matches the declared family size ({declared})")
    print()

    # ---- B: is the sign of the headroom a valid per-query bound? --------
    print("=" * 78)
    print("B  Is the headroom sign a valid bound on the per-query effect?")
    print("=" * 78)
    print("  If headroom <= 0 the base ordering is already optimal for that")
    print("  pool, so no reordering of it can score higher.")
    print()
    for arr, label in ((F, "dev-selected fusion"),
                       (R, "pure replacement (CE-alone)")):
        tot = len(arr)
        viol_pos = int(np.sum((H <= 0) & (arr > 1e-12)))
        unused = int(np.sum((H > 0) & (arr < -1e-12)))
        print(f"  {label}:")
        print(f"    observations                                  : {tot}")
        print(f"    headroom <= 0 and effect > 0 (bound violated) : {viol_pos}")
        print(f"    headroom > 0 and effect < 0 (headroom unused) : {unused}")
        print(f"    total sign disagreements                      : "
              f"{viol_pos + unused} ({100.0 * (viol_pos + unused) / max(tot, 1):.2f}%)")
        print()

    # ---- A: query-clustered correlation ---------------------------------
    print("=" * 78)
    print("A  Query-clustered correlation between headroom and effect")
    print("=" * 78)
    print("  Statistic: within each cell, the Spearman rho between a query's")
    print("  headroom and its effect; averaged over cells.  The cluster")
    print("  bootstrap resamples QUERIES, carrying every one of a query's")
    print("  observations with it.")
    print()

    # Precompute each cell's row indices and its query index vector.
    cell_rows = {c: np.flatnonzero(cell_arr == c) for c in cells}
    cell_qidx = {c: qidx[rows] for c, rows in cell_rows.items()}

    def mean_within_rho(sample_qidx: np.ndarray, y: np.ndarray,
                        x: np.ndarray) -> float:
        rhos = []
        for c in cells:
            rows = cell_rows[c]
            qc = cell_qidx[c]
            # map each observation's query to its slot in the resample
            pos = sample_qidx[qc]
            # collapse to one row per resampled query occurrence
            uniq, inv = np.unique(pos, return_inverse=True)
            if len(uniq) < 3:
                continue
            xs = np.zeros(len(uniq))
            ys = np.zeros(len(uniq))
            np.add.at(xs, inv, x[rows])
            np.add.at(ys, inv, y[rows])
            cnt = np.bincount(inv, minlength=len(uniq))
            r = _spearman(xs / cnt, ys / cnt)
            if not np.isnan(r):
                rhos.append(r)
        return float(np.mean(rhos)) if rhos else float("nan")

    rng = np.random.default_rng(args.seed)
    base_qidx = np.arange(n_q)
    for arr, label in ((F, "dev-selected fusion"),
                       (R, "pure replacement (CE-alone)")):
        point = mean_within_rho(base_qidx, arr, H)
        boots = []
        for _ in range(args.n_boot):
            samp = rng.integers(0, n_q, size=n_q)
            v = mean_within_rho(samp, arr, H)
            if not np.isnan(v):
                boots.append(v)
        boots = np.array(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        print(f"  {label}:")
        print(f"    mean within-cell Spearman rho = {point:+.4f}")
        print(f"    cluster bootstrap 95% CI      = [{lo:+.4f}, {hi:+.4f}]")
        print(f"    two-sided p (vs 0)             = {min(p, 1.0):.4f}")
        print()

    # ---- A1: control -- is the headroom better than the base level? -----
    print("=" * 78)
    print("A1  Control: does the headroom beat the base-nDCG level itself?")
    print("=" * 78)
    print("  headroom = oracle - base is mechanically larger when base is")
    print("  small, so it is compared against -base_ndcg (the same mechanical")
    print("  component, without the headroom).")
    print()
    for arr, label in ((F, "dev-selected fusion"),
                       (R, "pure replacement (CE-alone)")):
        h_point = mean_within_rho(base_qidx, arr, H)
        b_point = mean_within_rho(base_qidx, arr, NB)
        print(f"  {label}:")
        print(f"    rho(effect, headroom)   = {h_point:+.4f}")
        print(f"    rho(effect, -base_nDCG) = {b_point:+.4f}")
        print(f"    difference (headroom - base-level) = {h_point - b_point:+.4f}")
        print()

    # ---- A2: the between-corpus version (n=4, the one the paper quotes) --
    print("  Across corpora (hybrid stage, one point per corpus, n=4):")
    corpora = np.array(corpus_of)
    is_hybrid = np.char.startswith(cell_arr, "nfcorpus|hybrid|") | \
        np.char.startswith(cell_arr, "scifact|hybrid|") | \
        np.char.startswith(cell_arr, "scidocs|hybrid|") | \
        np.char.startswith(cell_arr, "fiqa|hybrid|")
    for arr, label in ((F, "fusion"), (R, "CE-alone")):
        hs, es = [], []
        for corp in ("fiqa", "nfcorpus", "scidocs", "scifact"):
            m = is_hybrid & (corpora == corp)
            hq = defaultdict(list)
            eq = defaultdict(list)
            for qi, hh, ee in zip(qidx[m], H[m], arr[m]):
                hq[qi].append(hh)
                eq[qi].append(ee)
            qh = float(np.mean([np.mean(v) for v in hq.values()]))
            qe = float(np.mean([np.mean(v) for v in eq.values()]))
            hs.append(qh)
            es.append(qe)
            print(f"    {corp:<10} headroom={qh:+.4f}  effect={qe:+.4f}")
        hs = np.array(hs)
        es = np.array(es)
        print(f"    Spearman rho ({label}) = {_spearman(hs, es):+.4f}  (n=4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
