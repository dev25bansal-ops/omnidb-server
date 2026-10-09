"""Verify that caching CE scores by (query, document) is rank-safe.

A cross-encoder's score is a function of the two input texts, not of the
candidate pool the pair happened to be retrieved into. The study therefore
scores the same pair many times over -- a document that appears under window
100 of the hybrid stage also appears under window 50, and under the dense and
lexical stages -- so ~83% of its pair-scorings are repeats. Caching them would
remove that fraction of the run's compute.

This checks the cache against rows already scored *without* it. Two things are
reported:

1. **Counterfactual replay.** Every row is re-scored with a first-occurrence
   cache and the CE ordering, its nDCG@10, and the fused ranking at every
   (alpha, k) on the grid are compared with what the uncached run stored. This
   is the decision quantity: if the counted rankings agree, no reported number
   can move.

2. **Stability margin.** Independently of that replay, it counts the rows that
   contain a *hazard*: two documents inside the top 10 whose relevance grades
   differ and whose CE scores are within the observed cross-row noise. Those
   are the only pairs a reordering could move nDCG with, so the count bounds
   how much luck the replay result rests on.

Usage: python bench/verify_score_cache.py [--reranker M] [--partition dev]
Writes bench/score-cache-verification.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ce_study_runner import (ALPHAS, RUNS, RRF_KS, qrels_metrics,  # noqa: E402
                             rank_rrf)

TOL = 1.2e-05  # max cross-row score spread measured on the MiniLM dev rows


def _qrels_path(corpus: str) -> Path:
    return HERE / ".cache3" / "beir" / f"{corpus}-qrels-test.tsv"


def _check_one(path: Path, reranker: str) -> dict:
    cache: dict[tuple, float] = {}
    qrels_cache: dict[tuple, dict] = {}
    n_rows = n_pair_scorings = 0
    identical_order = 0
    differing = []
    fused_checked = fused_differ = 0
    hazard_rows = hazard_pairs = 0
    hazard_examples = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["reranker"] != reranker:
            continue
        n_rows += 1
        pool, stored = row["pool_ids"], row["ce_scores"]
        if len(pool) != len(stored):
            raise ValueError(f"misaligned row {row['qid']}")

        # Replay with a first-occurrence cache: pairs already seen in an earlier
        # row reuse that score, exactly as the cache would serve them.
        replayed = []
        for doc, score in zip(pool, stored):
            key = (row["corpus"], row["qid"], doc)
            hit = cache.get(key)
            if hit is None:
                cache[key] = score
                hit = score
            replayed.append(hit)
        n_pair_scorings += len(pool)

        qk = (row["corpus"], row["qid"])
        if qk not in qrels_cache:
            qrels_cache[qk] = _load_qrels(_qrels_path(row["corpus"]), row["qid"])
        rel = qrels_cache[qk]

        order = [pool[j] for j in
                 sorted(range(len(replayed)), key=lambda j: (-replayed[j], j))]
        if order == row["ce_ranking"]:
            identical_order += 1

        # Every counted ranking on the grid, not just the CE order.
        for alpha in ALPHAS:
            for k_rrf in RRF_KS:
                fused_checked += 1
                a = rank_rrf(pool, stored, alpha, k_rrf)
                b = rank_rrf(pool, replayed, alpha, k_rrf)
                if a != b:
                    fused_differ += 1

        if order != row["ce_ranking"] or \
                qrels_metrics(order, rel)["ndcg10"] != qrels_metrics(row["ce_ranking"], rel)["ndcg10"]:
            differing.append({
                "corpus": row["corpus"], "stage": row["stage"],
                "stage_embedder": row["stage_embedder"], "window": row["window"],
                "qid": row["qid"],
                "stored_ndcg10": qrels_metrics(row["ce_ranking"], rel)["ndcg10"],
                "cached_ndcg10": qrels_metrics(order, rel)["ndcg10"],
                "first_divergence_rank": next(
                    (i for i, (a, b) in enumerate(zip(order, row["ce_ranking"])) if a != b),
                    None),
            })

        # Stability margin: grade-differing, near-tied pairs inside the top 10.
        top = sorted(range(len(stored)), key=lambda j: (-stored[j], j))[:10]
        row_hazard = 0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                ia, ib = top[i], top[j]
                gi, gj = rel.get(pool[ia], 0), rel.get(pool[ib], 0)
                if gi != gj and abs(stored[ia] - stored[ib]) <= TOL:
                    row_hazard += 1
                    if len(hazard_examples) < 5:
                        hazard_examples.append({
                            "corpus": row["corpus"], "stage": row["stage"],
                            "window": row["window"], "qid": row["qid"],
                            "top10_positions": [i, j],
                            "docs": [pool[ia], pool[ib]], "grades": [gi, gj],
                            "score_gap": abs(stored[ia] - stored[ib]),
                        })
        if row_hazard:
            hazard_rows += 1
            hazard_pairs += row_hazard

    unique = len(cache)
    ndcg_moved = sum(1 for d in differing
                     if d["stored_ndcg10"] != d["cached_ndcg10"])
    return {
        "reranker": reranker, "rows": n_rows, "pair_scorings": n_pair_scorings,
        "unique_pairs": unique,
        "redundancy": round(n_pair_scorings / unique, 3) if unique else None,
        "compute_saved_fraction": round(1 - unique / n_pair_scorings, 4) if n_pair_scorings else None,
        "rows_with_identical_ce_order": identical_order,
        "rows_with_differing_ce_order": n_rows - identical_order,
        "rows_whose_ndcg10_moved": ndcg_moved,
        "fused_rankings_compared": fused_checked,
        "fused_rankings_that_differ": fused_differ,
        "min_first_divergence_rank": min(
            (d["first_divergence_rank"] for d in differing
             if d["first_divergence_rank"] is not None), default=None),
        "hazard_rows": hazard_rows,
        "hazard_row_fraction": round(hazard_rows / n_rows, 6) if n_rows else None,
        "hazard_pairs": hazard_pairs,
        "hazard_tolerance": TOL,
        "hazard_examples": hazard_examples,
        "differing_examples": differing[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="ce-study-v1-20261009")
    ap.add_argument("--reranker", action="append", default=None,
                    help="repeatable; defaults to every reranker in the file")
    ap.add_argument("--partition", default="dev")
    ap.add_argument("--json", default=str(HERE / "score-cache-verification.json"))
    args = ap.parse_args()

    path = RUNS / args.run_id / f"{args.partition}-scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    present = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)["reranker"]
            if r not in present:
                present.append(r)
    targets = args.reranker or present

    results = {}
    for reranker in targets:
        results[reranker] = _check_one(path, reranker)
        r = results[reranker]
        print(f"{reranker} / {args.partition}")
        print(f"  rows                   {r['rows']:,}")
        print(f"  pair-scorings          {r['pair_scorings']:,}")
        print(f"  unique (qid, doc)      {r['unique_pairs']:,}  ({r['redundancy']}x redundancy)")
        print(f"  compute a cache saves  {100*r['compute_saved_fraction']:.1f}%")
        print(f"  CE order identical     {r['rows_with_identical_ce_order']:,} / {r['rows']:,}")
        print(f"  rows whose nDCG moved  {r['rows_whose_ndcg10_moved']:,}")
        print(f"  fused rankings compared{r['fused_rankings_compared']:>9,}")
        print(f"  fused rankings differ  {r['fused_rankings_that_differ']:,}")
        print(f"  hazard rows (top-10 grade-differing pair within {TOL:g})"
              f"  {r['hazard_rows']:,} ({100*r['hazard_row_fraction']:.3f}%)")
        for e in r["hazard_examples"][:2]:
            print(f"    e.g. {e['corpus']}/{e['stage']}/w{e['window']}/{e['qid']} "
                  f"grades {e['grades']} gap {e['score_gap']:.2e}")
        print()

    summary = {
        "run_id": args.run_id, "partition": args.partition,
        "checked_unix": time.time(), "tolerance": TOL, "rerankers": results,
    }
    dest = Path(args.json)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, dest)

    moved = sum(r["rows_whose_ndcg10_moved"] for r in results.values())
    fused = sum(r["fused_rankings_that_differ"] for r in results.values())
    print(f"wrote {dest}")
    print(f"VERDICT: {'RANK-SAFE' if not (moved or fused) else 'NOT SAFE'} "
          f"(nDCG moved on {moved} rows, {fused} fused rankings differ)")
    return 0 if not (moved or fused) else 1


def _load_qrels(path: Path, qid: str) -> dict:
    rel = {}
    if not path.exists():
        return rel
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == qid:
            rel[parts[1]] = int(parts[2])
    return rel


if __name__ == "__main__":
    sys.exit(main())