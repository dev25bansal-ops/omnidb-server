# Research benchmark: OmniDB (hybrid, weighted-RRF) vs vector-only engines

- datasets: BEIR nfcorpus, scifact (real IR benchmarks, human qrels), cap 2000 docs, k=10
- embedder: all-MiniLM-L6-v2 shared across ALL engines (vector parity)
- repetitions: 5 seeds, 95% CI = mean ± 1.96·SEM
- OmniDB runs hybrid BM25+dense with dense_weight=0.7 (the fix under test);
  sqlite-vec/LanceDB/Qdrant are vector-only baselines (they have no lexical leg).

## nfcorpus

| engine | nDCG@10 (95% CI) | recall@10 | insert s | p50 ms | p95 ms |
|---|---|---|---|---|---|
| omnidb | 0.2139 ± 0.0000 | 0.1092 | 1.49 | 3.75 | 7.60 |
| omnidb-norr | 0.3729 ± 0.0000 | 0.1684 | 1.43 | 3.79 | 7.47 |
| sqlite-vec | 0.3514 ± 0.0000 | 0.1631 | 0.02 | 2.03 | 2.41 |
| lancedb | 0.3510 ± 0.0000 | 0.1631 | 0.05 | 5.24 | 6.28 |
| qdrant | 0.3515 ± 0.0000 | 0.1631 | 3.36 | 3.31 | 4.09 |

### dense_weight sweep (OmniDB nDCG@10, mean ± CI)

| dense_weight | nDCG@10 |
|---|---|
| 1.0 | 0.2049 ± 0.0000 |
| 0.9 | 0.2094 ± 0.0000 |
| 0.7 | 0.2139 ± 0.0000 |
| 0.5 | 0.2107 ± 0.0000 |
| 0.3 | 0.2019 ± 0.0000 |

## scifact

| engine | nDCG@10 (95% CI) | recall@10 | insert s | p50 ms | p95 ms |
|---|---|---|---|---|---|
| omnidb | 0.3229 ± 0.0000 | 0.7000 | 1.48 | 7.88 | 11.79 |
| omnidb-norr | 0.7909 ± 0.0000 | 0.9034 | 1.42 | 7.70 | 11.49 |
| sqlite-vec | 0.7152 ± 0.0000 | 0.8414 | 0.02 | 2.00 | 2.38 |
| lancedb | 0.7152 ± 0.0000 | 0.8414 | 0.05 | 5.31 | 6.36 |
| qdrant | 0.7152 ± 0.0000 | 0.8414 | 3.32 | 3.28 | 4.00 |

### dense_weight sweep (OmniDB nDCG@10, mean ± CI)

| dense_weight | nDCG@10 |
|---|---|
| 1.0 | 0.2902 ± 0.0000 |
| 0.9 | 0.3073 ± 0.0000 |
| 0.7 | 0.3229 ± 0.0000 |
| 0.5 | 0.3279 ± 0.0000 |
| 0.3 | 0.3405 ± 0.0000 |

## Conclusion

**Finding 1 (negative, mechanistic):** memory-oriented reranking silently degrades
corpus retrieval. OmniDB's default config (recency/frequency boosts tuned for agent
memory) loses 40-56% nDCG@10 vs its own rerank-off path on standard BEIR benchmarks
(0.214→0.373 on nfcorpus; 0.323→0.791 on scifact). Mechanism: all corpus documents
share near-identical age/access distributions, so the boosts act as arbitrary
tie-breakers that reorder near-tied cosine neighbors.

**Finding 2 (positive):** with the memory-rerank disabled, weighted-RRF hybrid
(BM25+dense, dense_weight=0.7) **matches or beats every dedicated vector-only engine**
on both datasets — nfcorpus 0.3729 vs 0.3514/0.3510/0.3515 (sqlite-vec/LanceDB/Qdrant);
scifact 0.7909 vs 0.7152 across the board. The lexical leg adds real value on
title+abstract corpora when it is not fighting the memory-rerank.

**Finding 3 (tuning surface):** the optimal dense_weight is dataset-dependent
(0.7 on nfcorpus; ≤0.3 on scifact), so fusion weights are a per-deployment knob,
not a universal constant.

**Honesty notes:** engines are deterministic — cross-seed variance is exactly zero,
so CIs are degenerate (±0.0000) and reported as such rather than dressed up as
statistics. Scale is 2 datasets × 2000 docs × 1 embedder; no user study. The
earlier harness bug (LanceDB rows silently doubled by an "insert parity" step,
producing nDCG > 1.0) was caught by an invariant check, proven, and fixed — the
corrected numbers above supersede any earlier version of this file.

