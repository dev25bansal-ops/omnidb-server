# Research benchmark v3: full-size corpora, BM25 baseline, cross-encoder

- datasets (FULL SIZE, no cap): nfcorpus (3633 docs), scifact (5183 docs), scidocs (25657 docs), fiqa (57638 docs)
- embedders: minilm, bge-small, e5-small, bge-base (prefixes applied,
  identical vectors to every engine)
- baselines: bm25s (Lucene-equivalent BM25, k1=0.9 b=0.4) + qdrant
  (vector-only); omnidb = corpus_mode + weighted RRF 0.7/0.3;
  omnidb-dense-only = fusion 1.0/0.0 control; omnidb-ce = cross-encoder
  (BAAI/bge-reranker-base) rerank of omnidb top-50
- statistics: bootstrap CIs over queries; paired Wilcoxon + Holm;
  between-process latency spread (5 fresh subprocesses per engine)
- touche-2020 excluded: 382k docs ~ 10h+ embedding for a dataset that
  was metric-degenerate at 2k (v2 finding)

## nfcorpus
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3425 (0.3081–0.3783) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0379 | 0.0007 | +0.303 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0213 | 0.0074 | +0.219 |
| omnidb dense-only | 0.3182 (0.2848–0.3523) | -0.0243 | 0.0003 | +0.342 |
| omnidb + bge-reranker-base | 0.3134 (0.2798–0.3482) | -0.0292 | 0.0023 | +0.272 |
| qdrant (vector-only) | 0.3180 (0.2845–0.3521) | -0.0246 | 0.0003 | +0.343 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3617 (0.3264–0.3982) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0570 | 0.0000 | +0.481 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0405 | 0.0000 | +0.404 |
| omnidb dense-only | 0.3435 (0.3084–0.3789) | -0.0182 | 0.0040 | +0.259 |
| omnidb + bge-reranker-base | 0.3185 (0.2846–0.3531) | -0.0431 | 0.0000 | +0.384 |
| qdrant (vector-only) | 0.3431 (0.3079–0.3786) | -0.0185 | 0.0040 | +0.264 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3425 (0.3080–0.3780) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0378 | 0.0001 | +0.344 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0213 | 0.0034 | +0.242 |
| omnidb dense-only | 0.3242 (0.2902–0.3592) | -0.0183 | 0.0000 | +0.391 |
| omnidb + bge-reranker-base | 0.3150 (0.2813–0.3494) | -0.0275 | 0.0020 | +0.278 |
| qdrant (vector-only) | 0.3240 (0.2901–0.3591) | -0.0185 | 0.0000 | +0.384 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3747 (0.3397–0.4110) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0700 | 0.0000 | +0.556 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0535 | 0.0000 | +0.535 |
| omnidb dense-only | 0.3723 (0.3374–0.4082) | -0.0024 | 1.0000 | +0.041 |
| omnidb + bge-reranker-base | 0.3280 (0.2940–0.3630) | -0.0467 | 0.0000 | +0.414 |
| qdrant (vector-only) | 0.3727 (0.3378–0.4087) | -0.0020 | 1.0000 | +0.033 |

### between-process latency spread (nfcorpus, minilm, 5 fresh processes)

- omnidb: p50 across 5/5 processes = [8.100399994873442, 8.4703000029549, 8.560200003557838, 8.809500010102056, 9.024599989061244] ms
- qdrant: p50 across 5/5 processes = [4.597999999532476, 4.744500009110197, 5.680999995092861, 5.950399994617328, 6.069800001569092] ms

## scifact
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7052 (0.6620–0.7470) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0484 | 0.0174 | +0.280 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0263 | 0.0879 | +0.221 |
| omnidb dense-only | 0.6484 (0.6029–0.6917) | -0.0568 | 0.0000 | +0.726 |
| omnidb + bge-reranker-base | 0.7068 (0.6629–0.7497) | +0.0016 | 0.8696 | -0.033 |
| qdrant (vector-only) | 0.6484 (0.6029–0.6917) | -0.0568 | 0.0000 | +0.726 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7301 (0.6886–0.7703) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0733 | 0.0001 | +0.450 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0512 | 0.0010 | +0.392 |
| omnidb dense-only | 0.7057 (0.6618–0.7478) | -0.0244 | 0.0218 | +0.334 |
| omnidb + bge-reranker-base | 0.7239 (0.6822–0.7652) | -0.0062 | 0.5259 | +0.074 |
| qdrant (vector-only) | 0.7057 (0.6618–0.7478) | -0.0244 | 0.0218 | +0.334 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7294 (0.6871–0.7698) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0725 | 0.0001 | +0.447 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0505 | 0.0016 | +0.375 |
| omnidb dense-only | 0.6955 (0.6502–0.7374) | -0.0339 | 0.0023 | +0.404 |
| omnidb + bge-reranker-base | 0.7183 (0.6754–0.7599) | -0.0110 | 0.3062 | +0.115 |
| qdrant (vector-only) | 0.6955 (0.6502–0.7374) | -0.0339 | 0.0023 | +0.404 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7508 (0.7120–0.7891) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0939 | 0.0000 | +0.533 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0719 | 0.0000 | +0.481 |
| omnidb dense-only | 0.7421 (0.7017–0.7809) | -0.0087 | 0.3410 | +0.179 |
| omnidb + bge-reranker-base | 0.7232 (0.6822–0.7645) | -0.0275 | 0.0825 | +0.249 |
| qdrant (vector-only) | 0.7421 (0.7017–0.7809) | -0.0087 | 0.3410 | +0.179 |

### between-process latency spread (scifact, minilm, 5 fresh processes)

- omnidb: p50 across 5/5 processes = [19.87340000050608, 20.44610001030378, 20.613399989088066, 20.782399995368905, 21.55059999495279] ms
- qdrant: p50 across 5/5 processes = [8.150800000294112, 8.559000008972362, 8.577399989007972, 8.656000005430542, 8.733200011192821] ms

## scidocs
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2069 (0.1932–0.2208) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0595 | 0.0000 | +0.652 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0579 | 0.0000 | +0.645 |
| omnidb dense-only | 0.2170 (0.2031–0.2313) | +0.0101 | 0.0345 | -0.115 |
| omnidb + bge-reranker-base | 0.1755 (0.1627–0.1882) | -0.0314 | 0.0000 | +0.340 |
| qdrant (vector-only) | 0.2170 (0.2031–0.2313) | +0.0101 | 0.0345 | -0.115 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2002 (0.1866–0.2138) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0529 | 0.0000 | +0.604 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0512 | 0.0000 | +0.608 |
| omnidb dense-only | 0.2081 (0.1942–0.2222) | +0.0079 | 0.1069 | -0.097 |
| omnidb + bge-reranker-base | 0.1740 (0.1610–0.1869) | -0.0263 | 0.0000 | +0.294 |
| qdrant (vector-only) | 0.2081 (0.1942–0.2222) | +0.0079 | 0.1069 | -0.097 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.1802 (0.1668–0.1935) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0329 | 0.0000 | +0.411 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0312 | 0.0000 | +0.402 |
| omnidb dense-only | 0.1777 (0.1642–0.1909) | -0.0026 | 0.0840 | +0.107 |
| omnidb + bge-reranker-base | 0.1587 (0.1463–0.1711) | -0.0215 | 0.0000 | +0.267 |
| qdrant (vector-only) | 0.1777 (0.1642–0.1909) | -0.0026 | 0.0840 | +0.107 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2075 (0.1936–0.2213) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0602 | 0.0000 | +0.659 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0585 | 0.0000 | +0.657 |
| omnidb dense-only | 0.2185 (0.2041–0.2328) | +0.0110 | 0.0102 | -0.139 |
| omnidb + bge-reranker-base | 0.1745 (0.1616–0.1874) | -0.0330 | 0.0000 | +0.363 |
| qdrant (vector-only) | 0.2185 (0.2041–0.2328) | +0.0110 | 0.0102 | -0.139 |

### between-process latency spread (scidocs, minilm, 5 fresh processes)

- omnidb: p50 across 5/5 processes = [70.4571000096621, 71.48030000098515, 71.70869999390561, 72.58069999807049, 72.77420000173151] ms
- qdrant: p50 across 5/5 processes = [40.02519999630749, 40.396800002781674, 40.87660000368487, 41.430999990552664, 41.78140001022257] ms

## fiqa
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3796 (0.3518–0.4075) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1416 | 0.0000 | +0.658 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1435 | 0.0000 | +0.713 |
| omnidb dense-only | 0.3637 (0.3364–0.3913) | -0.0159 | 0.1082 | +0.131 |
| omnidb + bge-reranker-base | 0.3732 (0.3461–0.4005) | -0.0064 | 0.2677 | +0.067 |
| qdrant (vector-only) | 0.3637 (0.3364–0.3913) | -0.0159 | 0.1082 | +0.131 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3927 (0.3650–0.4199) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1547 | 0.0000 | +0.766 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1566 | 0.0000 | +0.819 |
| omnidb dense-only | 0.4025 (0.3735–0.4310) | +0.0099 | 0.4295 | -0.078 |
| omnidb + bge-reranker-base | 0.3699 (0.3429–0.3969) | -0.0228 | 0.0180 | +0.164 |
| qdrant (vector-only) | 0.4025 (0.3735–0.4310) | +0.0099 | 0.4295 | -0.078 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3824 (0.3549–0.4104) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1444 | 0.0000 | +0.669 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1463 | 0.0000 | +0.727 |
| omnidb dense-only | 0.3621 (0.3346–0.3898) | -0.0203 | 0.0097 | +0.196 |
| omnidb + bge-reranker-base | 0.3785 (0.3512–0.4059) | -0.0039 | 0.6775 | +0.026 |
| qdrant (vector-only) | 0.3621 (0.3346–0.3898) | -0.0203 | 0.0097 | +0.196 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3922 (0.3651–0.4196) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1543 | 0.0000 | +0.761 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1562 | 0.0000 | +0.818 |
| omnidb dense-only | 0.4050 (0.3769–0.4329) | +0.0127 | 0.1614 | -0.110 |
| omnidb + bge-reranker-base | 0.3728 (0.3458–0.3994) | -0.0195 | 0.0223 | +0.155 |
| qdrant (vector-only) | 0.4050 (0.3769–0.4329) | +0.0127 | 0.1614 | -0.110 |

### between-process latency spread (fiqa, minilm, 5 fresh processes)

- omnidb: p50 across 5/5 processes = [150.21259999775793, 151.4586999983294, 154.95389999705367, 165.2751000074204, 172.5252000032924] ms
- qdrant: p50 across 5/5 processes = [97.18319999228697, 100.07929999846965, 105.4465000052005, 107.75040001317393, 119.89180000091437] ms

## Honesty notes

- Cross-run variance is zero for deterministic engines; the reported
  uncertainty is over the query population (bootstrap). Between-process
  latency spread is reported separately as the honest substitute for a
  second machine (interpreter/JIT warmup is the dominant cross-process
  effect on identical hardware).
- bm25s is Lucene-equivalent BM25 (ICLR 2024). The pyserini (Anserini
  BM25) row in each table is a real run: Anserini's reference BM25 was
  executed under a portable Temurin JDK 21 (host previously had Java 8,
  which Pyserini cannot run; the JDK 21 toolchain was added for this
  benchmark). k1/b match the bm25s convention; per-corpus indexing time
  is recorded in the JSON (index_s: nfcorpus 14.2, scifact 2.7, scidocs 11.1, fiqa 104.0).
- Cross-encoder rerank measured only on omnidb's top-50; reranking
  qdrant's top-50 is the natural follow-up for a camera-ready.
- No user study; single machine (multi-process for latency); fiqa
  full size 57.6k docs; embedding cache under bench/.cache3/.
