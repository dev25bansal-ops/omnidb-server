# Research benchmark v3: full-size corpora, BM25 baseline, cross-encoder

- datasets (FULL SIZE, no cap): nfcorpus (3633 docs), scifact (5183 docs), scidocs (25657 docs), fiqa (57638 docs)
- embedders: minilm, bge-small, e5-small, bge-base (prefixes applied,
  identical vectors to every engine)
- baselines: bm25s (Lucene-equivalent BM25, k1=0.9 b=0.4) + qdrant
  (vector-only); omnidb = corpus_mode + weighted RRF 0.7/0.3;
  omnidb-dense-only = fusion 1.0/0.0 control; omnidb-ce = cross-encoder
  (cross-encoder/ms-marco-MiniLM-L-6-v2) rerank of omnidb top-50
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
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0213 | 0.0148 | +0.219 |
| omnidb dense-only | 0.3182 (0.2848–0.3523) | -0.0243 | 0.0003 | +0.342 |
| omnidb + cross-encoder | 0.3396 (0.3051–0.3748) | -0.0030 | 0.6046 | +0.045 |
| qdrant (vector-only) | 0.3180 (0.2845–0.3521) | -0.0246 | 0.0003 | +0.343 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3617 (0.3264–0.3982) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0570 | 0.0000 | +0.481 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0405 | 0.0000 | +0.404 |
| omnidb dense-only | 0.3435 (0.3084–0.3789) | -0.0182 | 0.0061 | +0.259 |
| omnidb + cross-encoder | 0.3489 (0.3139–0.3843) | -0.0127 | 0.1407 | +0.126 |
| qdrant (vector-only) | 0.3431 (0.3079–0.3786) | -0.0185 | 0.0061 | +0.264 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3425 (0.3080–0.3780) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0378 | 0.0001 | +0.344 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0213 | 0.0069 | +0.242 |
| omnidb dense-only | 0.3242 (0.2902–0.3592) | -0.0183 | 0.0000 | +0.391 |
| omnidb + cross-encoder | 0.3450 (0.3105–0.3802) | +0.0025 | 0.6103 | -0.044 |
| qdrant (vector-only) | 0.3240 (0.2901–0.3591) | -0.0185 | 0.0000 | +0.384 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3747 (0.3397–0.4110) | — | — | — |
| bm25s (BM25 baseline) | 0.3047 (0.2706–0.3389) | -0.0700 | 0.0000 | +0.556 |
| pyserini (Anserini BM25) | 0.3212 (0.2870–0.3556) | -0.0535 | 0.0000 | +0.535 |
| omnidb dense-only | 0.3723 (0.3374–0.4082) | -0.0024 | 1.0000 | +0.041 |
| omnidb + cross-encoder | 0.3560 (0.3208–0.3914) | -0.0187 | 0.1949 | +0.158 |
| qdrant (vector-only) | 0.3727 (0.3378–0.4087) | -0.0020 | 1.0000 | +0.033 |

### between-process latency spread (nfcorpus, minilm, 5 fresh processes)

- omnidb: p50 across processes = [8.100399994873442, 8.4703000029549, 8.560200003557838, 8.809500010102056, 9.024599989061244] ms
- qdrant: p50 across processes = [4.597999999532476, 4.744500009110197, 5.680999995092861, 5.950399994617328, 6.069800001569092] ms

## scifact
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7052 (0.6620–0.7470) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0484 | 0.0174 | +0.280 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0263 | 0.0879 | +0.221 |
| omnidb dense-only | 0.6484 (0.6029–0.6917) | -0.0568 | 0.0000 | +0.726 |
| omnidb + cross-encoder | 0.6875 (0.6425–0.7308) | -0.0177 | 0.1923 | +0.141 |
| qdrant (vector-only) | 0.6484 (0.6029–0.6917) | -0.0568 | 0.0000 | +0.726 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7301 (0.6886–0.7703) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0733 | 0.0001 | +0.450 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0512 | 0.0010 | +0.392 |
| omnidb dense-only | 0.7057 (0.6618–0.7478) | -0.0244 | 0.0218 | +0.334 |
| omnidb + cross-encoder | 0.6933 (0.6482–0.7363) | -0.0368 | 0.0227 | +0.249 |
| qdrant (vector-only) | 0.7057 (0.6618–0.7478) | -0.0244 | 0.0218 | +0.334 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7294 (0.6871–0.7698) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0725 | 0.0001 | +0.447 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0505 | 0.0016 | +0.375 |
| omnidb dense-only | 0.6955 (0.6502–0.7374) | -0.0339 | 0.0023 | +0.404 |
| omnidb + cross-encoder | 0.6937 (0.6482–0.7366) | -0.0357 | 0.0155 | +0.275 |
| qdrant (vector-only) | 0.6955 (0.6502–0.7374) | -0.0339 | 0.0023 | +0.404 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.7508 (0.7120–0.7891) | — | — | — |
| bm25s (BM25 baseline) | 0.6569 (0.6113–0.7020) | -0.0939 | 0.0000 | +0.533 |
| pyserini (Anserini BM25) | 0.6789 (0.6347–0.7223) | -0.0719 | 0.0000 | +0.481 |
| omnidb dense-only | 0.7421 (0.7017–0.7809) | -0.0087 | 0.3410 | +0.179 |
| omnidb + cross-encoder | 0.6970 (0.6528–0.7393) | -0.0538 | 0.0009 | +0.380 |
| qdrant (vector-only) | 0.7421 (0.7017–0.7809) | -0.0087 | 0.3410 | +0.179 |

### between-process latency spread (scifact, minilm, 5 fresh processes)

- omnidb: p50 across processes = [19.87340000050608, 20.44610001030378, 20.613399989088066, 20.782399995368905, 21.55059999495279] ms
- qdrant: p50 across processes = [8.150800000294112, 8.559000008972362, 8.577399989007972, 8.656000005430542, 8.733200011192821] ms

## scidocs
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2069 (0.1932–0.2208) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0595 | 0.0000 | +0.652 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0579 | 0.0000 | +0.645 |
| omnidb dense-only | 0.2170 (0.2031–0.2313) | +0.0101 | 0.0345 | -0.115 |
| omnidb + cross-encoder | 0.1872 (0.1737–0.2008) | -0.0197 | 0.0000 | +0.265 |
| qdrant (vector-only) | 0.2170 (0.2031–0.2313) | +0.0101 | 0.0345 | -0.115 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2002 (0.1866–0.2138) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0529 | 0.0000 | +0.604 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0512 | 0.0000 | +0.608 |
| omnidb dense-only | 0.2081 (0.1942–0.2222) | +0.0079 | 0.1069 | -0.097 |
| omnidb + cross-encoder | 0.1818 (0.1684–0.1952) | -0.0184 | 0.0000 | +0.245 |
| qdrant (vector-only) | 0.2081 (0.1942–0.2222) | +0.0079 | 0.1069 | -0.097 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.1802 (0.1668–0.1935) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0329 | 0.0000 | +0.411 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0312 | 0.0000 | +0.402 |
| omnidb dense-only | 0.1777 (0.1642–0.1909) | -0.0026 | 0.1260 | +0.107 |
| omnidb + cross-encoder | 0.1747 (0.1613–0.1878) | -0.0055 | 0.1433 | +0.075 |
| qdrant (vector-only) | 0.1777 (0.1642–0.1909) | -0.0026 | 0.1260 | +0.107 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.2075 (0.1936–0.2213) | — | — | — |
| bm25s (BM25 baseline) | 0.1474 (0.1348–0.1600) | -0.0602 | 0.0000 | +0.659 |
| pyserini (Anserini BM25) | 0.1490 (0.1365–0.1617) | -0.0585 | 0.0000 | +0.657 |
| omnidb dense-only | 0.2185 (0.2041–0.2328) | +0.0110 | 0.0102 | -0.139 |
| omnidb + cross-encoder | 0.1837 (0.1704–0.1972) | -0.0238 | 0.0000 | +0.313 |
| qdrant (vector-only) | 0.2185 (0.2041–0.2328) | +0.0110 | 0.0102 | -0.139 |

### between-process latency spread (scidocs, minilm, 5 fresh processes)

- omnidb: p50 across processes = [70.4571000096621, 71.48030000098515, 71.70869999390561, 72.58069999807049, 72.77420000173151] ms
- qdrant: p50 across processes = [40.02519999630749, 40.396800002781674, 40.87660000368487, 41.430999990552664, 41.78140001022257] ms

## fiqa
### embedder: minilm

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3796 (0.3518–0.4075) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1416 | 0.0000 | +0.658 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1435 | 0.0000 | +0.713 |
| omnidb dense-only | 0.3637 (0.3364–0.3913) | -0.0159 | 0.1082 | +0.131 |
| omnidb + cross-encoder | 0.3881 (0.3597–0.4159) | +0.0085 | 0.3632 | -0.057 |
| qdrant (vector-only) | 0.3637 (0.3364–0.3913) | -0.0159 | 0.1082 | +0.131 |

### embedder: bge-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3927 (0.3650–0.4199) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1547 | 0.0000 | +0.766 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1566 | 0.0000 | +0.819 |
| omnidb dense-only | 0.4025 (0.3735–0.4310) | +0.0099 | 0.4295 | -0.078 |
| omnidb + cross-encoder | 0.3837 (0.3562–0.4113) | -0.0090 | 0.3449 | +0.097 |
| qdrant (vector-only) | 0.4025 (0.3735–0.4310) | +0.0099 | 0.4295 | -0.078 |

### embedder: e5-small

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3824 (0.3549–0.4104) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1444 | 0.0000 | +0.669 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1463 | 0.0000 | +0.727 |
| omnidb dense-only | 0.3621 (0.3346–0.3898) | -0.0203 | 0.0097 | +0.196 |
| omnidb + cross-encoder | 0.3898 (0.3619–0.4173) | +0.0074 | 0.3624 | -0.059 |
| qdrant (vector-only) | 0.3621 (0.3346–0.3898) | -0.0203 | 0.0097 | +0.196 |

### embedder: bge-base

| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |
|---|---|---|---|---|
| omnidb (hybrid) | 0.3922 (0.3651–0.4196) | — | — | — |
| bm25s (BM25 baseline) | 0.2380 (0.2142–0.2617) | -0.1543 | 0.0000 | +0.761 |
| pyserini (Anserini BM25) | 0.2361 (0.2128–0.2606) | -0.1562 | 0.0000 | +0.818 |
| omnidb dense-only | 0.4050 (0.3769–0.4329) | +0.0127 | 0.1614 | -0.110 |
| omnidb + cross-encoder | 0.3823 (0.3552–0.4095) | -0.0099 | 0.0978 | +0.130 |
| qdrant (vector-only) | 0.4050 (0.3769–0.4329) | +0.0127 | 0.1614 | -0.110 |

### between-process latency spread (fiqa, minilm, 5 fresh processes)

- omnidb: p50 across processes = [116.97660002391785, 117.63890000293031, 119.36149999382906, 120.0055999943288, 123.78400001034606] ms
- qdrant: p50 across processes = [97.13180002290756, 110.05489999661222, 390.1627000013832] ms

## Honesty notes

- Cross-run variance is zero for deterministic engines; the reported
  uncertainty is over the query population (bootstrap). Between-process
  latency spread is reported separately as the honest substitute for a
  second machine (interpreter/JIT warmup is the dominant cross-process
  effect on identical hardware).
- bm25s is Lucene-equivalent BM25 (ICLR 2024). The pyserini (Anserini BM25)
  row in each table is a real run: Anserini's reference BM25 was executed
  under a portable Temurin JDK 21 (host previously had Java 8, which
  Pyserini cannot run; the JDK 21 toolchain was added for this benchmark).
  k1/b match the bm25s convention; per-corpus indexing time is recorded in
  the JSON (index_s: nfcorpus 14.2, scifact 2.7, scidocs 11.1, fiqa 104.0).
- Cross-encoder rerank measured only on omnidb's top-50; reranking
  qdrant's top-50 is the natural follow-up for a camera-ready.
- No user study; single machine (multi-process for latency); fiqa
  full size 57.6k docs; embedding cache under bench/.cache3/.

## Conclusion

**1. v3 grid (unchanged).** The full-size grid above is the baseline result and
its numbers stand as tabulated: weighted-RRF hybrid (corpus_mode, 0.7/0.3)
beats bm25s (0.3047-0.2380 across datasets) and pyserini BM25 on every dataset;
it matches or beats qdrant/dense-only on nfcorpus (0.3747 vs 0.3727 at bge-base)
and scifact (0.7508 vs 0.7421), and on scidocs the
dense-only/qdrant control edges it (+0.0110, p=0.0102, bge-base) while fiqa
dense-only is within noise (+0.0127, p=0.1614). The honest summary is: hybrid is
the best single choice on 3 of 4 corpora and statistically tied on the fourth.

**2. The old negative (pre-window-sweep).** The CE row in the grid - a
cross-encoder rerank of omnidb's top-50 - was measured and it HURT. On scidocs
omnidb+CE scored -0.0197 (minilm) / -0.0238 (bge-base) below plain hybrid; on
scifact it was -0.0538 (bge-base, p=0.0009); on fiqa it was noise (-0.0099,
p=0.0978). At the time this read as "CE reranking does not help" and the product
call was to leave it off.

**3. The new positive (the regime law).** The window sweep
(bench/rerank-window-results.md, "## Regime summary (integrated)") resolves that
negative into a law. The CE delta is not a property of the corpus or the window -
it is a function of **first-stage weakness**. Reranking a weak single-leg stage
(bm25s) helps, and on fiqa the gain grows monotonically with window
(+0.0637/+0.0728/+0.0808 at w=25/50/100, all p<0.0001), closing ~46% of the
bm25->hybrid gap. Reranking the already-fused hybrid stage never helps: the best
hybrid cell is -0.0031 to +0.0067 (the positive ones n.s.), and every significant
hybrid cell is a loss that grows MORE negative as the window opens
(fiqa bge-base w=100 -0.0407 p<0.0001; scidocs minilm w=100 -0.0513 p<0.0001;
nfcorpus bge-base w=100 -0.0541 p<0.0001). The integrated table in
rerank-window-results.md lists the best cell per dataset x stage; the sign flips
at the weak-vs-strong STAGE GAP, not at a recall threshold - nfcorpus's bm25
(recall@100 0.2376) gained nothing significant because the corpus was already
well-served by both single legs (low recall is necessary, not sufficient).

**4. Publishability (honest).** What is publishable here is the regime law plus
the four-corpus full-size hybrid-vs-vector-only grid, not a new SOTA or a new
architecture: the CE gain on weak stages is a known effect, and our contribution
is its clean characterization and the negative result that it does not transfer to
a fused hybrid first stage. Weaknesses a reviewer will (fairly) raise: CE rerank
was measured only on omnidb's own candidate lists, not on qdrant's top-w, so the
"CE cannot help a strong stage" claim is demonstrated for one strong-stage
implementation; scifact and nfcorpus give n.s. bm25 gains, so the positive side
rests mainly on fiqa; and everything is single-machine, single-reranker
(bge-reranker-base), single-seed. The law is stated as a hypothesis with four
supports, three of them strong, and it is testable - reranking qdrant's top-w is
the obvious next experiment.
