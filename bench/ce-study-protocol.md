# CE second-stage study: locked analysis protocol

Protocol v1, frozen before held-out evaluation on 2026-10-08. This study tests whether CE signal is weaker than a strong hybrid, useful only as an interpolation signal, or harmful under specific first stages/domains. No broad “CE adds nothing” claim will be made from a standalone reranking experiment alone.

## Data and split

- Collections: full BEIR `nfcorpus`, `scifact`, `scidocs`, `fiqa`, using the locally cached official test corpus/query/qrels files. Record SHA-256 for each file in the run manifest.
- Per corpus, sort query IDs by SHA-256 of `omnidb-ce-study-v1|<corpus>|<query_id>`; first `floor(0.40*n)` IDs are development, remaining IDs are test. For `n >= 2`, clamp the development count to `[1, n-1]`. This uses neither qrels values nor model outputs. Save exact IDs and hashes before any reranker test analysis.
- All candidate-pool rankings may be retrieved for all queries, but reranker/alpha/model choices may use only development-query outcomes. The test partition remains sealed until parameters and analysis code are frozen.

## Models and matrix

Five pinned rerankers: existing `cross-encoder/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`, plus `BAAI/bge-reranker-large`, `mixedbread-ai/mxbai-rerank-base-v2`, and `ncbi/MedCPT-Cross-Encoder`. Exact SHAs are in `ce-study-manifest.json`. The large BGE and mxbai are stronger-candidate general models based on size/reported reranking results, not assumed to outperform on these pools. MedCPT is a biomedical/PubMed-trained domain probe; it is not presumed in-domain for SciDocs and is not described as a general-purpose model.

- First stages: OmniDB hybrid, exact dense-only, OmniDB FTS5 sparse-only, bm25s, and Pyserini. Hybrid and dense-only have four encoder configurations; sparse-only, bm25s, and Pyserini are embedder-independent. Windows: 25, 50, 100; evaluation cutoff K=10.
- Qdrant `:memory:` is exact local brute-force in the installed `qdrant-client`; it is retained only as an independent exact-dense validation, not treated as an independent ANN first stage or duplicated in the confirmatory family. No ANN claim is made unless a separate HNSW run is actually conducted.
- Fixed-pool candidates, raw per-pair CE scores, query IDs, qrels, tokenizer max length, truncation, revisions, and ranking outputs are saved. Every stage is retrieved fresh at each window width; no stage's w-window list is produced by slicing a wider retrieval. `bench/verify_prefix_slicing.py` tested that shortcut on nfcorpus and it is NOT sound for the lexical stages: `recall()` passes the fetch width into the FTS `LIMIT`, so `bm25()` scores only the admitted rows and top-100[:25] differs from a direct top-25 on 2 of 3 windows (bm25s likewise; dense and Pyserini are prefix-stable, but all stages use the same fresh-retrieval path so the rule has no exceptions to remember).

## Selection and estimands

- Development tuning: for each corpus x first-stage x reranker, select alpha from `{0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1}` and RRF k from `{20, 60, 100}` using development mean nDCG@10, pooling all three windows only after query-level evaluation; tie-break toward alpha=0, then smaller k. Freeze one alpha/k per corpus x first-stage x reranker before test evaluation. Report window-specific test results at these frozen settings. Standalone CE ordering and pure CE rank are secondary descriptive outcomes.
- Primary estimand: arithmetic mean of paired held-out per-query nDCG@10 differences, frozen development-selected hybrid/first-stage + CE rank blend minus the same base first-stage ranking. Secondary outcomes: CE-only ordering, standalone CE score-level, unblended CE reranking, pool-oracle nDCG@10, judged@10/unjudged@10, and condensed nDCG@10 after removing unjudged documents. Standard BEIR nDCG remains primary; condensed nDCG is robustness-only.
- Report `CE standalone nDCG - base nDCG` versus reranking delta across cells as descriptive evidence, with uncertainty clustered by query within corpus; cells sharing queries are not treated as independent observations. No n=4 t-approximation correlation is confirmatory.
- Practical reporting threshold: +0.01 nDCG@10. It is not an equivalence margin. Do not say gains above a bound are ruled out unless the held-out interval/equivalence procedure justifies it.

## Inference and multiplicity

- Primary test: two-sided paired sign-flip permutation test for the held-out mean difference, retaining exact-zero differences as zero contributions; PCG64 seed 20261008, 100,000 Monte Carlo sign patterns, p=`(b+1)/(B+1)`. Paired percentile bootstrap CI: 20,000 resamples, seed 20261009. Report Wilcoxon signed-rank as sensitivity only and state that it targets a different distributional/median-like null; exact-zero handling is explicit.
- Confirmatory family A is all held-out frozen-blend-minus-base contrasts: five rerankers x 11 first-stage variants (8 embedder-specific hybrid/dense variants + 3 embedder-independent lexical variants) x 3 windows x 4 corpora = 660 contrasts. Each contrast is one corpus/stage/reranker/window with its embedder-specific query population; BM25/FTS5 shared stages occur once per corpus, not once per encoder. Deduplicate pooled stages; do not count the same sparse/BM25 result once per embedding model. Holm is applied across all 660. If any planned cell is unavailable, retain its slot as missing and report why; do not shrink the family after seeing outcomes.
- Family B comprises prespecified robustness contrasts (dense-only CE, FTS5 sparse-only CE, judged/condensed robustness) and is corrected jointly within that family. Other model/window/alpha sweeps are exploratory, with estimates and CIs emphasized.
- For the existing sparse-control table, show its current per-corpus m=2 Holm result and an m=8 cross-corpus sensitivity; under m=8, nfcorpus hybrid-minus-sparse and SciDocs dense-only-minus-hybrid are approximately 0.0644 and do not survive.

## Candidate model notes

- MiniLM-L6: general English MS MARCO baseline.
- BGE-reranker-base: XLM-R-based baseline.
- MiniLM-L12: 33.4M parameters, max positions 512, Apache-2.0; MS MARCO capacity comparison, not assumed stronger.
- DistillRankNET: 33.4M, max positions 512, Apache-2.0; MS MARCO Passage/SPLADE-oriented, not called domain-specialized.
- MedCPT: max positions 512, public-domain; biomedical/PubMed search-log probe. SciFact is the closest in-domain corpus; SciDocs is not presumed biomedical. Interpret other-corpus results as out-of-domain.

## Stopping and provenance

No substitutions, parameter changes, or reranker/window additions after viewing held-out test outcomes. Download only the exact pinned revisions listed in `ce-study-manifest.json`; verify revision and file checksums before inference. If an exact pinned model cannot be fetched or loaded, record the failure and leave that condition unavailable; do not silently change revision. Use new versioned run paths, atomic cell checkpoints, corpus/model revisions, code SHA, software/device metadata, and checksums; never overwrite `research-v3` historical results. Freeze scripts and manifest hashes before test scoring; publish all per-query results needed to regenerate tables.
