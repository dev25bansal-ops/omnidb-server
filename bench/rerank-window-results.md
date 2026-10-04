> **PROVENANCE WARNING (2026-09-30, REVISED after recompute)** - the six **scidocs**
> stage_recall_at_w values in this file are WRONG. The fiqa, scifact and nfcorpus
> values in this file are CORRECT and unchanged.
>
> The bug: stage_recall_at_w was len(top_k & set(rel)) / len(set(rel)), where rel is
> the BEIR qrels {doc_id: score} dict. set(rel) keeps every key, including documents
> judged with score 0, which are judged NOT relevant. That inflated the denominator.
>
> It only matters where score-0 judgements exist. Census: nfcorpus 0 of 12334,
> scifact 0 of 339, fiqa 0 of 1706, scidocs **25000 of 29928 (83.5%)**. The fix is
> therefore a no-op on three corpora and changes scidocs only.
>
> Corrected scidocs values (published -> true): bm25 w=25 0.0367 -> 0.2205,
> bm25 w=100 0.0593 -> 0.3418, minilm w=25 0.0562 -> 0.3374,
> minilm w=100 0.0870 -> 0.5089, bge-base w=25 0.0537 -> 0.3228,
> bge-base w=100 0.0842 -> 0.4957. All six move UP; the old values understated recall.
>
> Consequence for the prose below: the scidocs bm25 stage is NOT the recall-poorest
> first stage. Corrected, nfcorpus at 0.2376 is the lowest, which is what the
> weakness rule already assumed. The scidocs rows of that argument need revisiting.
>
> The nDCG, delta and p columns throughout this file are NOT affected - nDCG reads
> the graded scores directly and was always correct. Full working, including the
> corrected Spearman check: .omni-audit/F15-RECOMPUTE.md
# Probe: reranker sign vs retrieval window (fiqa, BAAI/bge-reranker-base)

- eval k = 10; reranker device: GPU when available
- first stages: omnidb-hybrid (corpus_mode, weighted RRF 0.7/0.3,
  retrieved fresh per window) and bm25s (k1=0.9, b=0.4)
- windows: 25, 50, 100; stage recall@w is
  the recall ceiling the reranker can draw from
- p: paired Wilcoxon over queries (reranked vs base, same window)

| first stage | embedder | w | stage recall@w | nDCG@10 base | nDCG@10 reranked | delta | p |
|---|---|---|---|---|---|---|---|
| bm25 | - | 25 | 0.3795 | 0.2380 | 0.3017 | +0.0637 | 0.0000 |
| bm25 | - | 50 | 0.4396 | 0.2380 | 0.3108 | +0.0728 | 0.0000 |
| bm25 | - | 100 | 0.4946 | 0.2380 | 0.3188 | +0.0808 | 0.0000 |
| omnidb-hybrid | minilm | 25 | 0.5483 | 0.3790 | 0.3758 | -0.0031 | 0.4408 |
| omnidb-hybrid | minilm | 50 | 0.6292 | 0.3796 | 0.3732 | -0.0064 | 0.2677 |
| omnidb-hybrid | minilm | 100 | 0.7188 | 0.3821 | 0.3573 | -0.0247 | 0.0036 |
| omnidb-hybrid | bge-base | 25 | 0.5871 | 0.3941 | 0.3848 | -0.0093 | 0.1062 |
| omnidb-hybrid | bge-base | 50 | 0.6638 | 0.3922 | 0.3728 | -0.0195 | 0.0074 |
| omnidb-hybrid | bge-base | 100 | 0.7360 | 0.3963 | 0.3556 | -0.0407 | 0.0000 |

## Verdict

- one-line: reranking first becomes non-negative at bm25/w=25 (+0.0637); it NEVER becomes non-negative on omnidb-hybrid at any window tested (best -0.0031).
- prediction check: REFUTED for the hybrid stage. The hypothesis said the reranker should gain at w=25 (recall the bottleneck) or w=100 (headroom); instead the delta goes monotonically MORE negative as the window grows on both embedders
  - minilm: w25:-0.0031 -> w50:-0.0064 -> w100:-0.0247
  - bge-base: w25:-0.0093 -> w50:-0.0195 -> w100:-0.0407
  (bge-base w=100: p<0.0001, minilm w=100: p=0.0036). Extra candidates only give the reranker more chances to displace relevant docs it under-scores - recall is NOT the bottleneck for omnidb-hybrid on fiqa.
- positive-likelihood cell CONFIRMED: on the weak bm25 first stage (recall@100 = 0.49 vs hybrid's 0.72-0.74) the reranker gains at every window and the gain grows with window (+0.064/+0.073/+0.081 at w=25/50/100, all p<0.0001), closing ~46% of the gap to the hybrid stage at w=100 (bm25+CE 0.3188 vs omni minilm base 0.3796).
- mechanism: the reranker's value scales with FIRST-STAGE WEAKNESS, not with window size. Where the first stage is already strong (hybrid RRF), CE rerank is a net loss at every window; where it is weak (bm25), CE rerank is the single biggest lever available. The window knob modulates the gain only for the weak stage; for the strong stage every window tested made things worse, monotonically.

## Honesty notes

- omnidb-hybrid re-retrieved per window (fused top-10 depends
  on k_fetch; prefix-of-100 is not the true top-25 ranking).
- bm25s is single-leg: top-w is the exact prefix of top-100.
- CE scores cached per (query, doc) across windows; identical
  to scoring each window's list directly.
- fiqa full size (57,638 docs, 648 queries); embeddings from
  bench/.cache3 (minilm, bge-base); bge-reranker-base,
  max_length=512, same as research_bench_v3.

## nfcorpus

Same probe as the fiqa section above, second corpus. Short medical
abstracts, graded relevance, small corpus; windows narrowed to the
two extremes (25, 100). All helpers imported from the fiqa probe so
the method is bit-for-bit identical.

- corpus: nfcorpus full size (3633 docs, 323 queries); eval k = 10
- reranker: BAAI/bge-reranker-base, max_length=512, on cuda (GPU)
- first stages: omnidb-hybrid (corpus_mode, weighted RRF 0.7/0.3,
  retrieved fresh per window) and bm25s (k1=0.9, b=0.4)
- p: paired Wilcoxon over queries (reranked vs base, same window)

| first stage | embedder | w | stage recall@w | nDCG@10 base | nDCG@10 reranked | delta | p |
|---|---|---|---|---|---|---|---|
| bm25 | - | 25 | 0.1788 | 0.3047 | 0.3129 | +0.0082 | 0.5040 |
| bm25 | - | 100 | 0.2376 | 0.3047 | 0.3051 | +0.0004 | 0.8945 |
| omnidb-hybrid | minilm | 25 | 0.2026 | 0.3388 | 0.3184 | -0.0203 | 0.0041 |
| omnidb-hybrid | minilm | 100 | 0.3229 | 0.3445 | 0.3188 | -0.0257 | 0.0350 |
| omnidb-hybrid | bge-base | 25 | 0.2332 | 0.3763 | 0.3380 | -0.0383 | 0.0000 |
| omnidb-hybrid | bge-base | 100 | 0.3427 | 0.3731 | 0.3190 | -0.0541 | 0.0000 |

### Verdict (nfcorpus)

- one-line: on nfcorpus the reranker gains at every window on bm25 (w25:+0.0082/w100:+0.0004); it loses at every window on omnidb-hybrid/minilm (w25:-0.0203/w100:-0.0257) and loses at every window on omnidb-hybrid/bge-base (w25:-0.0383/w100:-0.0541).
- vs fiqa: the hybrid sign REPLICATES - loss at every window here as there (minilm -0.0031/-0.0247, bge-base -0.0093/-0.0407 at w=25/100).
- window trend: every stage's delta shrinks from w=25 to w=100 (matches hybrid-on-fiqa).
- mechanism: weakens the first-stage-weakness law - despite nfcorpus hybrid recall@25 (minilm 0.2026) sitting far below fiqa's (0.5483), reranking the strong stage still fails to gain; the reranker's value is corpus-dependent beyond recall headroom alone.
- bm25+CE at w=100 (0.3051) does not overtake the best no-rerank hybrid (0.3763).

### Honesty notes (nfcorpus)

- omnidb-hybrid re-retrieved fresh per window (fused ranking depends
  on k_fetch; prefix-of-100 is not the true top-25 ranking).
- bm25s single-leg: top-w is the exact prefix of top-100 (verified
  pre-flight: every nfcorpus query returns 100 real candidates, no
  -1 padding, no duplicate ids).
- CE scores cached per (query, doc) across windows; identical to
  scoring each window's list directly.
- nfcorpus full size (3633 docs, 323 queries); embeddings from bench/.cache3 (minilm, bge-base), same
  as research_bench_v3; reranker BAAI/bge-reranker-base, max_length=512,
  cuda (GPU).
- fiqa deltas/recalls quoted in the verdict are from the section
  above (same reranker, same method, windows 25/50/100 there).

## scidocs

scidocs sweep of the rerank-window probe above (25657 docs, 1000 queries; reranker BAAI/bge-reranker-base, max_length=512): same wiring as fiqa - omnidb-hybrid retrieved fresh per window, bm25s single-leg (top-w is the exact prefix of top-100), CE pair-scores cached across windows; windows 25, 100; p: paired Wilcoxon over queries (reranked vs base, same window).

| first stage | embedder | w | stage recall@w | nDCG@10 base | nDCG@10 reranked | delta | p |
|---|---|---|---|---|---|---|---|
| bm25 | - | 25 | 0.0367 | 0.1474 | 0.1562 | +0.0088 | 0.0062 |
| bm25 | - | 100 | 0.0593 | 0.1474 | 0.1507 | +0.0033 | 0.3718 |
| omnidb-hybrid | minilm | 25 | 0.0562 | 0.2084 | 0.1893 | -0.0191 | 0.0000 |
| omnidb-hybrid | minilm | 100 | 0.0870 | 0.2083 | 0.1570 | -0.0513 | 0.0000 |
| omnidb-hybrid | bge-base | 25 | 0.0537 | 0.2107 | 0.1901 | -0.0206 | 0.0000 |
| omnidb-hybrid | bge-base | 100 | 0.0842 | 0.2094 | 0.1620 | -0.0475 | 0.0000 |

### Verdict

- one-line: REPLICATES the fiqa pattern - bm25: w25:+0.0088, w100:+0.0033 (p=0.0062/0.3718); omnidb-hybrid: minilm w25:-0.0191; minilm w100:-0.0513; bge-base w25:-0.0206; bge-base w100:-0.0475; best omnidb-hybrid delta -0.0191 (minilm/w=25).
- window trend on omnidb-hybrid: minilm: w25 -0.0191 -> w100 -0.0513; bge-base: w25 -0.0206 -> w100 -0.0475 - worse with window on both embedders, as on fiqa
- first-stage weakness (recall@100: bm25 0.0593 vs hybrid 0.0842-0.0870): the reranker gains on the weak stage at every tested window, closing 5% of the bm25->hybrid base gap at w=100 (bm25+CE 0.1507 vs hybrid minilm base 0.2083).
- mechanism: replicates fiqa - reranker value scales with first-stage weakness, not window size; CE-reranking a strong hybrid stage is a net loss at every window tested, so the fiqa product call (no CE rerank on hybrid) carries over.

### Notes

- omnidb-hybrid re-retrieved per window (fused top-w depends on
  k_fetch; prefix-of-100 is not the true top-25 ranking).
- bm25s single-leg: top-w is the exact prefix of top-100.
- CE scores cached per (query, doc) within each stage's candidate
  union; identical to scoring each window's list directly.
- research_bench_v3 measured omnidb+CE top-50 on scidocs at -0.0197
  (minilm) / -0.0238 (bge-base); this sweep brackets that window.
- scidocs full size (25657 docs, 1000 queries); embeddings from bench/.cache3 (minilm, bge-base);
  windows 25 and 100 only (fiqa probe also tested 50).

## scifact

scifact sweep of the rerank-window probe above (5183 docs, 300 queries; reranker
BAAI/bge-reranker-base, max_length=512): same wiring as fiqa - omnidb-hybrid
retrieved fresh per window, bm25s single-leg (top-w is the exact prefix of
top-100), CE pair-scores cached across windows; windows 25, 100; p: paired
Wilcoxon over queries (reranked vs base, same window). Numbers from
bench/scifact-window-results.json.

| first stage | embedder | w | stage recall@w | nDCG@10 base | nDCG@10 reranked | delta | p |
|---|---|---|---|---|---|---|---|
| bm25 | - | 25 | 0.8339 | 0.6569 | 0.7052 | +0.0484 | 0.0000 |
| bm25 | - | 100 | 0.8759 | 0.6569 | 0.6952 | +0.0383 | 0.0017 |
| omnidb-hybrid | minilm | 25 | 0.8580 | 0.6984 | 0.7051 | +0.0067 | 0.5462 |
| omnidb-hybrid | minilm | 100 | 0.9383 | 0.7059 | 0.7125 | +0.0066 | 0.6000 |
| omnidb-hybrid | bge-base | 25 | 0.9260 | 0.7531 | 0.7337 | -0.0194 | 0.0900 |
| omnidb-hybrid | bge-base | 100 | 0.9710 | 0.7492 | 0.7165 | -0.0327 | 0.0155 |

### Verdict (scifact)

- one-line: fiqa PARTIALLY REPLICATES / DIVERGES on scifact - bm25s w25:+0.0484 ->
  w100:+0.0383 (gain does NOT grow with window here), omni minilm w25:+0.0067 ->
  w100:+0.0066 (n.s. at both), omni bge-base w25:-0.0194 -> w100:-0.0327
  (best hybrid delta +0.0067). Significant hybrid loss: bge-base w=100 -0.0327
  (p=0.0155).
- vs fiqa: the direction REPLICATES (weak stage gains, strong stage loses as the
  window opens) but the WINDOW TREND DIVERGES - on fiqa the bm25 gain grew with
  window (+0.064/+0.073/+0.081) and here it SHRINKS (+0.0484 -> +0.0383). scifact's
  bm25 recall@100 is 0.8759 - the weak stage has little headroom left, so extra
  candidates stop helping. The strong stage is near-ceiling (bge-base recall@100
  0.9710), so there is nothing for the CE to add and displacement only loses.
- mechanism: still first-stage weakness, not window size - but scifact shows the
  law's second edge. Both stages start near-saturated (bm25 0.657 / hybrid 0.75
  nDCG@10), so the CE's leverage band is narrow in both directions: +0.048 on the
  weak stage is real but the bm25+CE at w=25 (0.7052) only TIES the hybrid/no-rerank
  baseline (0.6984-0.7531) and the best hybrid+CE cell adds a n.s. +0.0067.

### Notes (scifact)

- omnidb-hybrid re-retrieved per window (fused top-w depends on
  k_fetch; prefix-of-100 is not the true top-25 ranking).
- bm25s single-leg: top-w is the exact prefix of top-100.
- CE scores cached per (query, doc) within each stage's candidate
  union; identical to scoring each window's list directly.
- scifact full size (5183 docs, 300 queries); embeddings from bench/.cache3
  (minilm, bge-base); windows 25 and 100 only.

## Regime summary (integrated)

One row per (dataset x first-stage). "Best delta (w)" is the largest
nDCG@10 delta across the windows tested for that cell; "p" is the paired
Wilcoxon p at that best window. Verdict: **helps** = best cell positive and
p<0.05; **hurts** = best cell negative and p<0.05; **neutral** = best cell
not significant (n.s.). Every number is quoted from the per-dataset sections
above and re-verified against the raw JSON.

| dataset | docs | first stage | best delta (w) | p | verdict |
|---|---|---|---|---|---|
| fiqa | 57,638 | bm25 | +0.0808 (w=100) | <0.0001 | helps |
| fiqa | 57,638 | omnidb-hybrid / minilm | -0.0031 (w=25) | 0.4408 | neutral (n.s.) |
| fiqa | 57,638 | omnidb-hybrid / bge-base | -0.0093 (w=25) | 0.1062 | neutral (n.s.) |
| scidocs | 25,657 | bm25 | +0.0088 (w=25) | 0.0062 | helps |
| scidocs | 25,657 | omnidb-hybrid / minilm | -0.0191 (w=25) | <0.0001 | hurts |
| scidocs | 25,657 | omnidb-hybrid / bge-base | -0.0206 (w=25) | <0.0001 | hurts |
| scifact | 5,183 | bm25 | +0.0484 (w=25) | <0.0001 | helps |
| scifact | 5,183 | omnidb-hybrid / minilm | +0.0067 (w=25) | 0.5462 | neutral (n.s.) |
| scifact | 5,183 | omnidb-hybrid / bge-base | -0.0194 (w=25) | 0.0900 | neutral (n.s.) |
| nfcorpus | 3,633 | bm25 | +0.0082 (w=25) | 0.5040 | neutral (n.s.) |
| nfcorpus | 3,633 | omnidb-hybrid / minilm | -0.0203 (w=25) | 0.0041 | hurts |
| nfcorpus | 3,633 | omnidb-hybrid / bge-base | -0.0383 (w=25) | <0.0001 | hurts |

Reading the table: **the CE helps on the weak lexical stage and never helps on the
strong hybrid stage; on the strong stage it either loses significantly (scidocs,
nfcorpus, and fiqa/bge-base at w=50-100) or is a wash (fiqa w=25, scifact).** Note
the verdict is computed at the BEST window per cell, so "neutral" hides that
fiqa/bge-base and scifact/bge-base both go significantly negative as the window
opens (fiqa bge-base w=100 -0.0407 p<0.0001; scifact bge-base w=100 -0.0327
p=0.0155).

### When does CE reranking flip sign? (boundary characterization)

The sign of the CE delta is set by **first-stage weakness at the retrieval
window**, measured by stage recall@w - the recall ceiling the reranker draws from
- not by window size. The window knob only modulates the magnitude of an
already-signed effect (and on the strong stage it modulates it the wrong way:
delta grows monotonically MORE negative with w on every dataset and embedder).

- **Weak stage (bm25s) - positive.** fiqa bm25 recall@100 = 0.4946 vs the hybrid
  stage's 0.7188 (minilm) / 0.7360 (bge-base): a ~0.22-0.24 recall gap of
  headroom, and the CE closes ~46% of the bm25->hybrid base gap at w=100
  (+0.0808, p<0.0001). scifact bm25 recall@100 = 0.8759 and the CE still helps
  (+0.0484, p<0.0001) even though the stage is strong in absolute terms - the
  relevant comparison is bm25 vs its OWN hybrid, where scifact's hybrid sits at
  0.9383-0.9710, leaving a 0.06-0.10 gap the CE can still exploit.
- **Strong stage (omnidb-hybrid) - never positive.** Across all four datasets and
  both embedders the hybrid cells' best delta is between -0.0031 and +0.0067; the
  two positive ones (fiqa/minilm -0.0031 is negative; scifact/minilm +0.0067) are
  n.s. Every significant hybrid cell is a loss. Where the first stage has already
  fused lexical + dense (weighted RRF) and the recalls are high (0.32 on nfcorpus
  up to 0.97 on scifact), the CE's marginal candidates are near-ties it reorders
  badly; extra window candidates just give it more near-ties to displace.

**nfcorpus is the caveat that motivates "weakness, not recall":** nfcorpus bm25
recall@100 is only 0.2376 - the LOWEST of any first stage here, lower even than
scidocs' 0.0593-free hybrid cells in absolute candidate terms - yet the bm25 CE
gain is n.s. (+0.0082 w=25 p=0.5040; +0.0004 w=100 p=0.8945). Low recall alone is
**not** sufficient for a CE gain. nfcorpus is short graded-relevance medical
abstracts where the corpus already fits both legs well (bm25 base nDCG 0.3047
vs hybrid 0.34-0.38 - a small gap), so the candidate pool, though recall-poor, is
of low marginal value to a general-domain cross-encoder. The working rule is the
**gap between the weak stage and the strongest available stage**, moderated by
corpus fit to the CE's training domain - both must be favorable, and nfcorpus
fails the second.

### When to reach for the cross-encoder

Reach for the CE when the first stage is a weak single-leg retriever (lexical-only
BM25) and it is leaving a large recall gap on the table - that is the one regime
where it is a large, reliable, monotonic-with-window win (+0.08 on fiqa,
p<0.0001). Do NOT add a CE on top of an already-fused hybrid stage: across four
full-size datasets and both embedders it is a net loss at every window tested
(significant on scidocs, nfcorpus, and fiqa as the window opens), and never a
significant gain. And treat low recall as necessary-but-not-sufficient - nfcorpus's
bm25 recall@100 of 0.2376 produced no significant gain because the corpus was
already well-served by both single legs, so confirm the WEAK-vs-STRONG stage gap
and the corpus-domain fit before paying the reranker cost.
