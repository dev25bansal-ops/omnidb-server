# CE study: recorded deviations from the frozen protocol

Append-only log of departures from [ce-study-protocol.md](ce-study-protocol.md) in run
`ce-study-v1-20261009`. The protocol file itself is hash-locked: its SHA-256 is written into
`run-config.json` as an immutable field and every resume re-verifies it, so editing the protocol
to describe what actually happened would break the run it is supposed to govern. Deviations are
recorded here instead, and the run config records the mechanical ones inline.

Nothing here changes the estimand, the query split, or the 660-contrast confirmatory family.
Entries are limited to execution mechanics found to be unsound or wasteful during the run.

## 1. Batch size 4 -> 32

Recorded via `--record-batch-size-change`, which writes a `batch_size_history` entry into
`run-config.json` with the reason and timestamp. `batch_size` is one of the few mutable config
fields precisely so this change is auditable rather than silent.

**The original justification was too strong and is corrected here.** It rested on 40 real
scidocs / hybrid / bge-base / w100 pools, where batch 4 and batch 32 produced identical CE
rankings. Re-tested on 396 dev rows stratified across all 132 corpus/stage/window cells, the
change is *almost* rank-neutral but not exactly so:

| quantity | batch 4 vs 32 |
| --- | --- |
| rows where the CE ordering changed | 3 / 396 (0.8%) |
| rows where nDCG@10 changed | **0 / 396** |
| fused rankings differing on the alpha x k grid | 90 / 13,068 (0.69%) |

**Mechanism, measured.** A pair's score is deterministic at a fixed batch size - re-scoring the
same pool returns bit-identical values, and reversing or shuffling document order changes nothing.
What moves is the batch size itself: on this GPU, batch 4 vs batch 32 shifts individual scores by
up to 4.77e-07 (a different reduction order in the attention/matmul kernels), and that is enough
to reorder two nearly-tied documents. So the runs before and after the switch are not
bit-comparable, and a handful of near-ties resolve differently.

**Why this does not invalidate the run.** The shift is ~1e-07, i.e. six orders of magnitude below
the +0.01 nDCG@10 reporting threshold, and it changed nDCG@10 on 0 of 396 sampled rows. It is
recorded as a deviation because it is a real, if tiny, discontinuity in the artifact's history:
cells scored before the switch and cells scored after it are not bit-identical measurements. The
affected cells are exactly the ones the run config's `batch_size_history` timestamps delimit, and
the direction of the effect is not systematic (it is float noise, not a bias).

The alternative - re-scoring all 62,728 pre-switch rows at batch 32 to make the artifact
homogeneous - would cost roughly 20 hours of GPU time to remove a 1e-07 discrepancy that moves
no reported number. That trade is recorded here rather than taken silently.

## 1b. Redundant pair-scoring (found, deliberately not exploited)

A cross-encoder score depends only on the two input texts, so the study re-scores the same
(query, document) pair many times: a document in the window-100 hybrid pool also appears in the
window-50 pool and in the dense and lexical pools. Measured over the MiniLM dev rows, 1,741,541
pair-scorings cover only 296,795 unique pairs - **83% of the CE compute is redundant**.

Caching those scores would cut the remaining run by most of its wall-clock. It is **not** being
used, because `bench/verify_score_cache.py` shows it is not exactly rank-neutral: replaying every
scored row through a first-occurrence cache reproduces the CE ordering on 29,747 / 29,964 MiniLM
rows and 30,054 / 30,264 bge-base rows but only 3,276 / 3,300 bge-large rows, and it moved nDCG@10
on one bge-large row (nfcorpus/hybrid/w50/PLAIN-2550, 0.250807 -> 0.253722, a 0.0029 shift). The
cause is the same batch-size effect as above: a cache returns the score from whichever pool first
scored the pair, so it silently mixes batch compositions across rows.

The finding is kept because it is worth recording that the study's cost is ~6x its information
content, and because a future run could exploit it safely by *decomposing* the work rather than
caching it: score the unique pairs once per (reranker, query) in a single fixed batch, then join.
That is a protocol change, not an optimization, and cannot be applied to a run already in flight
without invalidating comparability with the rows already scored. It is therefore left for a v2
protocol.

## 2. Duplicate records in `dev-scores.jsonl` (300 keys, `BAAI/bge-reranker-base`)

**What happened.** The batch-32 restart was launched while the batch-4 worker was still finishing
its current corpus. Both processes appended to the same `dev-scores.jsonl`, so 300 keys appear
twice. All 300 are `fiqa` hybrid - the corpus in flight at the switch - split across
`bge-small` (223) and `minilm` (77), evenly over the three windows.

**Effect on reported metrics: none.** The two copies of each key agree on every stored metric:
`ce_ndcg10`, `base_ndcg10`, `oracle_ndcg10`, judged/unjudged at 10, both condensed variants, and
the full `pool_ids` list and `candidate_count`. Only the raw `ce_scores` differ, by at most
3.0e-06 (median 0.0) - the float-noise signature of the same pairs scored at a different batch
size. Recomputing fused rankings from both copies across the entire alpha x k grid gives 9,900
comparisons; 13 diverge, every one of them first at rank >= 60, and **0 of 9,900 change
nDCG@10**. Reported numbers are therefore untouched.

**Effect on dev selection: real, and that is why this is recorded.** `freeze_development_parameters`
averages per-query nDCG over rows, so an uncollapsed duplicate double-weights one query in one
cell. Under the raw file, exactly one of 93 dev cells selects differently:
`fiqa|bge-small|hybrid|BAAI/bge-reranker-base` chose alpha 0.5 / k 20 with the duplicate and
alpha 0.3 / k 60 without it. The deduplicated value is the correct one, since the true design has
one row per measured key.

**Fix.** `read_scores()` in `ce_study_runner.py` collapses repeated keys to their first record and
is now used at every read site: `freeze_development_parameters`, `apply_frozen_test_blends`, and
`ce_study_analyze.summarize`. Keeping the first copy vs the last is not load-bearing - both
select the same alpha/k in all 93 dev cells - so the tie-break rule cannot silently change a
result. The raw file is left as written; the duplicates are a faithful record of what the two
processes did, and the collapse happens at read time.

**Root cause fixed, not just the symptom.** `acquire_run_lock()` in `ce_study_runner.py` makes a
second concurrent writer on one run directory impossible rather than merely detectable: `main()`
takes an exclusive lock naming the live PID for the duration of every phase, refuses to start if
another live process holds it, and reclaims the lock when its owner is gone (killed process,
reboot) or the file is unreadable. A process re-acquiring its own lock is allowed, since it cannot
race itself.

**Regression tests.** `tests/test_ce_study_runner_contracts.py` covers the collapse (duplicates
drop, distinct rerankers are kept, absent file is a no-op), the lock (refuses a real second
process, reclaims a stale or corrupt lock, release is idempotent), and each test was falsified
against the injected bug before being kept.

## 3. Dev phase restart (execution, not measurement)

The dev worker was stopped early during the investigation above and resumed from the same
checkpoint. The stop was clean: `dev-scores.jsonl` ends on a newline, all 62,728 records parse,
and the worker did not need the torn-write recovery path. The resume re-derived its completed-key
set and skipped the two finished rerankers with `rows=0` before continuing `bge-reranker-large`,
so no cell was re-scored and no record was written twice by this restart. The run's own lock
file names the resuming PID, so a second writer cannot attach to the same phase.

This changes nothing about the estimand or the split; it is recorded only so the artifact's
timeline is reconstructible.

## 4. Protocol document not amended in place

By design, per the opening paragraph. The candidate-model notes in the frozen protocol name two
models considered but not selected (`MiniLM-L12-v2`, DistillRankNET) alongside the three that
were. The selection record is `ce-study-manifest.json` plus the run config's `rerankers` map; the
protocol text is historical and is not edited to match later decisions.

## 5. Host reboot mid-run (execution, not measurement)

The host rebooted at 2026-10-10 23:54 (Windows auto-maintenance; the session's `nohup` worker
died with it). The lock file named the now-dead PID 4752, so on next launch the run correctly
refused to start against it until the stale owner was verified gone. The dev partition file was
unharmed: `dev-scores.jsonl` ends on a newline, all 98,509 records parse, and the last row is a
complete `mxbai-rerank-base-v2/scidocs` record.

On resume, the runner's immutable-config check refused the start for the correct reason: the code
SHA had moved from `6536e130` (when the run began) to `016b1e9c` because of the two hardening
commits (`f2d49a5` dedup + lock, `016b1e9` score-cache verification). The diff between the two
SHAs touches only `read_scores` and the lock helpers - `score_partition`, `score_pairs`,
`load_reranker`, and every `model.predict` call are byte-identical - so the 98,509 already-scored
rows remain valid and the change was recorded with `--record-code-sha-change` naming that. The
resume then skipped the three finished rerankers and the two finished mxbai corpora with `rows=0`
and continued from `mxbai-rerank-base-v2/scidocs` at the expected rate.

This changes nothing about the estimand or the split; it is recorded only so the artifact's
timeline is reconstructible.