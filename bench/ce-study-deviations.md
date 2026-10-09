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

Justification measured, not assumed: across 40 real scidocs / hybrid / bge-base / w100 candidate
pools, batch 4 and batch 32 produced identical CE rankings and identical fused rankings. Only
wall-clock changed. This is a speed change, not a measurement change.

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

## 3. Protocol document not amended in place

By design, per the opening paragraph. The candidate-model notes in the frozen protocol name two
models considered but not selected (`MiniLM-L12-v2`, DistillRankNET) alongside the three that
were. The selection record is `ce-study-manifest.json` plus the run config's `rerankers` map; the
protocol text is historical and is not edited to match later decisions.