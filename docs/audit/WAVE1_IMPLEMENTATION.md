# Implementation Wave 1 — Folder Teams

**Status: shipped and verified.** Suite green, lint clean, mypy clean, mutation gate passing.

| Metric | Before | After |
|---|---|---|
| Tests | 292 | **369** (+77) |
| Failures | 2 (F1's AST test, on arrival) | **0** |
| Mutation score | 33% (unenforced) | **100%** (29/29, gated in CI) |
| `ef_search` | 64 | **128** |
| Lint (`omnidb_server/` + `tests/`) | clean | clean |
| mypy | clean | clean |

## What each folder team produced

| Team | Folder | Delivered |
|---|---|---|
| **F1** | core security | `test_config_llm_url.py` (llm_base_url https + loopback exemption), `test_skills_zip_bomb.py` (declared-size gate, no real bomb), `test_sql_injection_surface.py` (**25 KB**, AST classifier + 3 negative controls) |
| **F2** | retrieval + gates | `ef_search` 64→128 with a comment that no longer contradicts itself; `bench/baseline.py` gate rewritten; `bench/lexical_scaling.py` + results for the newly-measured FTS leg |
| **F3** | tests + mutation | `test_fusion_values.py`, `test_scoring_values.py` (exact pinned floats, bit-exact), `test_fusion_and_admission_properties.py` (Hypothesis; reference tokenizer written as a char scan so it cannot inherit the bug under test), `scripts/mutation_check.py` rewritten to be able to fail |
| **F5** | docs | Retraction pass over `AUDIT.md` / `DATASTORE_SPIKE.md` / `perf-report.md` / `coverage-report.md` — **12 distinct fabrications removed** |

## Three defects I found in the agents' own work

The teams were competent but not infallible. Each of these was mine to catch:

1. **F1's negative control asserted the wrong thing.** It claimed `",".join(chr(63) * len(ids))` was a *safe* shape. The classifier rejects it — correctly, since `chr()` is a runtime call and the design fails closed. The real code uses `",".join("?" * len(ids))` (`store.py:267,613,624,904`). Fixed the control to the real shape, **and added a second control asserting the classifier still rejects the `chr()` form**, so the fail-closed property is itself pinned.

2. **The AST classifier had three false-positive classes** — 10 sites, all safe:
   - `_SELECTIVE` imported via `from .store import` in **5 modules**, but the check was `if alias.name in sibling_strs` where `sibling_strs` is keyed by *module stem*. Never matched.
   - imports inside **function bodies** (`portability.py:143`) — the scan only walked `tree.body`.
   - `db.py:139`'s `version`, unpacked from `MIGRATIONS` — `literal_eval` cannot resolve `_MIGRATION_1`, so the table was invisible to the checker.

   **In every case the classifier was right and the world was wrong.** I fixed the classifier, not the assertions. A guard that cries wolf on 10 legitimate sites trains people to add bypasses.

3. **`mutation_check.py` had 8 of 30 anchors silently `NOT_APPLICABLE`** — the RRF half of the table stopped matching when `rrf` grew a `weights` argument. A stale anchor is indistinguishable from an unkilled mutant, so the 33% score was computed over a table that was mostly not testing anything. Now: **fail on stale anchors**, exclude declared-equivalent mutants from the denominator, and **exit non-zero below `--min-score`** (proved by `--min-score 100.1` → exit 1).

## The mutation score went 33% → 100% honestly

F3's reviewer subagent did the thing that makes the number mean something: **it mutated the new tests themselves** and required each to go red, with comment-only controls required to stay green. 14/14 behaved correctly.

One real weakness surfaced that way: a bare `assert frequency == 0.0` accepts `-0.0` in IEEE-754. The test now checks the sign bit explicitly.

## F5's retractions are the uncomfortable part

Twelve fabricated or stale numbers removed. The load-bearing ones:

- `AUDIT.md` claimed release criterion 7 passed on a "final runtime stability run" at p95 ratio **3.758**. The artifact says **5.337** with `validation_gate_passed: False`. **Criterion 7 is not satisfied.**
- `perf-report.md`'s vector-sweep table was wrong on **every row** — 384-d/100k/ef128 read 0.383/0.632/0.911 where `bench/results.md:20` says **0.519/0.536/0.934**.
- `perf-report.md` and `AUDIT.md` both cited the *same digest* (`c1e17113…`) for two artifacts whose digests are `4ca9b6be…` and `52de74cb…`.
- `DATASTORE_SPIKE.md` carried **six** soak figures, none matching the file it cited — the p95 was *understated* by 4.6 ms, so the retraction made the result look worse than the artifact, not better.

**A structural finding, not just a list.** The two files carrying the false I12/I14 evidence are **gitignored and untracked** (`.gitignore:30-31`). A performance claim whose only support is an untracked file cannot be re-checked by a third party — and in this case the working-copy artifact was a *failing* run. That is how the fabrication happened, and it is a tooling gap, not a writing gap.

## Two things a reader should not conclude

- **"The suite is well tested now."** It is better tested. `portability.py` sits at 75.7% and `server.py` at 63.2%; `cli.py` is 0%-covered by design (integration-only).
- **"Criterion 7 can be signed off."** It cannot. The gate fails, and no passing run exists in the repository.

## Where the next wave should go

1. **`LICENSE` — still absent.** 15 minutes, blocks a package already on PyPI, and gates every other packaging item.
2. **The 2-week tranche code fixes** — ordering bugs in `reembed_home`/`rebuild_index`, `forget_many`/`purge` variable batching, the `doctor` marker shape check. All are small and all are in the audit.
3. **`bench/lexical_scaling.py` needs a read.** F2 built it and left results, but the FTS leg is now the product's dominant retrieval cost (209.8 ms p50 at 60k, 75% of the total) and the fix direction is still unmeasured. That is the highest-value open thread.
