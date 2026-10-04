# Implementation Wave 3 — The FTS Leg, Settled

> ## STATUS UPDATE 2026-09-30 - one finding below is now fixed
>
> This document is a point-in-time record. Its body describes the code as it
> was when the wave closed and is deliberately NOT rewritten. One item has
> since been remediated, so two statements in the body are now stale:
>
> - **The `server.py` fail-open finding (item 2, plus the "bottom line" and
>   the "not fixed here" bullets) is FIXED.** The fail-open `_request_tenant`
>   is deleted and replaced by `_stdio_tenant`, which is correct because stdio
>   carries no token. `attach_tools` now takes an injected `tenant_resolver`;
>   stdio passes the single-tenant default and `http_server` passes the strict
>   `current_tenant`, which is no longer dead code. Pinned by
>   `tests/test_tenant_propagation.py`.
> - **`consolidate` is now tenant-scoped too (2026-09-30).** It was the last
>   open item here. `run_consolidation` takes a tenant and fences all four row
>   accesses: the episode scan, both `_mark_consolidated` queries, and both
>   dedup-pool queries, plus explicit `tenant=` on both store calls. The tool
>   passes the authenticated tenant, so the warning below no longer applies on
>   this count either. Pinned by `tests/test_consolidation_tenant.py`.
> - **The CLI is deliberately unchanged.** `omnidb consolidate` keeps the
>   default tenant: it is the single-tenant local operator path and carries no
>   tenant concept on any other command.
>
> Audit trail: `.omni-audit/AUDIT-REPORT.md` (F11) and
> `.omni-audit/F15-RECOMPUTE.md`.

**Status: the highest-value open thread from Wave 2 is closed, and closing it
invalidate two published artifacts.** Suite green at the start of the wave
(463 passed, 5 skipped, 1 xfailed), mypy clean over 25 source files, and
`ruff check omnidb_server tests bench scripts` now passes where it reported 15
errors on arrival.

| Metric | Wave 2 | Now |
|---|---|---|
| Tests | 393 | **463** (+70) |
| `ruff` over `bench/` + `scripts/` | 15 errors | **clean** |
| FTS lexical-leg cause | 3 unmeasured hypotheses | **measured; 2 refuted** |
| Published lexical artifacts reproducible | claimed yes | **no — both void** |
| Criterion 7 gate in CI | absent | still absent (diagnosed) |

## The headline: two published measurements are void

Both lexical-leg harnesses build their corpus by inserting into `memories` over
raw SQL. They cannot see the rows.

`db.py:96-103` (`_MIGRATION_5`) drops all three FTS triggers and replaces the
external-content table with a standalone one; the index is now synced
explicitly, in-transaction, by the store write path (`store.py:571`, `:658`). A
freshly created database therefore has **no triggers at all** — verified by
reading `sqlite_master` — and `memory_fts` is empty after a raw insert.

Reproduced directly. `bench/lexical_scaling.py` at n=800 on current code:

```
n=    800  p50    0.018 ms  p95    0.029 ms  match set 0 rows (specific 0)
```

against its own committed artifact, `bench/lexical-scaling-results.txt:21-22`:

```
lexical_match_rows_800: 800
lexical_specific_rows_800: 800
```

**The failure is silent, which is what made it survive two waves.** A MATCH
expression that matches nothing returns in microseconds, so the sweep prints a
flat, fast, healthy-looking line and an exponent fitted to it. `0.949` — the
sub-linear alpha that contradicted my `1.60` — is the growth of *an empty index*.
The same construction is in `bench/verify_qdrant_rootcause.py`, so its
`209.8 ms p50 at 60k` lexical figure is also unreproducible.

`lexical_scaling.py` also mislabels its own output: the artifact was produced
with `sizes: 800,2000,4000` but publishes the key `lexical_growth_5k_to_60k`,
and its `gate_status` text asserts "8.4x over the p50 one at 60k" — a number
that run never measured.

## PERF-A2-07-R closed: the ranking clause is 99.6% of the leg

`bench/lexical_cause.py` times SQL variants that differ in exactly one property
each, on one corpus, with the FTS index correctly populated. Six variants, all
production-legal statements; the differences are the ranking clause and the
join. Measured on Python 3.14.4 / win32, 12 queries per size, 2 repeats,
`k_fetch=50`; full artifact in `bench/lexical-cause-results.txt`.

| variant | n=5,000 | n=25,000 | α |
|---|---|---|---|
| **V0** production: `ORDER BY bm25(memory_fts) LIMIT 50` | **4.740 ms** | **30.135 ms** | **1.149** |
| V1 same, `ORDER BY` removed | 0.047 ms | 0.095 ms | 0.440 |
| V2 FTS-only, `ORDER BY rank`, no join | 5.708 ms | 27.758 ms | 0.983 |
| V3 join + predicates, no ranking | 0.051 ms | 0.079 ms | 0.274 |
| V5 identical to V0, match set of width **1** | 0.021 ms | 0.026 ms | 0.148 |
| V4 `COUNT(*)` over the match expression only | 0.796 ms | 6.209 ms | 1.276 |
| match width (OR-ed, as shipped) | 5,000 | 25,000 | — |
| match width (single term) | 1 | 1 | — |

Deltas at n=25,000:

| comparison | cost | reading |
|---|---|---|
| V0 − V1 | **+30.040 ms** | ranking the match set is **99.6%** of the leg |
| V0 − V5 | **+30.109 ms** | and it scales purely with **match-set width** |
| V0 − V2 | +2.378 ms | the `JOIN memories` re-check is ~8%, **not** the cause |
| V1 − V4 | −6.114 ms | fetch and join are cheaper than MATCH itself |

So, against the three candidate causes the audit left open:

1. **Full match-set sort — confirmed, and it is the whole story.** `bm25()` is a
   computed value with no index, so SQLite scores every row the MATCH expression
   touches before `LIMIT` applies. V1 removes only the ordering and drops from
   30.135 ms to 0.095 ms.
2. **The JOIN re-check — refuted.** V2 does the same ranking with no join at all
   and is not faster (α 0.983 vs 1.149). The join is a rounding error.
3. **Sort spill to a temp b-tree — not a separate cause**, as the audit itself
   suspected; it is a mechanism of (1).

**α is 1.149 — super-linear.** The `1.60 vs 0.949` contradiction is resolved:
`0.949` was fitted to an empty index. The two numbers were never measuring the
same thing.

**The actual root cause is one line of `fusion.py`, not the SQL.**
`fts_match_query` (`fusion.py:34-41`) ORs up to 24 lowercased word tokens. An
ordinary question — *"what did note 7 say about the model provider config"* —
shares `note`, `the`, `model`, `provider`, `config` with most rows in the
corpus, so the match set **is** the corpus, and the leg ranks 25,000 rows to
return 50. V5 is the proof: identical statement, identical ranking machinery,
identical `LIMIT`, and a match set of width 1 — 0.026 ms. Ranking is free.
Ranking 25,000 rows is not.

The fix direction follows from V0 − V5: narrow the match set. `lexical_cause.py`
measures that candidate as V6 — the same terms AND-ed instead of OR-ed, same
SQL text, so any difference is attributable to the expression — but **the 60k
band was not completed and no speedup figure is claimed.** See "Not done".

## The harness bug class, and the guard that now catches it

`lexical_cause.py` **refuses to publish** (exit 3) when the median match width
is 0, or when the width-1 probe does not return exactly 1. Both guards fired
during this wave, which is the only reason the void above was found rather than
re-measured.

The second guard exists because my own first version of the probe was wrong. It
used the bare row number as the "unique" token and reported **width 52 at
n=5,000** — because `"7"` also tokenises out of every row whose `i % 97` bucket
is 7. Fixed by folding the id into one alphanumeric token (`ref7qz`). A
discriminator that is not itself asserted is just another number.

This is the third time in three waves that a check was written which cannot
fail: Wave 2's `len()` shape check on the wrong side of `len()`, and Wave 1's
`mutation_check.py` with 8 of 30 anchors silently `NOT_APPLICABLE`. The pattern
is consistent enough to name: **a guard whose condition is not exercised by a
case that would flip it is documentation, not protection.**

## `skills.py` reviewed (Wave 2 left this unread)

The unreviewed 37-line diff adds `_enforce_bundle_size_budget` and calls it
before the first `zf.read()`. **Sound, no defect found.** Two properties worth
recording because they are what make it a real defence rather than a heuristic:

- `ZipInfo.file_size` is the *declared uncompressed* size, and CPython's
  `ZipExtFile.read()` caps every read at `_left = _orig_file_size`. So a
  metadata-only check is a genuine upper bound on what a bomb can produce — it
  is not an estimate that a crafted archive could lie its way past.
- The 64 MiB aggregate is summed over **every** `infolist()` entry, not only the
  two files that get read, so padding the archive with extra members cannot slip
  past the total. That is the detail a naive fix gets wrong.

Scope note: the guard covers the zip path only. If `import_skill` also accepts a
directory-shaped bundle, that path is not covered by this review.

## Also fixed: a file that could not run on its own declared floor

`bench/compute_probe_27.py:86` used PEP 701 nested same-quote f-strings — a
**SyntaxError on Python < 3.12** — while `pyproject.toml:20` declares
`requires-python = ">=3.11"` and `pyproject.toml:66` sets ruff
`target-version = "py311"`. Swapping quotes cannot fix it, so the label is now
hoisted to a `key` variable that is byte-identical to the old expression.

That matters beyond tidiness: the script's sole output,
`bench/review_probe_deltas.json`, is the declared support for the paper's
headline claim (`docs/GROUND_TRUTH.md:272`, `docs/PROJECT_UNDERSTANDING.md:147`).
Reproducing it on the project's declared minimum interpreter previously required
editing the source first.

All 15 ruff findings in `bench/compute_probe_27.py` and
`bench/compute_review_stats.py` are cleared as bare `a; b` splits — no
reordering, no renames, no annotations — so no measurement can shift.
`ruff check omnidb_server tests bench scripts` → `All checks passed!`, and the
absence of the `invalid-syntax` finding is the 3.11 proof.

## Security: the unreviewed modules, reviewed

Four things in the tree had never been read by anyone —
`tenancy.py`, `tenant_schema.py`, `http_server.py`, `docs/adr/`. Findings, with
the correction that matters most stated first.

**The audit's working assumption was wrong in the safe direction.** There *is* an
auth layer. `run_http` defaults `host="127.0.0.1"` (`http_server.py:177`), and
`build_registry` refuses to start with zero credentials
(`http_server.py:163-171`), wired at `cli.py:339` *before* `run_http`. An
unauthenticated remote memory read is not reachable without an operator typing
`--host 0.0.0.0`.

### BLOCKING

1. **`consolidate` is a live cross-tenant read *and* write.**
   `server.py:113-123` exposes it with no `tenant=` argument;
   `consolidate.py:54-58` reads every tenant's episodes with no `tenant_id`
   predicate, and `consolidate.py:314-318` writes the derived facts via
   `store.remember(...)` with no tenant, so they land as `tenant_id='default'`
   (`store.py:495`). Any authenticated tenant calling `consolidate(hours=9999)`
   sends another tenant's private memories to its own configured LLM provider
   and gets the results back as readable rows. This is attack **A6/G6** in
   `docs/MULTI_TENANCY_THREAT_MODEL.md:166`, which the document still marks
   "fully open".
2. **`server.py:19-36` fails open, contradicting the design it copies.**
   `_request_tenant` returns `default` when `get_access_token()` is `None`
   (`:33-34`) and when the claim is missing (`:36`), while
   `http_server.current_tenant` deliberately raises in both cases
   (`http_server.py:120-129`, and `:26-27` asserts "a tool that cannot resolve a
   tenant raises rather than running unscoped"). `server.py` is the one actually
   wired into the tools; `current_tenant` is dead outside tests
   (`tests/test_http_auth.py:27,113,120`). Any middleware or context-propagation
   gap silently degrades to unscoped.
3. **Per-tenant storage is half-wired, and the ADR says otherwise.**
   `tenant_schema.register_and_create` has no production caller —
   `build_stack` (`server.py:145-168`) never calls it — and no `tenant-add`
   subcommand exists (`cli.py:28-129` has only `tenant-key`). So per-tenant FTS
   tables are never created in a real deployment and isolation rests entirely on
   SQL predicates. `docs/adr/ADR-08-multi-tenant-rebuild.md:193` and `:195-196`
   assert both exist.

### Non-blocking

- `tenancy.py:90-98` — `hash_key`'s docstring claims "a **salted** digest"; the
  body is bare `hashlib.sha256` with no salt anywhere in the package. Harmless
  against `os.urandom(24)` keys (`:244`) but the comment misdescribes it.
- `store.py:835-840` — the `search` refusal compares against the literal
  `"default"` rather than `tenancy.DEFAULT_TENANT`. The value is operator-chosen
  at `tenants.json` load and never validated as reserved.
- `store.py:1053-1060` — `rebuild_index(reason)` takes no tenant and rebuilds
  globally. Not exposed as an MCP tool today (CLI `compact` only), but any
  future wrapper inherits a global rebuild.
- `server.py:83-86` — `schema()` is unscoped and returns the `meta` global KV map.
- Dense leg is still global-ANN-then-filter (`recall.py:74`), so threat-model
  G7 and the G4 timing side channel are unmet — exactly as predicted at
  `MULTI_TENANCY_THREAT_MODEL.md:41-45,169`.

### Verified sound

Constant-time credential comparison with no recognised-but-rejected oracle
(`tenancy.py:164-172`); duplicate digests rejected at construction
(`:141-146`); tenant ids validated against `[A-Za-z0-9_-]` **before** any
interpolation (`tenancy.py:67-86`, `tenant_schema.py:58-59`); no CORS or cookie
surface; `recall` fences the BM25 leg, the fused-id refetch, the trigger-pattern
leg and the extra-rows lookup; id-addressed writes fence **inside the
statement** rather than pre-checking (`update_memory`, `forget_many`, `link`,
`supersede`, and both ends of the recursive `related` CTE).

**Bottom line: do not expose `serve-http` on a non-loopback interface until
`consolidate` is tenant-scoped and `_request_tenant` fails closed.**

> UPDATED 2026-09-30: of those two conditions, `_request_tenant` fails
> closed is now DONE (see the status note at the top). `consolidate` was also
> unscoped and is now DONE as well, so both conditions are satisfied: the
> `serve-http` warning below no longer applies.

## Criterion 7 diagnosed: a metric defect, not a regression

The gate is `bench/soak.py:298-300` with limits at `:50-52` — p95 ratio ≤ 4.0,
WAL < 80 MiB, RSS growth < 300 MiB. It is the only stability gate in the repo.

`bench/soak-results-v2.txt` reports `p95_ratio: 5.337`,
`validation_gate_passed: False`. Of the three conjuncts, **exactly one fails and
it is the ratio**: WAL 2.267 MB (2.8% of budget) and RSS growth 29.586 MB (9.9%)
both pass with room. `p95_last_third_ms: 20.739` against a budget of
`4.0 × 3.886 = 15.544 ms`.

**It was never a regression.** Four runs of one workload
(`workload_sha256: c85dd173…` in all four):

| run | first-third p95 | last-third p95 | ratio | verdict |
|---|---|---|---|---|
| `soak-results-v2.txt` | 3.886 | 20.739 | **5.337** | False |
| `soak_stab.txt` | 4.558 | 18.942 | **4.156** | **False** |
| `soak_stab2.txt` | 6.430 | 16.928 | 2.632 | True |
| `soak_stab1.txt` | 7.036 | 16.738 | 2.379 | True |

The numerator is stable (16.7–20.7 ms). **The denominator swings 3.886 → 7.036,
an 81% spread on identical code.** The gate is being crossed by baseline
variance. Both "passing" runs are gitignored (`.gitignore:29-30`) and carry a
different `source_identity_sha256`, so neither is admissible evidence anyway.

**Mechanism.** `recall.py:228-235` bumps `access_count`/`last_accessed_ms` on
every recall and `:238-244` commits that write plus the query-log row, so
**every read is a durable write** — 4,706 of 32,942 ops (14.29%) are commits,
each a WAL append plus fsync under the lock the next write needs. Hence
`recall_p50 5.255 / remember_p50 0.254` = **20.7×**. The soft-deleted tail grows
monotonically to 25,883 rows, so per-recall SQL work grows across the run.

**The gate is not wired into CI.** `.github/workflows/ci.yml:104-125` runs only
`bench/baseline.py` and `tests/test_semantic.py`. `bench/soak.py` can therefore
never fail a build, and `tests/test_post_publish.py:196-198` asserts only that
the keys *exist* — never the verdict.

**Criterion 7 cannot be signed off, and the number must be fixed rather than
re-baselined.** Minimum honest path: (a) drop the per-recall commit, batching
the access bump into the existing 1.0 s `maintenance_tick()` — which changes
`frequency` ranking semantics and must be documented as such; (b) re-run ≥120 s
and commit the artifact **un**-gitignored; (c) execute `bench/soak.py` in CI and
fail on exit 1; (d) gate on the **median** of N≥5 runs, because a single run is
not reproducible at ±81%.

Separate observation, and deliberately kept separate: a *ratio* against a
first-third baseline that itself varies 81% cannot support a 4.0 verdict, so the
acceptance **metric** needs re-specifying — a warm-up exclusion or a denominator
floor — before the next run, not after a failure is seen. **The 4.0 threshold
itself should stand until the write path is fixed and a distribution exists.**

## Not done

- **The 60k band of `bench/lexical_cause.py` was not measured, and the V6
  (AND-ed terms) speedup is unquantified.** The tool shell died mid-sweep — every
  subsequent command returned success with empty stdout, including
  `python --version`. That is an infrastructure failure, not a script failure:
  the 5k and 25k bands completed and the artifact was written before it happened.
  **The α = 1.149 verdict rests on two sizes, not three.**
- `bench/lexical_scaling.py` and `bench/verify_qdrant_rootcause.py` still build
  corpora the broken way. Both need the same FTS back-fill and the same width
  guard before any of their numbers may be cited again. Until then,
  `lexical-scaling-results.txt` and the PERF-A2-07-R figures in
  `A2_PERFORMANCE.md` should be treated as **UNREPRODUCIBLE**.
- _(2026-09-30: `_request_tenant` fail-closed is FIXED and `consolidate`
  tenant scoping is FIXED. The ADR-08 `tenant-add` / `build_stack` wiring below is
  still NOT fixed.)_
- `consolidate` tenant scoping, `_request_tenant` fail-closed, and the ADR-08
  `tenant-add` / `build_stack` wiring are **not fixed here** — they are product
  changes with a live security surface and need their own wave and their own
  regression tests.
- Scratch files `_probe7.py` and `_tmp_gate_review.py` are still in the repo
  root, and the `_qcr/` benchmark scratch (≈950 MB) is still on D:. The shell
  death blocked deletion.
- `bench/compute_review_stats.py:88` builds `fam_bm25`, which `:90` immediately
  overwrites; its `k.endswith("bm25/w25")` predicates can never match a
  `corpus/label/wN` key. A one-line dead store, out of scope here.
- No fix is proposed for `fusion.fts_match_query`. Narrowing the match set is
  the right direction, but it trades recall for latency and that trade is a
  product decision — `fusion.py:14-23` already documents that the lexical leg
  adds noise at *any* weight on a strong embedder. It needs its own
  recall-measurement experiment, not a perf patch.
