# 7 · Cross-Verification

Source: **F1** (claim verification) · **F2** (bug verification) · **F3** (integration consistency) · **F4** (roadmap).

This is the wave that justified the other 25. It ran **adversarially** — the default posture
was skepticism, and it **refuted 4 claims and downgraded 3 more**, 2 of them mine, in my own
`GROUND_TRUTH.md` brief.

---

## §1 · Verification verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | LEGAL-01: no LICENSE in `omnidb-server/` | **CONFIRMED** |
| 2 | BUG-15: `re-embed --model` bricks the home | **REFUTED (STALE)** |
| 3 | SEC-02: one ReDoS regex wedges the store | **PARTIALLY** |
| 4 | BUG-16c: doctor clean on a corrupt marker | **REFUTED as stated** |
| 5 | `.github` at `omnidb-server/`, not the root | **CONFIRMED** |
| 6 | 22 modules / 4,195 LOC; store 1028 | **PARTIALLY** |
| 7 | Soak FAIL: p95_ratio 5.337 vs 4.0 | **CONFIRMED** |
| 8 | `recall` is a write path | **CONFIRMED** |
| 9 | `corpus_mode` undocumented in ARCHITECTURE.md | **CONFIRMED** |
| 10 | All 22 AUDIT-I implemented | **PARTIALLY** |

**1** — a glob over `d:/OMNIDB` returns 73 LICENSE hits, all under `tools/jdk legal/` and
`texlive/`; **zero** in the package. Confirmed by me independently.

**2** — `embeddings.py:128-140` parses `fastembed:<model>` and calls `_load_fastembed(model_name)`;
`make_embedder:162-166` passes `stored_name` through; `:114-119` asserts the identity
round-trips. `test_embedder_identity.py:45-63,66-88` re-embed to a non-default model and
reopen successfully. **My §3.2 described a pre-fix tree.**

**3** — the bounds are real (`validation.py:14-15`, `recall.py:29-30,141,161,170`), the lock is
real (`store.py:158`, **20** `@_serialized` sites), anyio's default **40** is confirmed. **But**
trigger patterns are supplied by the *writer* of the memory, the default deployment is
single-user local stdio, and the same agent owns the home — **self-inflicted, single trust
domain.** A cross-tenant DoS only under the hosted target. Also: the 2048 cap is on **patterns**,
not backtracking steps.

**4** — `doctor.py:82` `n_pending = len(_json.loads(...))`; `len()` always returns a non-negative
int, so `:89-94` is **unreachable**. But `:83-88` *does* return `ok:False`, asserted by
`test_doctor_truthfulness.py:83-95`. **The real, narrower defect:** a valid-JSON-wrong-type
marker → `len()` = 1 → reported `ok:True, "1 content revisions"`. **Silently mislabelled, not
clean.**

**5** — `ci.yml:8-9` `permissions: contents: read`; `:36` `OMNIDB_RUN_PERF: "1"`; `:37`
`--cov-fail-under=80`; `:45-75` `hnsw-required` with skips-as-failures; all `uses:` SHA-pinned.

**6** — 22 modules and 4,195 LOC **CONFIRMED** by my own count. 21 per-module counts match
exactly. **`store.py` = 1,028 is a non-blank count** — physical is ~1,116. Both are true.

**7** — `soak-results-v2.txt:35` `p95_ratio: 5.337`, `:41` `validation_gate_passed: False`,
`:16` `548.91` ops/s. Gate at `soak.py:52`.

**8** — `recall.py:219-221` access bump; `:225-233` query-log + `conn.commit()`.

**9** — 3 hits in `config.py`, 3 in `recall.py`, **0** in `ARCHITECTURE.md`.

**10** — I05, I13, I21 all verified against source. **But I12's evidence claim is false** —
`AUDIT.md:394` asserts p95 3.758 PASS; the artifact says **5.337 / False**.

### The BUG-15 residual, stated precisely
A home is unopenable **not** because `make_embedder` cannot resolve the model, but when the
**installed fastembed catalog no longer contains it** — `_load_fastembed:97-102` raises
`OMN-006`, and `portability.py:185` `reembed_home` resolves the **source** through the same
path, so **you cannot migrate off an unresolvable pin.** That is the genuine lockout, and
BUG-15 does not describe it. `OMNIDB_EMBEDDER=auto` (the default) *does* resolve the stored model.

---

## §2 · Bug verification (agent F2)

| # | Claim | Verdict |
|---|---|---|
| 1 | `import_skill(overwrite=True)` non-atomic | **CONFIRMED** — `skills.py:176` `forget_many` commits, then `:177-183` loops `remember` per row. **Not "half-installed" in the DB-integrity sense — the loss is the original.** |
| 2 | Episodes retired without a stored fact | **CONFIRMED, narrower than implied** — `consolidate.py:325` gates on `created or deduplicated`, but `consumed_ids` (`:333`) is per-successful-chunk, appended at `:254` before fact-level validation. **The `:326-331` comment documents this as intentional.** |
| 3 | `remember_many` rollback hole | **CONFIRMED but near-unreachable** — `lastrowid is None` is impossible for `INTEGER PRIMARY KEY`. A latent trap, not a live bug. |
| 4 | Empty BruteForceIndex on load failure | **CONFIRMED** — `server.py:192-193` calls `__init__` not `.load`, and `_reconcile_index` is **not** suppressed read-only, so a read-only `search` silently returns **zero dense results** on any index-load error. |
| 5 | Unbounded `IN (...)` | **PARTIALLY / OVERSTATED** — see doc 1 §0.5. |
| 6 | hnsw returns < k | **CONFIRMED but overstated as live** — hnswlib's `knn_query` *does* honour `mark_deleted`, so the filter is usually a no-op. **Defensively justified as written.** |
| 7 | No L2 normalization | **CONFIRMED (severity low)** — IP ranking is still *monotonic* with cosine per-document, but cross-document scores scale by ‖v‖, so long docs win ties. Affects only the `semantic` extra. |
| 8 | Trigger re-fetch unfiltered | **REFUTED (safe)** — see doc 1 §0.2. |
| 9 | Predicate re-typed 5× | **CONFIRMED, count = 5** — 4 in consolidation, 1 in store. |
| 10 | RRF dynamic range | **REFUTED (arithmetic wrong)** — see doc 1 §0.3. |
| 11 | Raising `k_fetch` lifts the ceiling | **REFUTED** — see doc 1 §0.4. |
| 12 | `import_jsonl` unchunked encode | **CONFIRMED** — `portability.py:516` one call vs 512 and 256 elsewhere. Mitigated: it encodes *before* any write. |

### A source-quality flag worth recording
**#2 and #10 target behaviour the code pre-emptively justifies in comments.** These look like
findings raised against documented, measured design decisions. **And #11's own hypothesis-test
record in the bench file marks it "REFUTED"** — C2 re-raised a hypothesis the project had
already tested and rejected.

---

## §3 · Integration consistency (agent F3)

| # | Contradiction | Resolution |
|---|---|---|
| C1-C2 | The brief's §3.2 BUG-15 was cited as justification for other work | Any finding whose "why" is "prevents lockout" is invalid. Re-file as an enhancement, **2-4 h not 0.5 d**. |
| C4 | The UAT premise said "4 score fields"; `uat.md:65` asserts 3 | The **actual** shape returns 4 (incl. `frequency`). UAT *understates*. A shape change must touch `recall.py:122-133` **and** `:190-201` atomically — the result dict is built twice. |
| C5 | Any `tags`/`keys` migration | **Hard blocker:** `store.py:188` hardcodes `set_meta(...,"3")`; `db.py:6` = 3; **6 test modules** pin `== SCHEMA_VERSION` and one tests `SCHEMA_VERSION+1` rejection. The migration must land **with** the bump in one commit. |
| C6 | `fusion_scale` | Does not exist; additive so no conflict — but the **measured baseline changes** the moment any weight moves. Benchmark first. |

### The impossible set
**A5 ("never narrow the store lock") vs B1 ("narrow it") cannot both hold for SEC-05.**
Resolution: **A5 wins.** The lock is the correct fix for the concurrency bug; the SEC-05 fix is
**off-lock matching on a bounded pool**, not a narrower lock. **C2 (raise k_fetch) vs B1:** the
pool re-fetch is O(k_fetch), so raising it raises ANN + FTS + re-fetch cost. **C2 loses** — and
it was already refuted on quality grounds.

### Deduplicated work list
Six items were proposed by 2-3 agents each under different names. Merged: `LICENSE`+PEP 639 ·
nested-quantifier screen · logging + correlation + tenant attribution · unlink-after-construct
(both sites) · `HnswlibIndex.update` cap · `final_score` unit tests · `llm_base_url` validation ·
zip byte budget · soak evidence + retractions · tenancy bypass checklist · `re-embed` config
surface · `_after_commit` extraction · `recall` result shape · `import_jsonl` chunking.

### Gaps no agent covered until F3
1. **The test suite is not runnable from the repo root** — `.pytest_cache/lastfailed` =
   `{"omnidb-server/tests": true}`, a bare directory key pytest writes **only on a collection
   error**. *No plan is verifiable until this is fixed. It should be item #0.*
2. `hypothesis` in `[dev]` only → collection error on a clean install.
3. Nothing verified that `texlive/`/`tools/` (644 MB) stay **out of an sdist** — only that they
   are untracked.
4. `server.py`'s 63.5% coverage is **structural**, not incidental — and in a hosted deployment
   the transport *is* the product.
5. The `_acquire_home_lock` duplicate-release paths were never examined for double-acquire or
   fd-leak.

---

## §4 · Prioritisation (agent F4)

**Triage rule:** `(Severity × UserImpact × Confidence) ÷ Effort-hours`, with two hard overrides —
any legal/publication defect ships regardless of score; anything under 20 is backlog and does
not enter a tranche without an attached deletion.

**Week 0 (unblocks verification of everything else):** `tests/conftest.py` + root pytest config ·
`hypothesis` → a `test` extra.

**The 2-week tranche (closed set — 10 days):** `LICENSE` + PEP 639 (**15 m**) · `llm_base_url`
https validation, loopback exempt (1 h) · zip decompressed-byte cap (1 h) · construct-then-swap
in `rebuild_index`/`reembed_home` (1 h) · `HnswlibIndex.update` cap growth (2 h) · `doctor`
marker shape check (30 m) · expose `supersede`/`link` **or delete the README claim** (4 h) ·
rename `half_life_hours` → `recency_tau_hours` (15 m) · **retract the fabricated numbers** in
`AUDIT.md:394` / `perf-report.md:48` / `DATASTORE_SPIKE.md:28` (0.5 d) · `ef_search` 64→128
(**30 m**) · `forget_many`/`purge` batching (3-5 h).

**The 1-month tranche:** CI gates that can actually fail (`min` not `max`; a **cited-number
resolver** that fails if any doc number is absent from a tracked file) · mutation 33%→≥85%
enforced · UAT automation · `obs.py` logging · `errors.py` taxonomy · QA-22 path scrubbing.

**The 1-quarter tranche:** retrieval quality made falsifiable (a tracked 100-query set with a
**Recall@5 floor**) · `store.py` decomposition behind `IndexKeeper` · SQ8 + mmap · retention
ceiling on `queries.query` · in-repo SIGKILL chaos harness.

### Blocked on humans — decide in week 1, these cost one conversation
| Decision | Consequence |
|---|---|
| **Will OmniDB ever be multi-user?** | No → delete the threat model, the 4 xfails, and every "multi-agent namespace" claim. Yes → a 6-8 week storage rebuild that re-plans the quarter. |
| **Local-only or hosted?** | Decides SEC-05, SEC-06, OPS-01, and the whole auth surface. **Local-only demotes SEC-05 to P2.** |
| **RPO/RTO commitment** | No number → no backup guarantee can be published. |
| **Are benchmark artifacts tracked or gitignored?** | The current mix is *how* the fabricated-numbers incident happened. |
| **Licensor confirmation** | Apache-2.0 is inferred from `pyproject` + PyPI metadata. The maintainer must confirm they hold the right to grant it. |

### Risk register for the plan itself
| Risk | Signal | Mitigation |
|---|---|---|
| **Burnout** — ~150 findings feel infinite | Two consecutive days without a merged fix | The 2-week set is closed; "not now" is a **decision**, not a deferral. Stop when it ships. |
| **Re-inflation into the original ambition** (the vault's named risk, already realised once) | Any new module, tool, or "while we're here" | Zero new features in 1.0. Admit-a-feature-admits-a-deletion. The ratchet is 4,300 LOC, not "up to you". |
| **Evidence-integrity recurrence** — the repo has already shipped fabricated numbers once | Prose edited without an artifact | The cited-number resolver is a CI gate. The fix-pass must **delete** claims, not rewrite them. |
| Tenancy answered "yes" in month 3 | Silence on the decision | Force it in week 1; it costs one conversation. |
| Recall regression from the unlink/cap fixes | Parity eval run only after | Run the dry eval **before and after** the 2-week tranche. |
| The paper `.docx` still gets submitted | Someone reads the `.docx` | Delete the artifact; keep the `.tex`, which builds clean. |