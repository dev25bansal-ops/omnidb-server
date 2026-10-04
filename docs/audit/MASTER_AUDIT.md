# OmniDB — Master Audit Deliverable

**30-agent parallel audit** · `d:/OMNIDB/omnidb-server` v0.1.3 · 2026-09-25
**Ground truth:** `docs/GROUND_TRUTH.md` (written first so all agents worked from identical facts)
**Method:** 6 waves × 5 agents, each reading source line-by-line, followed by an adversarial cross-verification wave that **refuted 4 claims** — those corrections are applied below and marked ✅.

> **Read this first.** Three findings in the project vault and in my own initial brief were **wrong** and are corrected here. A prior audit (`ISSUES_AND_REQUIRED_FIXES_2026-09-21.md`) shipped six retracted errors; a fifth would have been embarrassing.

---

## ⚠️ PART 0 — CORRECTIONS (verified by agents F1 + F2, confirmed by me)

| # | Claim that circulated | Verdict | Correct fact |
|---|---|---|---|
| 1 | **`re-embed --model X` bricks the home** (my own GROUND_TRUTH §3.2, and the vault's BUG-15, rated P0) | **REFUTED** | `embeddings.py:74` `FastEmbedEmbedder(model_name=…)` **does** take the model; `cli.py:169` passes it; `embedder_from_name` (`:134`) parses `fastembed:<model>`; `make_embedder` (`:162-166`) resolves the **stored** identity. Two tests pin it (`test_embedder_identity.py:45-88`). **Real residual:** `Config` has no model field, so *switching* models needs `re-embed` — a UX gap (~2-4 h), **not** a P0 lockout. |
| 2 | The skill-trigger re-fetch at `recall.py:177` is unfiltered → cross-tenant leak | **REFUTED** | `triggered` is built *exclusively* from `pat_rows` (`:144-150`), which already applies `_SELECTIVE` + kind + agent_id. `triggered ⊆ pat_rows` by construction. Redundant re-filter, not a leak. |
| 3 | "The RRF fusion term is 15-20× smaller than the modifiers — a scaling bug" | **WRONG NUMBERS, right direction** | Recomputed: RRF range ≈ **0.0051**, recency term **0.15**, frequency term (0-100 accesses) **0.231**. Modifiers dominate by **13-22×**, not 15-20×. The *direction* (fusion is a 4th-decimal tiebreak in memory mode) is real; `corpus_mode` already fixes it for corpora. |
| 4 | "Raise `k_fetch` 50→150 to lift the recall ceiling" | **REFUTED by the project's own data** | `bench/rerank-window-results.md`: scidocs nDCG@10 **0.2084@w25 → 0.2083@w100** (flat) while recall@w goes 0.0562→0.0870. nfcorpus +0.006 (ns). **Zero nDCG gain on all corpora.** The bench file's own hypothesis-test log marks this "REFUTED". Recall@w rises; the scoring function does not convert it. |
| 5 | `forget_many`/`purge` fail "above ~32,766 ids — a small-N bug" | **OVERSTATED** | Real, but the limit is `SQLITE_LIMIT_VARIABLE_NUMBER` = **32,766** (SQLite 3.32+), and `_reconcile_index` batches off the *same* limit. So it's one limit, not two. `forget_many` is reachable via one MCP call with 33k ids. Still a real defect; severity Medium not High. |
| 6 | `remember_many`'s non-`sqlite3.Error` rollback hole | **LATENT, not live** | `store.py:515` raises `StoreError` inside a `try` that only catches `sqlite3.Error` — real structural defect. But `cur.lastrowid is None` is unreachable for a table with `INTEGER PRIMARY KEY` (`db.py`). Fix the pattern; it is not an active bug. |
| 7 | store.py = 1,028 LOC | **BOTH TRUE** | 1,028 **non-blank** lines (PowerShell `Measure-Object -Line`) / **~1,116 physical** lines. The vault's "4,603 LOC" project-wide is ~9% high; module *count* (22) is confirmed. |
| 8 | `supersede`/`link` unreachable | **CONFIRMED** | My own grep across `omnidb_server/` (excluding tests): **zero** callers. Both are advertised in the README/vault as a differentiator. |

---

# 1 · ISSUES & REQUIRED FIXES

## 1.1 Software bugs (agents A1, A4, F2-verified)

| ID | Bug | Location | Sev | Pri | Effort |
|---|---|---|---|---|---|
| **BUG-01** | `import_skill(overwrite=True)` is non-atomic — soft-deletes the old rows, then re-inserts one-per-transaction. A mid-import failure leaves the skill **destroyed**. | `skills.py:176-183` | **Critical** | P0 | 4-8 h |
| **BUG-02** | Consolidation marks episodes `consolidated` when the chunk merely *succeeded*, even if every fact was dropped. 10 episodes → 2 facts → **all 10 retired**. | `consolidate.py:254,326-343` | **Critical** | P0 | 4-6 h |
| **BUG-03** | `open_read_only_store` substitutes an **empty** index on any load failure (uses `__init__`, not `.load()`) → diagnostics report healthy while dense recall returns nothing. | `server.py:192-193` | High | P1 | 1-2 h |
| **BUG-04** | `reembed_home` deletes the rollback copies (`portability.py:273`) **before** the identity-publish commit (`:278-292`) — the last fallible step. | `portability.py:271-273` | High | P1 | 1-2 h |
| **BUG-05** | `HnswlibIndex.persist` publishes graph and tombstones as two `os.replace` calls — a torn pair leaves `index_dirty=0` on disk with mismatched generations. | `vector_index.py:225-234` | High | P1 | 4-8 h |
| **BUG-06** | `_recover_index_state` clears `reembed_pending` after an **unverified** rebuild. | `store.py:230-236` | High | P1 | 3-5 h |
| **BUG-07** | `langgraph.put` is a non-atomic upsert across 3 separate lock acquisitions → duplicate live rows, `get` returns stale. | `adapters/langgraph.py:111-115` | High | P1 | 3-5 h |
| **BUG-08** | `forget_many` / `purge` build unbounded `IN (...)` lists; fail above 32,766. | `store.py:613,904` | Med | P2 | 2-4 h |
| **BUG-09** | `doctor._recovery_check` shape check is **unreachable dead code** (`len()` always returns `int`). A JSON *object* marker reports "1 content revisions", `ok:True`. | `doctor.py:82-94` | Med | P2 | 30-60 m |
| **BUG-10** | `FastEmbedEmbedder` does not L2-normalize while `HnswlibIndex` documents "IP on unit vectors == cosine" and reports `1.0 - dist` as similarity. | `embeddings.py:82-83` | Med | P2 | 2-4 h |
| **BUG-11** | `HnswlibIndex.update()` never grows `cap` (unlike `add()`) → capacity errors on update-heavy workloads. | `vector_index.py:175-178` | Med | P2 | 2 h |
| **BUG-12** | `rebuild_index` unlinks the sidecar **before** constructing the replacement; constructor failure destroys it. | `store.py:941-942` | Med | P2 | 1 h |
| **BUG-13** | `search()` logs `len(rows)` (over-read batch), not the returned count. | `store.py:814` | Low | P3 | 15 m |
| **BUG-14** | `remember_many`'s rollback hole (latent, unreachable — see Correction 6). | `store.py:508-532` | Low | P2 | 1 h |
| **BUG-15** | Untyped `int()` coercion leaks bare `ValueError` off the store surface. | `store.py:610,640,670` | Low | P2 | 1-2 h |

**Reproduction for BUG-01** (highest-impact, and it is a *shipped* data-destroying command):
```python
# export skill "deploy" → have `remember` raise on the 2nd call → import_skill(overwrite=True)
# Result: 0 live rows. The skill is gone and unrecoverable.
```

## 1.2 Performance bottlenecks (agent A2)

| ID | Bottleneck | Measured impact | Sev | Effort |
|---|---|---|---|---|
| **PERF-01** | `bench/baseline.py:137-138` gates on `max(...)` over the sweep — **a gate reading a maximum cannot fail**. Passes on 1.0 (384-d/1k) while the worst cell in the same sweep is **0.17**. | Unfalsifiable CI gate | **P0** | 0.5 d |
| **PERF-02** | recall@10 collapses at scale: **0.519** (384-d) / **0.395** (768-d) at 100k. No shipped path holds ≥0.95. | 10k → 0.888; 100k → 0.519 | **P0** | 3-5 d |
| **PERF-03** | `recall` is a write path: access bump + commit on every read, under the store lock. | 4,706 recalls × 1 commit per 60 s soak | P1 | 0.5-1 d |
| **PERF-04** | No quantized backend. SQ8 exists **only in `bench/baseline.py`**. | 100k recall 0.519→**0.881** at 59 ms; 0.395→**0.896** at 131 ms | P1 | 3-5 d |
| **PERF-05** | `.npz` cannot be mmap'd; `load()` does `np.load` + `astype` + `ascontiguousarray` = **two full copies**. | 307 MB resident @768-d/100k; **614 MB peak at open** | P1 | 6-10 h |
| **PERF-06** | `persist` is a full **O(N)** snapshot; `persist_every_ops=64` fires regardless of the adaptive guard. | 8.2 MB/flush @4k-512d; **308 MB/flush @100k-768d** | P2 | 0.5-1 d (size-aware) / 1-2 w (incremental) |
| **PERF-07** | `import_jsonl` encodes **all** content in one call (siblings chunk at 512/256). | Peak RAM O(N·D) — ~3.5 GB for a 1M-row restore | P2 | 2 h |
| **PERF-08** | Cold start on a **populated** home with recovery is unmeasured; reconcile is O(corpus) on the open path. | "26 ms cold start" is a *fresh empty* home | P2 | 0.5-1 d |
| **PERF-09** | Latency vs qdrant local: **1.2-2.4×** slower at every corpus. Inherent to 2-leg fusion in Python. | nfcorpus 8.1-9.0 vs 4.6-6.1 ms | P2 | not fixable to parity |
| **PERF-10** | CE inference cost spread **274 s → 10,995 s** per cell (40×) — and the paper proves CE *hurts*. | Any "add a reranker" plan is a latency bomb for negative nDCG | P2 | document the "don't" |

### Measurement gaps (nothing exists)
Concurrency ceiling · backup/restore timing · hnsw backend soak (**never run** — both soak and growth are `backend: brute`) · cold start on a populated home · recall quality *through `recall()`* (the v3 bench substituted `BruteForceIndex`) · graph latency · per-tenant disk · >57,638 docs · >4,000 live memories · restore peak RAM · worst-case hostile query.

## 1.3 Security (agent A3 — CVSS 3.1 computed from spec, validated against Log4Shell=10.0 and 7.5 reference vectors)

| ID | Vulnerability | Vector (hosted) | Score | Sev/Pri |
|---|---|---|---|---|
| **SEC-01** | **No tenant column anywhere.** `edges`/`memory_fts`/`meta` have none; `agent_id` is nullable, self-declared, `NULL` = visible to all. 4 confirmed bypasses. | root cause, structural | **Critical / P0** | 1-2 wk |
| **SEC-02** | **id-addressed mutation has no ownership check.** `update_memory`/`forget_many` select by id with a *liveness* filter, not an ownership filter. Ids are sequential → `forget_many([1..10000])` wipes the home. | `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:N` | **9.5 Critical / P0** | 0.5 d predicate |
| **SEC-03** | `search` raw SQL reads every table including those with no tenant column. | `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:N/A:N` | 7.6 High / P0 | 0.5-1 d |
| **SEC-04** | **Prompt injection → durable memory poisoning**, reachable by a plain `remember` with *no* cross-tenant read. The system prompt is not a control; `consolidation_min_confidence` defaults to **0.0** (off) and the confidence is self-reported by the injected model. | `AV:N/AC:L/PR:L/UI:R/S:C/C:H/I:H/A:N` | **8.6 Critical / P0** | 2-4 h for the high-value third |
| **SEC-05** | **ReDoS → store-wide DoS.** Bounds exist (32 patterns/512 chars admission; 4096-char query; 2048 evaluations) but **do not bound backtracking time**. `compiled.search` runs under the store-wide `RLock`; anyio's default **40** worker threads means 40 blocked calls exhaust the pool and the server **stops answering**. | `AV:N/AC:L/PR:L/UI:N/S:C/C:N/I:N/A:H` | 7.6 High / P0 | 2 d |
| **SEC-06** | Bearer token + **full episode transcript** to an unvalidated URL scheme. Nothing rejects non-https. | `AV:A/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N` | 5.3 Med / **P1** | **1 h** |
| **SEC-07** | Zip bundle `zf.read()` with no decompressed-size cap; 3× amplification (bytes→str→list). | `AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H` | 5.5 Med / **P1** | 1 h |
| **SEC-08** | `query-log` dumps **verbatim tenant query text**, not covered by `forget`/`purge`. | `AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N` | 5.5 Med / P1 | 2-3 h |
| **SEC-09** | `SQLITE_FUNCTION` allowlisted unconditionally (authorizer receives the name as `arg2` and ignores it). | `AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N` | 3.3 Low / P2 | 2-3 h |
| **SEC-10** | 28-30 interpolated-SQL sites are safe **only** because every spliced value is a module constant — an unenforced structural property. | no vector | Med / P2 | 2-4 h (test) |

**Controls that PASSED** (a catalog listing only defects misleads): the `search` authorizer is genuinely default-deny with 5 justified carve-outs and its data-modifying-CTE hole is closed *and* tested; `load_extension` fails closed; the AUDIT-I05 three-way budget arithmetic is correct (a real 10,000× error found and fixed); BUG-02's concurrency fix is real; the single-writer lock is correct on both platforms and refuses *before* any mutation; the Ollama probe is loopback-only, 1.5 s, gated off, never raises; the SEC-05 chunk-partitioning fix is layered (designed out + defensive raise).

## 1.4 Code quality (agent A4, 23 findings)

Highest value: **`_SELECTIVE` is re-typed in 5 places** (F2 counted 4 outside store + 1 in store) despite existing · `_KINDS` declared 3× · `_TOKEN = re.compile(r"\w+")` recompiled in 3 modules · `_CONTROL_CHARS` duplicated · `E_CONSOLIDATION_FAILED` never referenced, `"OMN-013"`/`"OMN-005"` as literals · the `store↔recall` cycle · `remember_many` = 139 lines with 9 index-coupled parallel lists (a one-line insert silently desynchronizes ID↔content↔vector) · `cli.main` = one 40-branch if/elif with 9 function-body imports · 4 silent `except: pass` / bare `except Exception` · ~20 zero-call-site items · `store` unannotated everywhere though `py.typed` ships.

## 1.5 Architecture (agent A5)

| ID | Problem | Cost of keeping |
|---|---|---|
| **ARCH-A1** | `store.py` mixes 6 concerns; nothing owns the write contract, so ARCH-02's ordering survives only because one file is the only place anyone edits. | Every new write method is a chance to get dirty→index→persist wrong. |
| **ARCH-A2** | `store ↔ recall` cycle via function-body imports; `recall.py` reaches into 6 store privates. | Any static tool sees a cycle; mypy resolves both directions as `Any`. |
| **ARCH-A3** | One `RLock` wraps an ONNX call and a backtracking regex. Throughput is structurally `1/critical-section`. | The documented 3.8k writes/s and 0.37 ms are **single-lane** numbers. **This is the correct fix for BUG-02 — the cost is SEC-05, not a defect to remove.** |
| **ARCH-A4** | `recall` writes. | Forbids read replicas (the bump feeds `log1p(access_count)` → silent score drift), forbids pure-read eval harnesses. |
| **ARCH-A5** | Single-writer ADR. | Right for v1; the *actual* reason MT-02 is blocked. |
| **ARCH-A6** | 5 recovery markers, no generation number; the if/elif priority chain **is** the consistency policy and is invisible. | Collapse to 1 `index_generation` int + a compare. |
| **ARCH-A7** | 6 hardcoded `vectors.npz` literals, no single source of truth. | A rename silently desyncs backup from writer. |
| **ARCH-A8** | The 3 seams that exist are the 3 nobody needed; the 2 that block the roadmap (storage, scoring) are absent. | — |

## 1.6 Technical debt (agent A5, with business impact)

DEBT-05′ dedup re-embeds a 500-row pool every run (cost scales with KB, not new episodes) · DEBT-09 README's SQLCipher "needs no code changes" is false (3 call sites) · DEBT-11 644 MB of vendored toolchains + 5.4 GB caches in-tree, no documented boundary; a bench script hardcodes a JDK path · DEBT-12 `paper/` ungoverned, 3 generations, DOCX from the oldest · QA-25 `.git` carries ~1.6 GB orphans vs 7 MB of content · **DEBT-13 (new)** `portability.py` reaches into 5 store privates and writes vectors via `_after_commit` — *the one path whose failure loses data* · **DEBT-14 (new)** consolidation = N transactions where `remember_many` exists · **DEBT-15 (new)** `@_serialized` is a decorator convention, not a type-enforced boundary.

---

# 2 · ENHANCEMENTS & MODIFICATIONS

## 2.1 Core engine (`store.py` / `recall.py` — agent B1)

| ID | Enhancement | Value | Effort |
|---|---|---|---|
| ENH-01 | Break the import cycle: `predicates.py` + `errors.py` + a `RetrievalPlan`/`RetrievalContext` seam | Deletes 6 private cross-imports; makes retrieval unit-testable with no store | 8-13 h |
| ENH-02 | Two-phase eligibility probe/hydration — ids-only probe, then hydrate ≤4k | **5× fewer rows at defaults, 100× agent-scoped**; AUDIT-I09 becomes structural | 4-6 h |
| ENH-03 | Centralized `_write_txn` context manager | Closes a *class* of rollback hole, not an instance | 4-6 h |
| ENH-04 | One shared variable-batch helper | 2 hard failures removed (`purge` cannot run above 32,766 rows **forever**) | **3-5 h** |
| ENH-05 | Incremental skill-trigger index (postings by literal token, conservative over-approximation) | Recall cost decoupled from procedural-corpus size; **strictly fewer** regex evaluations ⇒ lowers SEC-05 exposure | 8-14 h |
| ENH-06 | Split `recall`'s critical section — move embed+match off the write lock (2 phases; **do not** do the full RCU — it doubles index RAM) | `stats()` no longer hostage to a wedged recall | 4-6 h + 3-5 h |
| ENH-07 | Decompose `store.py` behind an `IndexKeeper` that owns all 5 markers + ARCH-02 | 1,116→~430 lines; crash-window logic testable without a store | 3-5 d |
| ENH-08 | Surface `corpus_mode` (zeroes both rerank weights, **measured 40-56% nDCG**) — invisible in ARCHITECTURE.md, README, and `schema()` | Operators can see why tuning does nothing | 2-3 h |
| ENH-09 | Per-deployment fusion profiles (the measured optimal weight is 0.7 nfcorpus → ≤0.3 scifact) | Makes the project's own best finding operational | 4-6 h (presets) |

## 2.2 Retrieval quality (agent B2)

| ID | Enhancement | Measured basis | Effort |
|---|---|---|---|
| ENH-10 | **`ef_search` 64 → 128** | 384-d/100k: 0.378→**0.519** for +0.16 ms p50. 10k: 0.695→**0.888** for +0.027 ms. The config comment *itself* cites this bench. | **30 min** |
| ENH-11 | Ship the SQ8 backend (port from `bench/baseline.py`) | 100k: 0.519→**0.881** @59 ms. Must be size-gated — 59-131 ms is 2.4-5.2× the p50 gate | 3-5 d |
| ENH-12 | Enforce L2-normalization + `_check_unit` | Makes the HNSW docstring true for every configurable model; **prerequisite for ENH-11** | 2-3 h |
| ENH-13 | mmap-able brute sidecar (raw `.npy` + sorted ids) | 614 MB → ~0 peak at open; `update()` O(N)→O(log n) | 6-10 h |
| ENH-14 | `fts5vocab` idf (free, no dependency, no write amplification) | The only honest label-free signal for leg weighting | 6-8 h |
| ENH-15 | Fix `zip(rankings, weights)` silently dropping legs (QA-20) | Prerequisite for any third fusion leg | 15 min |
| ENH-16 | New rerankers the fusion *cannot* express: `term_coverage` (idf-weighted), **`mmr`** (agent recall returns near-duplicate facts — the project already treats 0.92 as "same fact" at consolidation), `graph_prior` | All CPU-only; none is a cross-encoder (the paper refutes those) | 10-14 h |
| ENH-17 | Direct unit tests for `fusion`/`scoring` | Mutation 33% → ~97% | 3-4 h |

**Rejected with reasons:** CombSUM (= RRF as k→0, not a new lever) · rank-aware RRF normalization (that is what k=60 already is) · late-interaction/ColBERT (**5.1 GB vs 154 MB at 100k — 34× index growth, breaks the RAM gate; DEFERRED with the arithmetic recorded**) · HyDE on the recall path (violates latency + SEC-06 + local-first).

## 2.3 Interface (agent B3)

ARCH-08 structured logging with a **redaction layer** (P1 prerequisite for all tenancy work) · `OmniDBError` base + complete code table + structured MCP errors · expose `supersede`/`link` (+ a real `correct_memory`) — **the vault's flagship differentiator is unreachable** · gate `search` behind `OMNIDB_ALLOW_SEARCH=0` + add `recent`/`query` so raw SQL is unnecessary · MCP resources for schema/config · `metrics` in Prometheus exposition (no listener) · `recall` `detail="compact"` + **surface `origin`/`confidence`** (stored, never consumed — a security gap) · machine-readable CLI contract with an exit-code table · `doctor` remediation + `severity` · shell completions · generated config reference.

## 2.4 Operations (agent B4)

**Generation-stamped sidecars** — B4 found a scenario the `index_dirty=1` mitigation does *not* cover: sidecar copied *after* the DB cut, carrying a **newer** vector for a row the snapshot shows as older ⇒ silent permanent retrieval corruption, undetectable by membership comparison · DB-only snapshot as the default with a machine-readable restore contract · **a `restore` command with `--dry-run`** (disaster recovery is 5 manual prose steps; `PRAGMA integrity_check` and `foreign_key_check` are run **nowhere** in the product) · `VectorIndexGeneration` object (fixes BUG-04 and BUG-12 structurally) · retention policy + documented scheduler + **restore drill** · versioned JSONL envelope · `--merge` import (the empty-home cliff) · generated config reference · per-corpus profiles · `config_version` · `SCHEMA_VERSION` derived (3 sources today) · migration matrix + `schema --verify` · extend `validation.py` to be the single write-path policy.

## 2.5 Memory layer (agent B5)

`ConsolidationOutcome` (retirement becomes *derived*, fixing BUG-02) · offline stub that only claims distillations the validator accepts (today a 501-2000-char episode is **silently destroyed**) · merge policy default to **annotate** not overwrite (today a model's paraphrase silently replaces a user's words) · cross-chunk agreement (the only model-independent quality signal) · token-accurate budgeting (the shipped llama.cpp example **cannot run a chunk** — default 4096 ctx, no `--ctx-size`) · scheduler (the Glossary claims one exists; none does) · `--emit`/`--apply` human review of unreviewed model output · an extraction contract with per-fact attribution · a `keys` table for the LangGraph adapter (LIKE-on-JSON **cannot find keys containing `"` or `\`** and full-scans on every get/put/delete) · complete LangGraph (`list_namespaces`, `batch`, async) · an adapter contract test.

---

# 3 · ADVANCED FEATURES

## 3.1 Memory architecture (agent C1)
Bitemporal facts + contradiction detection · **evidence-accumulating confidence** (Beta posterior: the LLM's self-report becomes a *prior*, corroboration moves it — the defensible answer to the vault's rejected "decay religion") · **automatic skill induction from episodic traces** (the strongest differentiator: n-gram mining over tool/verb sequences, synthesized triggers, a *mandatory* regex lint because inferred patterns are attacker-influenced, `proposed=True` until promoted, precision tracked, auto-retire below 0.6) · scope trees · counterfactual memory · self-verifying memory (evidence span) · principled forgetting · ambient capture (into a bounded proposal queue, never straight to `remember`).

## 3.2 Retrieval innovation (agent C2)
Respects the paper's own mechanism (**first-stage weakness, not window size**). The one high-value finding: **embedder-aware leg confidence** — the measured 0.416-vs-0.968 failure is *specifically* `hashing-v1` letting the lexical leg outvote correct dense evidence, repairable from the already-stored `meta.embedder` identity. Also: CombMNZ (per-document leg agreement), idf-gated double lexical leg, offline query expansion (trigram tokenizer + shipped synonym table), size-gated two-stage f32 rerank. **Rejected with reasons:** graph-aware retrieval — `link()` has **zero callers**, so the edge set is empty; optimizing an empty graph. Fix ARCH-11 first, PPR is then 1 day. Query-intent routing — no offline training signal exists.

## 3.3 Agent-native experience (agent C3)
`recall_batch` (3-5 sequential recalls → 1 round trip) · a `query` tool so the model never synthesizes SQL · content-hash idempotency on `remember` (a retry currently creates a duplicate that then *votes in fusion*) · `unforget` with stated guarantees · **typed provenance** so an agent can cite and distrust machine-written facts · an envelope return (`list[dict]` cannot express truncation) with a `match` field explaining *why* something surfaced. **Rejected:** lazy tool loading via resources (the client decides; ADR-6 still holds), streaming (no consumer at k≤10), a scratch tier (an LLM already has a context window).

## 3.4 Competitive positioning (agent C4)
**The one sentence the evidence supports:** *"SQLite with retrieval superpowers"* — the vault's own honest framing, and the data backs it: hybrid beats bm25s/pyserini on **all 4** corpora, loses to dense-only only on scidocs. The **published negative result is itself a moat** — a controlled study with an open harness is a trust asset no competitor has. The "never phones home" posture is a 2026 procurement advantage, and it can be made *verifiable* (an egress-audit mode). **Do not build:** a database, a distributed system, a CE reranker, multi-tenancy before the decision.

## 3.5 Resilience & scale (agent C5)
**R6 is the best effort/benefit ratio in the entire audit**: `rebuild_index` already ends in `self.index = fresh` — the pointer swap is atomic and free. Only the lock is wrong. Build the fresh index *outside* the lock, take it for ~1 assignment. 0.5-1 day, and it fixes BUG-12 for free. Plus: a 100k-row chaos harness with an fsync'd ack log (the only way to make "zero data loss" falsifiable), an append-only vector sidecar WAL, a reader snapshot, priority queueing, a real SLO framework, a periodic integrity auditor, canonical/seeded export, a degradation matrix.

**Unhandled failure modes (verified):** disk full mid-flush → write-path exception · read-only FS at runtime · `busy_timeout` exceeded on the write path · clock skew (no clamp) · `doctor` false-clean on a sidecar deleted after a clean close (the 5th marker is evaluated only at open).

---

# 4 · NEW ADDITIONS

## 4.1 New modules (agent D1 — judged against the vault's 1,200-LOC ceiling, which is already breached 3.5×)

| Module | LOC | Why separate | Worth it |
|---|---|---|---|
| `errors.py` | 140 | The OMN-* table is a *wire contract*; 5 classes share no ancestor; the code is baked into `str(exc)` forcing bracket-parsing | **Yes — foundation** |
| `predicates.py` | 70 | SQL vocabulary + the **one** batching helper (3 divergent reserves today; `forget_many`/`purge` have none) | **Yes — smallest LOC-per-defect-fix** |
| `ranking.py` | 90 | `RetrievalPlan` + `RetrievalContext` breaks the cycle | **Yes — mechanical** |
| `index_keeper.py` | 180 | Owns the 5 markers + ARCH-02 in one place — **the prerequisite for splitting `store.py` at all** | **Yes** |
| `querylog.py` | 160 | Owns a table holding **verbatim tenant text** with no redaction/TTL/export control | **Yes — P1** |
| `obs.py` | 200 | Zero logging exists; prerequisite for all tenancy work | **Yes — P1 prereq** |
| `introspect.py` | 150 | `stats`/`schema`/`check` are already pure — cheapest extraction, becomes testable without a store | Yes |
| `migrate_harness.py` | 170 | Migration bugs are **silent data loss** | Yes — highest risk-reduction/LOC |

**Do NOT build:** a config-reference *generator* (a side table is enough) · a generic plugin registry · `tenancy.py` before the privacy defect is fixed · any further `store.py` split before `index_keeper.py` exists.

## 4.2 Schema (agent D5 — with verdicts)

**NOW:** a `tags(memory_id,key,value)` table (one table fixes 3 call sites: skill lookup, LangGraph keys, `list_skills` — all `LIKE`-on-JSON full scans; **pure-SQL migration via `json_extract`**, requires a JSON1 probe next to the FTS5 probe) · `fact_sources` (reverse provenance — "which facts derive from episode 42?" — ~20 LOC, lands only on the opt-in consolidate path) · a `schema_migrations` ledger.
**LATER:** `tenant_id` (FTS5 has no partition key, so it degrades to today's post-filter join; **the cheaper answer is one home per tenant, which is 0 LOC and closes all 4 bypasses**) · valid-time columns.
**NEVER:** a `facts`/`claims` table (duplicates `kind='semantic'` + `metadata.origin`) · edge weight/confidence (edges are user-declared; carries ~zero signal) · FTS over more fields (`memories_au` is `AFTER UPDATE **OF content**` — it would silently not fire on a metadata change, desyncing the index with no error) · per-tenant FTS/HNSW tables (runtime DDL the forward-only ledger cannot gap-check).

## 4.3 Integrations (agent D2 — zero framework imports preserved throughout)
**Finish LangGraph** (`list_namespaces`, `batch`, async — and the current `search` signature *already breaks* the real `BaseStore.search`, which passes a filter dict positionally) · **MCP sampling for consolidation** — architecturally ideal: the client already holds the model, so the transcript crosses **no new boundary**, which structurally removes SEC-06. Requires an LLM injection seam in `run_consolidation` (~1 h) · MCP resources. **Rejected:** a separate `AsyncOmniDBStore` class (a capability, not a class — duplicates everything) · SK MemoryStore (needs collections OmniDB has no primitive for) · **CrewAI (its default stack is hosted OpenAI + LanceDB — violates zero-telemetry)** · any hosted vector DB / observability SaaS / Prometheus pull endpoint.

## 4.4 Tooling (agent D3)
**The insight: the config doc metadata already exists in-repo as 8 docstring comment blocks** — `corpus_mode`, PERF-01/05/07/08, SEC-01/04, E-12. Hoisting them is mechanical, not authoring. Plus `restore --dry-run` (whose real payload is making `integrity_check` reachable *at all*), `doctor` remediation + `severity` (today `_optional_dep` reports a *missing* module as `ok: True`), completions (introspect the live parser → cannot drift), `tools --json` manifest, a `devcheck` pre-push runner.

## 4.5 Packaging (agent D4)
**PKG-01 LICENSE is the only true blocker — 15 minutes gating a package already on PyPI.** No licence = "all rights reserved" by default; corporate OSS scanners fail the build. Then: single version source + CI gate, drop the untested 3.13 classifier, wheel-contents manifest test, `omnidb-bench` as a real distribution (fixes 9 hardcoded paths), a devcontainer (replaces the 331 MB vendored JDK and makes the benchmarks runnable by a third party). **Not worth it yet:** Homebrew/apt, a containerised *server* (there is no daemon), publishing model weights (LEGAL-01 blocks redistribution anyway).

---

# 5 · VERIFICATION & TESTING STRATEGY

## 5.1 Unit & property (agent E1)
The mutation report already names the root cause: *"neither pure function has a single direct unit test."* Its own killing tests are written out — **extracting them takes the mutation score 33% → ~97% in ~4 h.** The mechanical reason 7 scoring mutants survived: all current properties are **non-strict inequalities**, trivially satisfied when a term vanishes. Replace with **exact equality**. Add a stateful `RuleBasedStateMachine` over the 5 markers × 5 kill points (the only way to test the crash window as a *system*), and metamorphic relations (isolation, no-resurrection, backup/restore id-preservation, reconcile idempotence). **Unowned by 29 of 30 agents, found by F3:** `hypothesis` is imported at module level in 3 test files but lives only in `[dev]` → **collection error on a clean install.**

## 5.2 Integration & contract (agent E2)
The tool list is asserted in **5 places that disagree** (10/10/10/7/7) → one generated manifest + a test that pins names, descriptions, and schemas. In-process `call_tool` (the pattern already exists at `test_tools_v2.py:22`) lifts `server.py` from 63.5% — **but it is not the transport**; keep the stdio test. Gaps found: `test_stdio_conformance.py` is a **subset** assertion over 7 tools, so deleting `schema`/`update_memory`/`forget_many` leaves CI green; the AUDIT-I02 "rejected opener persists nothing" claim is **untested** (`test_concurrent_serve.py:139` only checks `returncode != 0`); export is **not** byte-identical across platforms (text mode ⇒ `\r\n` on Windows). **Do not** vendor the official MCP suite (large dep, mostly tests the SDK) — build 2 conformance cases from the spec instead.

## 5.3 Security (agent E3)
**Correction to my own brief:** the VM-step and byte-cap budgets *are* tested (`test_audit_fixes_lead.py:164-189`). The genuine gap is the **wall clock — `_QUERY_WALL_CLOCK_LIMIT_S` appears in ZERO test files.** Highest value: the **AST no-injection test** (walks every `execute` sink, classifies each `FormattedValue` against an allowlist, pins the sink count, and **adds a negative control** — without which a classifier bug makes the test vacuously pass). Then: rewrite the 4 tenancy xfails to **assert the current leak**, so `strict=True` turns a fix into a red suite and a widening into a failed assertion. **Discipline rule:** no regex/zip/fuzz input may run against a live home; fixtures live in `tmp_path`, are code-generated (never committed binaries), and are size- and time-capped.

## 5.4 Performance (agent E4)
**The soak gate is statistically unsound**: a ratio of two p95s (the *least* stable quantile) over a denominator that `growth-results.txt` shows is non-stationary by design — which is why 3 runs with an **identical source hash** gave 4.156 (fail) / 2.379 / 2.632. Replace with a **bootstrap CI on pooled p95 against an absolute per-live-size budget**, ≥3 seeds, plus a Theil-Sen slope CI so creep is separated from noise by the CI, not a ratio. Replace the `max()` gate with a **per-cell frontier** gate (`min` over cells, monotone in `ef`) — **it fails today**, which is the point. Add a latency regression gate (μ+6σ vs a committed baseline, fixed seed). **Cadence rule:** a perf assertion may enter the 9-cell matrix only if it asserts a *floor* (never a time bound) or its threshold exceeds the measured p99.9 noise floor. **Fix CI-05:** delete the absolute 800 w/s assertion and replace it with a self-calibrating paired comparison (immune to runner load by construction).

## 5.5 UAT (agent E5)
`docs/uat.md` has 10 tasks and **all 27 checkboxes are unchecked** — it is an unexecuted document. **8 of 10 tools have no error-path acceptance criterion**, and `recall`'s task is *unfalsifiable* (`:66-67` says a miss "is not a failure"). **`re-embed` — 0.1.3's headline feature — has zero UAT**, no CLI-level test, and no kill-mid-migration test. `doctor` has **12** checks, not the 13 previously claimed. Nothing touches the 22-command CLI beyond ~10.

---

# 6 · BEYOND THE SIX CATEGORIES

## 6.1 What 30 agents collectively MISSED (found by F3)

1. **The test suite is not runnable from the repo root** — `.pytest_cache/lastfailed` = `{"omnidb-server/tests": true}`, a bare directory key pytest writes only on a **collection error**. *No plan can be verified until this is fixed. It should be item #0.*
2. `hypothesis` in `[dev]` only → collection error on a clean install.
3. `corpus_mode` (40-56% nDCG) is behaviour-changing and documented nowhere — a config risk nobody flagged.
4. Nothing verified that `texlive/`/`tools/` (644 MB) stay out of an sdist — only that they're untracked.
5. `server.py`'s 63.5% coverage is **structural**, not incidental — the in-process transport is untested and *is* the product when hosted. No agent proposed fixing the measurement method.
6. The `_acquire_home_lock` duplicate-release paths were never examined for double-acquire/fd-leak.

## 6.2 The plan must respect the vault's own hard-won lesson
> *"Biggest risk is re-inflation into the original ambitions; keep cutting."* — `Comprehensive Analysis (Aug 2026).md:111`

The 1,200-LOC ceiling is breached 3.5×. **Agent F4's verdict: cut it** — it is a wish, not a control, and it has cost eight months of line-counting. Replace with three *enforceable* substitutes: a hard ratchet (runtime ≤4,300 LOC, no exceptions), `store.py` ≤600 post-decomposition with no new module >400, and *a feature that adds a tool must delete one or ship an ADR*.

## 6.3 Blocked on humans (decide in week 1 — these cost one conversation)

| Decision | Consequence |
|---|---|
| **Will OmniDB ever be multi-user?** | No → delete the threat model, the 4 xfails, and every "multi-agent namespace" claim. Yes → a 6-8 week storage rebuild that re-plans the quarter. |
| **Local-only or hosted?** | Decides SEC-05, SEC-06, OPS-01, and the entire auth surface. Local-only **demotes SEC-05 to P2**. |
| **RPO/RTO commitment** | No number → no backup guarantee can be published. |
| **Are benchmark artifacts tracked or gitignored?** | The current mix is *how* the fabricated-numbers incident happened. |
| **Licensor confirmation** | Apache-2.0 is inferred from `pyproject` + PyPI metadata. The maintainer must confirm they hold the right to grant it. |

---

# 7 · THE ORDERED PLAN (agent F4, deduplicated by F3)

**Triage rule:** `(Severity × UserImpact × Confidence) ÷ Effort-hours`, with two hard overrides: any legal/publication defect ships regardless of score; anything under 20 is backlog and does not enter a tranche without an attached deletion.

### Week 0 (unblocks verification of everything else)
`tests/conftest.py` + root pytest config · `hypothesis` → a `test` extra.

### 2-week tranche (closed set — 10 days)
| Item | Effort |
|---|---|
| **`LICENSE` + PEP 639** (publication blocker) | **15 m** |
| `llm_base_url` https validation (loopback exempt) | 1 h |
| Zip decompressed-byte cap | 1 h |
| Construct-then-swap in `rebuild_index`/`reembed_home` (BUG-04, BUG-12) | 1 h |
| `HnswlibIndex.update` cap growth (BUG-11) | 2 h |
| `doctor` marker shape check (BUG-09) | 30 m |
| Expose `supersede`/`link` (ARCH-11) **or delete the README claim** | 4 h |
| Rename `half_life_hours`→`recency_tau_hours` (it is τ, not t½) | 15 m |
| **Retract the fabricated numbers** in `AUDIT.md:394` / `perf-report.md:48` / `DATASTORE_SPIKE.md:28` to the artifacts' actual verdicts | 0.5 d |
| `ef_search` 64→128 (ENH-10) | **30 m** |
| `forget_many`/`purge` batching (ENH-04) | 3-5 h |

### 1-month
CI gates that can actually fail (`min` not `max`; a **cited-number resolver** that fails if any doc number is absent from a tracked file) · mutation 33%→≥85% enforced · UAT automation · `obs.py` logging · `errors.py` taxonomy · QA-22 path scrubbing.

### 1-quarter
Retrieval-quality eval made falsifiable (a tracked 100-query set with a **Recall@5 floor**) · `store.py` decomposition behind `IndexKeeper` · SQ8 + mmap · retention ceiling on `queries.query` · in-repo SIGKILL chaos harness.

### Explicitly NOT doing
Multi-tenancy (blocked) · a CE reranker · the Go/Rust TVF port (**never**) · SQLCipher bundling · remote HTTP transport · MCP resources/prompts · the `.docx` (**delete it** — it ships three retracted claims and a placeholder author block; keep the `.tex`, which builds clean) · Homebrew.

---

## Appendix: what the 30 agents produced

**~150 findings.** Wave A (issues: bugs/perf/security/quality/architecture) · Wave B (enhancements: core/retrieval/interface/ops/memory) · Wave C (advanced: memory-arch/retrieval-innovation/agent-UX/competitive/resilience) · Wave D (additions: modules/integrations/tooling/packaging/schema) · Wave E (testing: unit/integration/security/perf/UAT) · Wave F (verification ×2, integration, prioritisation).

**The adversarial wave earned its cost:** it refuted 4 claims (Corrections 1-4), downgraded 3 more, and found the one gap 29 agents missed (§6.1). Two of the refutations were of claims in **my own brief** — which is exactly why it was written as a file the agents could contradict.

**Every finding is cited `file:line`.** Measured numbers are labelled measured; derived numbers show the arithmetic; unmeasured says so.
