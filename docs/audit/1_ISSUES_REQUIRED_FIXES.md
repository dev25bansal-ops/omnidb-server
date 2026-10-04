# 1 · Issues & Required Fixes

Source: **A1** (bugs) · **A2** (perf) · **A3** (security) · **A4** (quality) · **A5** (arch+debt).
**Read §0 first — 6 claims did not survive verification.**

---

## §0 · Corrections to my own brief (agents F1 + F2, confirmed by me)

| # | Claim | Verdict | Correct fact |
|---|---|---|---|
| 1 | `re-embed --model X` bricks the home (my brief §3.2, vault BUG-15, **P0**) | **REFUTED** | `embeddings.py:74` `FastEmbedEmbedder(model_name=…)` **does** take the model; `cli.py:169` passes it; `embedder_from_name` (`:134`) parses `fastembed:<model>`; `make_embedder` (`:162-166`) resolves the **stored** identity. Two tests pin it (`test_embedder_identity.py:45-88`). **Real residual:** `Config` has no model field, so *switching* models needs `re-embed` — a **2-4 h UX gap, not a P0 lockout.** |
| 2 | Skill-trigger re-fetch at `recall.py:177` unfiltered → cross-tenant leak | **REFUTED** | `triggered` is built *exclusively* from `pat_rows` (`:144-150`), which already applies `_SELECTIVE` + kind + agent_id. `triggered ⊆ pat_rows` by construction. Redundant re-filter, not a leak. |
| 3 | RRF fusion is 15-20× smaller than the modifiers | **numbers wrong, direction right** | Recomputed: RRF range ≈ **0.0051**, recency **0.15**, frequency (0-100) **0.231** ⇒ modifiers dominate **13-22×**. `corpus_mode` already zeroes them for corpora. |
| 4 | "Raise `k_fetch` 50→150 to lift the recall ceiling" | **REFUTED by the project's own bench** | `bench/rerank-window-results.md`: scidocs nDCG@10 **0.2084@w25 → 0.2083@w100** (flat) while recall@w goes 0.0562→0.0870. nfcorpus +0.006 (ns). **Zero nDCG gain on any corpus.** The bench file's own hypothesis log marks it "REFUTED". |
| 5 | `forget_many`/`purge` fail "above ~32,766 — a small-N bug" | **OVERSTATED** | Real, but that is `SQLITE_LIMIT_VARIABLE_NUMBER`, and `_reconcile_index` batches off the **same** limit. One limit, not two. Reachable via one MCP call with 33k ids. Medium. |
| 6 | `remember_many` rollback hole | **LATENT, not live** | `store.py:515` raises `StoreError` inside a `try` catching only `sqlite3.Error` — real structural defect. But `lastrowid is None` is unreachable for `INTEGER PRIMARY KEY`. Fix the pattern; not an active bug. |

---

## §1 · Software bugs

| ID | Bug | Location | Sev | Pri | Effort |
|---|---|---|---|---|---|
| **BUG-01** | `import_skill(overwrite=True)` non-atomic: soft-deletes old rows, then re-inserts **one row per transaction**. Mid-import failure leaves the skill **destroyed and unrecoverable**. | `skills.py:176-183` | **Critical** | P0 | 4-8 h |
| **BUG-02** | Consolidation marks episodes `consolidated` when the chunk merely *succeeded*, even if every fact was dropped. 10 episodes → 2 facts → **all 10 retired, permanently**. | `consolidate.py:254,326-343` | **Critical** | P0 | 4-6 h |
| **BUG-03** | `open_read_only_store` substitutes an **empty** index on any load failure (calls `__init__`, not `.load()`) → diagnostics report healthy while dense recall returns **nothing**. | `server.py:192-193` | High | P1 | 1-2 h |
| **BUG-04** | `reembed_home` deletes rollback copies **before** the identity-publish commit (the last fallible step). | `portability.py:271-273` | High | P1 | 1-2 h |
| **BUG-05** | `HnswlibIndex.persist` publishes graph + tombstones as **two** `os.replace` calls — a torn pair leaves `index_dirty=0` with mismatched generations. | `vector_index.py:225-234` | High | P1 | 4-8 h |
| **BUG-06** | `_recover_index_state` clears `reembed_pending` after an **unverified** rebuild. | `store.py:230-236` | High | P1 | 3-5 h |
| **BUG-07** | `langgraph.put` non-atomic upsert across 3 lock acquisitions → duplicate live rows; `get` returns the stale one. | `adapters/langgraph.py:111-115` | High | P1 | 3-5 h |
| **BUG-08** | `forget_many`/`purge` build unbounded `IN (...)` lists. | `store.py:613,904` | Med | P2 | 2-4 h |
| **BUG-09** | `doctor._recovery_check` shape check is **unreachable dead code** (`len()` always returns `int`). A JSON *object* marker reports "1 content revisions", `ok:True`. | `doctor.py:82-94` | Med | P2 | 30-60 m |
| **BUG-10** | `FastEmbedEmbedder` does not L2-normalize while `HnswlibIndex` documents "IP on unit vectors == cosine" and reports `1.0 - dist` as similarity. | `embeddings.py:82-83` | Med | P2 | 2-4 h |
| **BUG-11** | `HnswlibIndex.update()` never grows `cap` (unlike `add()`) → capacity errors on update-heavy workloads. | `vector_index.py:175-178` | Med | P2 | 2 h |
| **BUG-12** | `rebuild_index` unlinks the sidecar **before** constructing the replacement; constructor failure destroys it. | `store.py:941-942` | Med | P2 | 1 h |
| **BUG-13** | `search()` logs `len(rows)` (over-read batch), not the returned count. | `store.py:814` | Low | P3 | 15 m |
| **BUG-14** | `remember_many` rollback hole (latent, unreachable — §0.6). | `store.py:508-532` | Low | P2 | 1 h |
| **BUG-15** | Untyped `int()` coercion leaks bare `ValueError` off the store surface. | `store.py:610,640,670` | Low | P2 | 1-2 h |

### BUG-01 reproduction — a *shipped*, data-destroying command
```python
# export skill "deploy" → make the 2nd remember() raise → import_skill(overwrite=True)
# Result: 0 live rows. The skill is gone and unrecoverable.
# skills.py:176 forget_many() commits; :177-183 loops store.remember() per row,
# each its own transaction (store.py:527).
```

### BUG-02 reproduction
```python
# llm stub returns 2 valid facts from a 10-episode chunk (consolidation_chunk_episodes=10)
# consolidate.py:254  consumed_source_ids extended for the whole successful chunk
# consolidate.py:325  if created or deduplicated:   <-- gates on "any", not "all"
# consolidate.py:337  mark ALL 10 episodes consolidated -> source text now unreachable
```
`consolidate.py:326-331` documents this as *intentional* ("the LLM saw each transcript"). So it
is a real accounting gap the design **knows about and accepts**. The fix is not to stop — it
is to make retirement *derived* from what was stored (agent B5's `ConsolidationOutcome`).

### Controls that PASSED (a defect-only catalog misleads)
- The `search` authorizer is genuinely **default-deny** with 5 justified carve-outs; its
  data-modifying-CTE hole is closed **and tested** (`test_security_matrix.py`).
- `load_extension` fails closed — Python's sqlite3 disables extension loading.
- The AUDIT-I05 three-way budget **arithmetic is correct** (a real 10,000× error was found
  and fixed in an earlier pass).
- The store-wide RLock is **correct** on both platforms and refuses *before* any mutation.
- The Ollama probe is loopback-only, 1.5 s budget, gated off, never raises.
- The SEC-05 chunk-partitioning fix is **layered**: designed out *and* defensively raised.

---

## §2 · Performance bottlenecks

Gates are interactive latency, cold start, RAM — **not QPS** (ADR-7).

| ID | Bottleneck | Measured impact | Sev | Effort |
|---|---|---|---|---|
| **PERF-01** | `bench/baseline.py:137-138` gates on `max(...)` over the sweep — **a gate reading a maximum cannot fail.** Passes on 1.0 (384-d/1k) while the worst cell in the same sweep is **0.17**. | Unfalsifiable CI gate | **P0** | 0.5 d |
| **PERF-02** | recall@10 collapses at scale: **0.519** (384-d) / **0.395** (768-d) at 100k. No shipped path holds ≥0.95. | 10k → 0.888; 100k → 0.519 | **P0** | 3-5 d |
| **PERF-03** | `recall` is a **write path**: access bump + commit on every read, under the store lock. | 4,706 recalls × 1 commit per 60 s soak | P1 | 0.5-1 d |
| **PERF-04** | No quantized backend. SQ8 exists **only in `bench/baseline.py`**. | 100k recall 0.519→**0.881** @59 ms; 0.395→**0.896** @131 ms | P1 | 3-5 d |
| **PERF-05** | `.npz` cannot be mmap'd; `load()` = `np.load` + `astype` + `ascontiguousarray` = **two full copies**. | 307 MB resident @768-d/100k; **614 MB peak at open** | P1 | 6-10 h |
| **PERF-06** | `persist` is a full **O(N)** snapshot; `persist_every_ops=64` fires regardless of the adaptive guard. | 8.2 MB/flush @4k-512d; **308 MB/flush @100k-768d** | P2 | 0.5-1 d size-aware / 1-2 w incremental |
| **PERF-07** | `import_jsonl` encodes **all** content in one call (siblings chunk at 512/256). | Peak RAM O(N·D) — ~3.5 GB for a 1M-row restore | P2 | 2 h |
| **PERF-08** | Cold start on a **populated** home with recovery unmeasured; reconcile is O(corpus) on the open path. | "26 ms cold start" is a *fresh empty* home | P2 | 0.5-1 d |
| **PERF-09** | Latency vs qdrant local: **1.2-2.4×** slower at every corpus. Inherent to 2-leg fusion in Python. | nfcorpus 8.1-9.0 vs 4.6-6.1 ms | P2 | not fixable to parity |
| **PERF-10** | CE inference spread **274 s → 10,995 s** per cell (40×) — *and the paper proves CE hurts*. | Any "add a reranker" plan = latency bomb for negative nDCG | P2 | document the "don't" |

### The three persist triggers
`store.py _maybe_persist` fires on: `persist_every_ops=64` (op count), `persist_min_ops=8` +
adaptive interval, `persist_max_s=5.0`. AUDIT-I18 **deliberately** lets the op-count trigger
override adaptive time control. At 100k-768d that is a **308 MB full rewrite every 64
operations.** The guard exists for crash recovery; it is also the dominant write cost at
scale. Make it size-aware or move to an incremental sidecar.

### Measurement gaps — nothing exists
Concurrency ceiling · backup/restore timing · hnsw backend soak (**never run** — both soak
and growth are `backend: brute`) · cold start on a populated home · recall quality *through
`recall()`* (the v3 bench substituted `BruteForceIndex`) · graph latency · per-tenant disk ·
>57,638 docs · >4,000 live memories · restore peak RAM · worst-case hostile query ·
multilingual quality.

### Measured detail worth keeping
- Recall p50 **0.37 ms** @<1k rows; p95 **17.6 ms** in soak at 548.91 ops/s.
- `remember` **3.8 k/s**; `remember_many(1000)` **~29 k/s**.
- Brute p50 0.99 → 3.39 ms from 110 → 4,000 rows.
- **`ef_search` 64→128 is the cheapest quality win in the repo**: 0.378→0.519 @100k for
  **+0.16 ms**; 0.695→0.888 @10k for **+0.027 ms**. `config.py:100-102` *itself* cites this
  bench while shipping 64.

---

## §3 · Security

**CVSS 3.1 computed from the base equations, validated against Log4Shell=10.0 and a 7.5
reference vector before any number was taken.** The 09-21 audit computed CVSS by hand, got it
wrong three ways, and retracted every score. Coefficients that were wrong: `PR:L` is 0.62
under Scope-Unchanged but **0.66** under Scope-Changed; `UI:N` is **0.85**, never 1.0; `ISC`
with `A:H` is **0.56**, not 0.50.

**(A) As shipped** = local stdio, one session, no listener. **(B) Hosted multi-tenant** =
tenants mutually distrustful. Scores are (B) where they differ.

| ID | Vulnerability | (B) vector | Score | Sev/Pri |
|---|---|---|---|---|
| **SEC-01** | **No tenant column anywhere.** `edges`/`memory_fts`/`meta` have none; `agent_id` nullable, self-declared, `NULL` = visible to all. 4 confirmed bypasses. | structural | **Critical / P0** | 1-2 wk |
| **SEC-02** | **id-addressed mutation has no ownership check.** `update_memory`/`forget_many` filter by *liveness*, not ownership. Ids are sequential → `forget_many([1..10000])` wipes the home. | `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:N` | **9.5 Critical / P0** | 0.5 d |
| **SEC-03** | `search` raw SQL reads every table including those with no tenant column. | `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:N/A:N` | 7.6 High / P0 | 0.5-1 d |
| **SEC-04** | **Prompt injection → durable memory poisoning**, reachable by a plain `remember` with **no** cross-tenant read. The system prompt is not a control; `consolidation_min_confidence` defaults to **0.0** and the confidence is self-reported by the injected model. | `AV:N/AC:L/PR:L/UI:R/S:C/C:H/I:H/A:N` | **8.6 Critical / P0** | 2-4 h for the high-value third |
| **SEC-05** | **ReDoS → store-wide DoS.** Bounds exist (32/512 admission; 4096-char query; 2048 evals) but **do not bound backtracking time**. `compiled.search` runs under the store `RLock`; anyio's default **40** workers means 40 blocked calls exhaust the pool and the server **stops answering**. | `AV:N/AC:L/PR:L/UI:N/S:C/C:N/I:N/A:H` | 7.6 High / P0 | 2 d |
| **SEC-06** | Bearer token + **full episode transcript** to an unvalidated URL scheme. Nothing rejects non-https. | `AV:A/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N` | 5.3 Med / **P1** | **1 h** |
| **SEC-07** | Zip `zf.read()` with no decompressed-size cap; 3× amplification (bytes→str→list). | `AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H` | 5.5 Med / **P1** | 1 h |
| **SEC-08** | `query-log` dumps **verbatim tenant query text**, not covered by `forget`/`purge`. | `AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N` | 5.5 Med / P1 | 2-3 h |
| **SEC-09** | `SQLITE_FUNCTION` allowlisted unconditionally (the authorizer receives the name as `arg2` and ignores it). | `AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N` | 3.3 Low / P2 | 2-3 h |
| **SEC-10** | 28-30 interpolated-SQL sites are safe **only** because every spliced value is a module constant — an unenforced structural property. | no vector | Med / P2 | 2-4 h (test) |

### On SEC-01 and the CVSS blind spot
CVSS v3.1 has **no term for how many tenants one row leaks to**, so a per-row cross-tenant
disclosure and a single-tenant disclosure of the same content score identically. Use the
severity column for triage; **do not let a 7.3 talk you out of a Critical.**

### SEC-05 severity depends on a decision only the owner can make
As shipped (local stdio, one trust domain, the agent that writes trigger patterns is the same
agent that reads them) the mechanism is real but **self-inflicted**. It becomes a
**cross-tenant DoS** only under the hosted target. If the answer to "will OmniDB ever be
multi-user?" is **no**, SEC-05 drops to P2.

### SEC-10 — the mechanical test (high value, no exploit needed)
Walk every `execute`/`executemany`/`executescript` sink via `ast`; classify each
`FormattedValue` as SAFE only if it resolves to a module constant, a `', '.join()` over names
from a static list, or an int `*_CAP`. Assert `unsafe == []` and **pin the sink count**.
**Add a negative control** — inject a bad `execute` into a temp copy and assert the classifier
flags it — or a classifier bug makes the test vacuously pass.

---

## §4 · Code quality (agent A4, 23 findings)

| Sev | Finding | Location |
|---|---|---|
| High | `_SELECTIVE` re-typed in **5 places** despite existing | `consolidate.py:56,158,166`; `portability.py:337`; `store.py:1054` |
| High | `_KINDS` declared **3×** | `store.py:24`; `portability.py:17`; `skills.py:12` |
| High | `store ↔ recall` import cycle held by function-body imports | `store.py:733` ↔ `recall.py:10` |
| High | `remember_many` = 139 lines, **9 index-coupled parallel lists**; a one-line insert silently desynchronizes ID↔content↔vector | `store.py:399-537` |
| High | `run_recall` ~186 lines, nesting depth 5, **3 duplicated result-dict builders** | `recall.py:48-234`; `:122,190,204` |
| High | **4 silent `except: pass` / bare `except Exception`** degradations | `store.py:119`; `server.py:192`; `llm.py:21`; `vector_index.py:289,304` |
| Med | `_TOKEN = re.compile(r"\w+")` recompiled in **3 modules** | `fusion.py:7`; `rerank.py:30`; `embeddings.py:14` |
| Med | `_CONTROL_CHARS` duplicated verbatim | `store.py:27`; `consolidate.py:12` |
| Med | `E_CONSOLIDATION_FAILED` **never referenced**; `"OMN-013"`/`"OMN-005"` as literals | `store.py:47`; `consolidate.py:260`; `cli.py:293` |
| Med | `cli.main` = one **40-branch if/elif** with 9 function-body imports | `cli.py:124-275` |
| Med | `reembed_home` 180 lines, try/finally ×2, 6 mutable locals | `portability.py:129-310` |
| Med | `_classify_sqlite_error` classifies by **substring match** on SQLite message text | `store.py:1105-1115` |
| Med | ~8 unnamed magic numbers incl. one in a module whose own rule is "named constant + test" | `store.py:120,768,961,162`; `recall.py:93,104`; `consolidate.py:33,107` |
| Med | `store` param unannotated everywhere though `py.typed` ships | `recall.py:48`; `consolidate.py:53,185`; `portability.py:364,426` |
| Med | Docstring says a freshness walk exists; the code does neither | `consolidate.py:163-166` vs `:139-145` |
| Low | `_PATTERN_CACHE` process-global, no reset hook, no test isolation | `recall.py:19` |
| Low | 0-byte `__init__.py` in `docs/`/`bench/`/`examples/`/`scripts/` make them importable top-level | contradicts the `pyproject.toml:43-45` comment |
| Low | `_sanitize` caps at 2000 chars, then `_valid_fact` rejects at 500 — **1500 wasted chars per fact** | `consolidate.py:46-50` vs `:122-124` |

**One flag:** my brief called `recall.py:104`'s `max(k*4,k)` a "no-op". **It is not** — with
`k=1, k_fetch=50` it silently discards ~46 scored candidates, and unlike `search()` it sets no
`truncated` flag.

---

## §5 · Architecture (agent A5)

| ID | Problem | Cost of keeping |
|---|---|---|
| **ARCH-A1** | `store.py` mixes 6 concerns; nothing owns the write contract, so ARCH-02's ordering survives only because one file is the only place anyone edits. | Every new write method is a chance to get dirty→index→persist wrong. |
| **ARCH-A2** | `store ↔ recall` cycle via function-body imports; `recall.py` reaches into 6 store privates. | Any static tool sees a cycle; mypy resolves both directions as `Any`. |
| **ARCH-A3** | One `RLock` wraps an ONNX call and a backtracking regex. Throughput is structurally `1/critical-section`. | The documented 3.8k writes/s and 0.37 ms are **single-lane** numbers. **This is the correct fix for the concurrency bug — the cost is SEC-05, not a defect to remove.** |
| **A4** | `recall` writes. | Forbids read replicas (the bump feeds `log1p(access_count)` → silent score drift), forbids pure-read eval harnesses. |
| **A5** | Single-writer ADR. | Right for v1; the *actual* reason SEC-01 is blocked. |
| **A6** | 5 recovery markers, no generation number; the if/elif priority chain **is** the consistency policy and is invisible. | Collapse to 1 `index_generation` int + a compare. |
| **A7** | 6 hardcoded `vectors.npz` literals, no single source of truth. | A rename silently desyncs backup from writer. |
| **A8** | The 3 seams that exist are the 3 nobody needed; the 2 that block the roadmap (storage, scoring) are absent. | — |

**ARCH-A3 is the one people will get wrong.** The store-wide RLock is the *correct* fix. Its
price is SEC-05's blast radius. Do not "optimize" it away — the fix for SEC-05 is off-lock
matching on a bounded pool, not a narrower lock.

---

## §6 · Technical debt (agent A5, with business impact)

| ID | Item | Business impact | Effort | Defer-risk |
|---|---|---|---|---|
| DEBT-01 | purge covers superseded rows | ✅ verified fixed — closed | — | — |
| DEBT-02 | wheel excludes `tests`/`docs` | ✅ verified fixed — closed | — | — |
| DEBT-05′ | dedup re-embeds its 500-row pool **every run** | With FastEmbed, every consolidation pays a full pool re-encode; cost scales with KB, not new episodes | 2-4 h | Low |
| DEBT-08 | ~20 zero-call-site items | Dead code reads as supported surface | 0.5 d | Low |
| DEBT-09 | README's SQLCipher "needs no code changes" is **false** (3 call sites) | Customer plans an encryption swap on a false promise | 1 h | Med |
| DEBT-11 | 644 MB vendored toolchains + 5.4 GB caches in-tree, no documented boundary; a bench script hardcodes a JDK path | Every clone re-derives the boundary; bench reproducibility is path-locked | 2-4 h | Med |
| DEBT-12 | `paper/` ungoverned: 3 generations, DOCX from the oldest, no build gate, no CI | A paper claim can contradict the code with no test noticing | 0.5 d | Low |
| QA-25 | `.git` carries ~1.6 GB orphans vs 7 MB of content | 200× clone payload, slow CI checkout | 10 m + approval | Low |
| **DEBT-13** | `portability.py` reaches into 5 store privates and writes vectors via `_after_commit` | **The one code path whose failure loses data** is coupled to store internals; a refactor silently breaks restore | 3-4 h | **High** |
| **DEBT-14** | consolidation = N transactions where `remember_many` exists | The slowest path in the product; O(N) fsyncs | 2-3 h | Med |
| **DEBT-15** | `@_serialized` is a decorator convention, not a type-enforced boundary | A new write method that forgets it reintroduces the concurrency bug silently; 20 sites, no test | 2 h | **High** |
| **DEBT-16** | `ARCHITECTURE.md:24` lists 4 CLI subcommands (20 exist); `corpus_mode` documented nowhere | The single most behaviour-changing flag is invisible | 2 h | Med |

### Closed during the audit — do not re-report
| ID | Item | Note |
|---|---|---|
| DEBT-01 | purge covers superseded rows | verified fixed |
| DEBT-02 | wheel excludes `tests`/`docs` | verified fixed |
| §0.1 | `re-embed --model` lockout (mine, P0) | **REFUTED** — `embeddings.py:128-140` resolves the stored model; 2 tests pin it |
| §0.4 | raise `k_fetch` to lift recall | **REFUTED by the project's own bench** |

---

## §7 · The 2-week tranche (F4's closed set — this is what to actually do)

Ordered by triage score `(Severity × UserImpact × Confidence) ÷ Effort-hours`, with two hard
overrides: any legal/publication defect ships regardless of score; anything under 20 is backlog
and does not enter a tranche without an attached deletion.

| # | Item | Effort | Why now |
|---|---|---|---|
| 1 | **`LICENSE` + PEP 639 `license-files`** | **15 m** | `pyproject.toml:10` declares Apache-2.0 and `MANIFEST.in:3` includes a file that does not exist. The package is already on PyPI. |
| 2 | `llm_base_url` https validation, loopback exempt | 1 h | SEC-A3-05: 1 h closes a bearer-token + transcript egress. Best effort/severity ratio in the catalog. |
| 3 | zip decompressed-byte cap | 1 h | SEC-A3-06: check `ZipInfo.file_size` before `zf.read()`. |
| 4 | construct-then-swap in `rebuild_index` / `reembed_home` | 1 h | BUG-A1-12 + BUG-A1-04, both pure ordering. |
| 5 | `HnswlibIndex.update` cap growth | 2 h | BUG-A1-11: extract `_ensure_capacity(n)`. |
| 6 | `doctor` marker shape check | 30 m | BUG-A1-09: `len()`-based guard is dead code. |
| 7 | expose `supersede`/`link` **or delete the README claim** | 4 h | The vault's flagship differentiator is unreachable from MCP. |
| 8 | rename `half_life_hours` → `recency_tau_hours` | 15 m | It is not a half-life. |
| 9 | **`ef_search` 64 → 128** | **30 m** | +0.117…+0.193 recall@10 for +0.027…+0.208 ms. `config.py:100-102` cites the bench that proves it, then ships 64. |
| 10 | `forget_many`/`purge` variable batching | 3-5 h | One helper; removes a permanent failure at >32,766 rows. |
| 11 | **retract the fabricated numbers** | 0.5 d | `AUDIT.md:394`, `perf-report.md:48`, `DATASTORE_SPIKE.md:28`, `results.md` vs `perf-report.md:34`. **Delete the claims, do not rewrite them.** |

**Item 11 is the one that is not a code change and matters most.** This repository has already
shipped numbers that do not exist in any artifact. A fix-pass that *edits prose* rather than
*deleting claims* reproduces the failure.

### Blocked on humans — week 1, one conversation
| Decision | Consequence |
|---|---|
| **Will OmniDB ever be multi-user?** | No → delete the threat model, the 4 xfails, every "namespace" claim. Yes → a 6-8 week storage rebuild that re-plans the quarter. |
| **Local-only or hosted?** | Decides SEC-05, SEC-06, OPS-01, and the whole auth surface. **Local-only demotes SEC-05 to P2.** |
| **RPO/RTO commitment** | No number → no backup guarantee can be published. |
| **Are bench artifacts tracked or gitignored?** | The current mix is *how* the fabricated-numbers incident happened. |
| **Licensor confirmation** | Apache-2.0 is inferred from `pyproject` + PyPI metadata. The maintainer must hold the right to grant it. |
