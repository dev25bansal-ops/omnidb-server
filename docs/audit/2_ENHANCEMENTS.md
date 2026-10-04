# 2 · Enhancements & Modifications

Source: **B1** (core engine) · **B2** (retrieval) · **B3** (interface) · **B4** (ops) · **B5** (memory layer).
Each item names the target, the concrete change, the measured basis, and the **trade-offs**.

---

## §1 · Core engine — `store.py` / `recall.py` (agent B1)

| ID | Enhancement | Value | Effort |
|---|---|---|---|
| ENH-01 | Break the import cycle: extract `predicates.py` + `errors.py`, then a `RetrievalPlan` frozen dataclass + `RetrievalContext` Protocol | Deletes 6 private cross-imports; makes retrieval unit-testable **with no store, no connection, no index** | 8-13 h |
| ENH-02 | Two-phase eligibility probe/hydration — an ids-only probe that applies eligibility+order+window in **one** statement, then hydrates ≤4k rows | **5× fewer rows at defaults, 100× agent-scoped.** AUDIT-I09 becomes *structurally* true rather than true-by-placement | 4-6 h |
| ENH-03 | Centralized `_write_txn` context manager (rollback on **any** exception, not just `sqlite3.Error`) | Closes a **class** of rollback hole, not an instance | 4-6 h |
| ENH-04 | One shared variable-batch helper | 2 hard failures removed — **`purge` can never run above 32,766 rows, permanently** | **3-5 h** |
| ENH-05 | Incremental skill-trigger index (postings by literal token; a **conservative over-approximation** so a match is never lost) | Recall cost decoupled from procedural-corpus size; **strictly fewer** regex evaluations ⇒ lowers SEC-05 exposure. Include a differential test vs today's full scan | 8-14 h |
| ENH-06 | Split `recall`'s critical section — move embed + match off the write lock. **Do not** do the full RCU/double-buffer: it doubles index RAM against a RAM gate | `stats()`/`check()` no longer hostage to a wedged recall | 4-6 h + 3-5 h |
| ENH-07 | Decompose `store.py` behind an `IndexKeeper` that owns all 5 markers + the ARCH-02 post-commit ordering | 1,116→~430 lines; crash-window logic testable without a store | 3-5 d |
| ENH-08 | Surface `corpus_mode` (zeroes both rerank weights, **measured 40-56% nDCG**) — invisible in ARCHITECTURE.md, README, and `schema()` | Operators can finally see why tuning does nothing | 2-3 h |
| ENH-09 | Per-deployment fusion profiles | The measured optimal weight is 0.7 (nfcorpus) → ≤0.3 (scifact); today one home = one weight | 4-6 h |

### The three highest-leverage ones, in detail

**ENH-04 (best effort-to-defect ratio in the whole audit, ~4 h).** `store.py:263` and
`recall.py:93` already batch on `SQLITE_LIMIT_VARIABLE_NUMBER` — with **two different magic
reserves**. `forget_many` (`:613`) and `purge` (`:904`) have **no batching at all**. Result:
`purge` on a home with >32,766 tombstoned rows **cannot complete, ever**; `check()` reports
nothing wrong. `forget_many` is reachable from a single MCP call with a 33k-element array.
One helper, both fixed, and the `-2`/`-4`/`-8` reserves stop being folklore.

**ENH-02.** Today the **entire fused pool** is re-fetched with `SELECT *`, then truncated to
`max(4k,k)`. At shipped defaults that is 100 rows hydrated, 20 kept — **5× waste**; agent-scoped,
2000 hydrated, 20 kept — **100× waste**. And because it is `SELECT *`, every discarded row
materialises its `content` (capped at 100k chars) only to be thrown away. Splitting the probe
(returns ids) from the hydrate (explicit column list) fixes both, and makes AUDIT-I09
impossible to reintroduce.

**ENH-05.** Every recall scans **all** live procedural rows and evaluates up to 2,048 regexes —
cost independent of `k`, `k_fetch`, or how many patterns could match. A literal-token prefilter
turns that into O(query tokens + candidates). The correctness argument matters: the prefilter is
an over-approximation, so the set of evaluations is a **subset** of today's and the triggered
set is **identical**. The differential test is what makes that falsifiable rather than hopeful.

---

## §2 · Retrieval quality (agent B2)

| ID | Enhancement | Measured basis | Effort |
|---|---|---|---|
| ENH-10 | **`ef_search` 64 → 128** | 384-d/100k: 0.378→**0.519** for +0.16 ms p50. 10k: 0.695→**0.888** for +0.027 ms. The config comment *itself* cites this bench. | **30 min** |
| ENH-11 | Ship the SQ8 backend (port from `bench/baseline.py`) | 100k: 0.519→**0.881** @59 ms. Must be size-gated — 59-131 ms is 2.4-5.2× the p50 gate | 3-5 d |
| ENH-12 | Enforce L2-normalization + a `_check_unit` assertion | Makes the HNSW docstring true for every configurable model. **Prerequisite for ENH-11** (SQ8's shortlist assumes unit vectors) | 2-3 h |
| ENH-13 | mmap-able brute sidecar (raw `.npy` + sorted-id array for O(log n) membership) | 614 MB → ~0 peak at open; `update()` O(N)→O(log n) | 6-10 h |
| ENH-14 | `fts5vocab` idf — free, no dependency, no write amplification | The only honest **label-free** signal available for leg weighting | 6-8 h |
| ENH-15 | Fix `zip(rankings, weights)` silently dropping legs | 15 min. Prerequisite for any third fusion leg | 15 min |
| ENH-16 | New rerankers the fusion *cannot* express: `term_coverage` (idf-weighted), **`mmr`**, `graph_prior` | All CPU-only; **none is a cross-encoder** | 10-14 h |
| ENH-17 | Direct unit tests for `fusion`/`scoring` | Mutation 33% → ~97% | 3-4 h |

### Why `mmr` is the best new reranker (and it is not a CE)
The project **already treats cosine 0.92 as "the same fact"** at consolidation
(`consolidate.py:23`). Agent recall applies **no** redundancy control at all — and because
`supersede`/`link` are unreachable from MCP, memory only **accretes**. So recall returns
near-duplicate facts that waste context. MMR removes redundancy, which is a property RRF
structurally cannot see. That distinction is the whole argument, and it is testable: assert
MMR preserves top-1 on a no-duplicate fixture, and that it drops pairwise similarity on a
duplicate-heavy one. **If it cannot preserve top-1, it is wrong and should not ship.**

### Rejected, with reasons (the valuable part)
| Rejected | Why |
|---|---|
| **Late-interaction / ColBERT** | Index size: one vector per token = **5.1 GB vs 154 MB at 100k** (34× growth). Breaks the RAM gate by 17×. CPU MaxSim lands in the same 100-200 ms range as SQ8-at-100k. **DEFERRED with the arithmetic recorded** so nobody re-litigates it from vibes. |
| **HyDE on the recall path** | Fails three project constraints at once: `consolidate` uses `timeout=60` against 25 ms p50 gates; it adds a **second credential egress** on the hottest path; and a recall that needs a network round-trip is not local-first. Ship it offline as `expand --dry-run` so an operator can measure it. |
| CombSUM | RRF as k→0. Not a new lever. |
| Rank-aware RRF normalization | That is what `k=60` already is. |
| A default CE reranker | The paper refutes the outcome **and** the cost spread is 40×. |
| Automatic per-query fusion weights | With no labels the only self-supervised objective (leg agreement) is maximized by `dense_weight == sparse_weight` — a **degenerate optimum**, and precisely the config that produced the 0.968→0.416 collapse. Ship presets; keep the knob operator-facing. |

---

## §3 · Interface — `server.py` / `cli.py` / `doctor.py` / MCP (agent B3)

| Enhancement | Why | Effort |
|---|---|---|
| **ARCH-08 structured logging** with a redaction layer | Zero logging exists. P1 prerequisite — SEC-05 and SEC-04 are undebuggable in production without attribution | 1-2 d |
| `OmniDBError` base + complete contiguous code table + structured MCP errors | 5 exception classes with **no common ancestor**; the code is baked into `str(exc)`, forcing bracket-parsing. `uat.md:114-122` makes bracket-parsing an **acceptance criterion** | 0.5-1 d |
| Expose `supersede`/`link` + a real `correct_memory` | The vault's flagship differentiator — "memory that updates instead of accreting" — is **unreachable from MCP today** | 2-4 h |
| Gate `search` (`OMNIDB_ALLOW_SEARCH=0`) + add `recent`/`query` | The 6 LOC switch is the single highest-value security lever for a hosted deployment | 4-6 h |
| MCP resources for schema/config | The vault deferred this to dodge churn; that call is now stale — the tool surface has been stable for 2 releases | 3-5 h |
| `metrics` in Prometheus exposition (no listener) | Turns a snapshot into a trend. `stats()["agents"]` is a tenancy leak — **omit the agent dimension** | 4-6 h |
| `recall` `detail="compact"` + **surface `origin`/`confidence`** | Provenance is stored and returned inside `metadata` but **never consumed** — a security gap, not just UX | 2-4 h |
| Machine-readable CLI contract + exit-code table | 22 leaf commands, 11 `json.dumps` sites, no published exit codes | 1-2 d |
| `doctor` remediation + `severity` + `--brief` | Today `_optional_dep` reports a **missing module as `ok: True`** | 4-6 h |
| Generated config reference | 28 fields, ~10 documented | 6-10 h |
| Shell completions from the live parser | Structurally cannot drift | 3-4 h |

### The three-way call on `search` — do **not** remove it
`store.schema()` exists **solely** to document the `search` surface (its docstring says so);
disaster recovery is ad-hoc SQL; and the authorizer works. Removing a working, documented,
tested safety valve in favour of "nicer" tools is a net loss. **Gate it, bound it, and make it
unnecessary** — that is the honest version.

---

## §4 · Operations & portability (agent B4)

### The finding that changes the backup design
B4 found a scenario the existing `index_dirty=1` mitigation does **not** cover. The sidecar is
copied **after** the DB snapshot, while the server keeps writing:

| t | Event |
|---|---|
| T0 | backup completes; snapshot has memory 5 = `"A"` |
| T1 | `update_memory(5,"B")` commits → `pending_reembed=[5]`, `index_dirty=1` |
| T2 | debounced persist writes the sidecar holding the **`"B"`** vector, clears the markers |
| T3 | backup copies **that** sidecar into the snapshot |

Restored home: rows say `"A"`, sidecar holds the `"B"` vector, `pending_reembed=[]`. Recovery
sees `index_dirty=1` → reconcile → **membership matches, revision set empty → no re-embed.**
The home opens clean, reports healthy, and every dense query for memory 5 is scored against
text the store no longer contains. **Silent, permanent retrieval corruption** — and
`test_portability.py:36-37` only simulates a *missing* sidecar, never a *newer-generation* one.

| Enhancement | Effort |
|---|---|
| **Generation-stamped sidecars** — a monotonic int in `meta` and inside the artifact, so "same cut" becomes checkable rather than assumed | 6-10 h |
| DB-only snapshot as the default, with a machine-readable restore contract | 4-6 h |
| **`restore` command with `--dry-run`** — DR is 5 manual prose steps; `PRAGMA integrity_check` and `foreign_key_check` run **nowhere** in the product | 8-12 h |
| `VectorIndexGeneration` object — owns its files, published or discarded. Fixes BUG-04 and BUG-12 **structurally** | 12-18 h |
| Retention policy (count **and** age) + documented scheduler + **restore drill** | 6-8 h |
| Versioned JSONL export envelope with a migration path | 6-10 h |
| `--merge` import (the empty-home cliff blocks the obvious job) | 10-14 h |
| Generated config reference · per-corpus profiles · `config_version` · derived `SCHEMA_VERSION` · migration matrix + `schema --verify` · extend `validation.py` to be the single write-path policy | 8-18 h each |

### The one that gives the most for the least
**`restore --dry-run` is not "the obvious restore command."** Its real payload is that
`PRAGMA integrity_check` and `foreign_key_check` become reachable **for the first time in a
shipped path**. A dangling `superseded_by` (the FK has no cascade) currently hides a row from
every live-set query forever, and **no tool in the box reports it.**

---

## §5 · Memory layer (agent B5)

### The consolidation accounting bug, and the structural fix
Three defects share one root cause: the code tracks "which episodes were *presented to the
LLM*" and then uses it as "which episodes may be *retired*."
1. Truncation discards facts but keeps the retirement (`raw_facts[:500]` runs *after* the
   consumed-ids list was fully extended).
2. A chunk that yields nothing durable still retires its episodes.
3. The offline stub retires by **list position** (`episodes[:len(facts)]`).

`ConsolidationOutcome` fixes all three by making retirement **derived** with a stated rule: a
fact that was *stored* or *deduped* retires its sources; one that was *dropped* or *truncated*
does not — because it is knowledge that now lives nowhere.

### Also
| Item | Why | Effort |
|---|---|---|
| Offline stub that only claims distillations the validator accepts | Today a 501-2000-char episode is **silently destroyed** (sanitized to 2000, rejected at 500, retired anyway) | 3-4 h |
| Merge policy default to **annotate**, not overwrite | Today a model's paraphrase **silently replaces a user's own words**, and the provenance trail is destroyed not added | 3-4 h |
| Cross-chunk agreement | The only **model-independent** quality signal — two independent chunks producing the same fact is evidence it is entailed | 3-4 h |
| Token-accurate budgeting | The shipped llama.cpp example **cannot run a chunk** (default 4096 ctx, no `--ctx-size`), and the report says only "OMN-013" | 4-6 h |
| A scheduler | The Glossary calls consolidation a "**Scheduled** background process". **No scheduler exists.** | 3-4 h |
| `--emit` / `--apply` human review | The only control that inspects unreviewed model output itself, rather than a proxy | 6-8 h |
| A real extraction contract with per-fact attribution | 5 lines today, no schema, no "return [] is correct", and provenance is chunk-shaped (10 ids for a fact from one) | 4-6 h |
| Anti-injection delimiters — **labelled mitigation, not a control** | A per-run nonce defeats a *static* payload; it does not defeat a persuaded model. The real controls are: never consolidate, human review, the recall-side trust signal, a non-zero confidence floor | 2-3 h |
| LLM circuit breaker + error classification | A dead endpoint costs 60 s × **every** chunk — up to 50 minutes — and the documented failure already shows `took_ms: 60001` | 6-8 h |
| A `keys` table for LangGraph | LIKE-on-JSON **cannot find keys containing `"` or `\`** and full-scans on every get/put/delete | 6-8 h |
| Complete LangGraph + an adapter contract test | The current `search` signature **already breaks** the real `BaseStore.search` (filter dict passed positionally) | 1-2 d |

### The memory-layer items in dependency order
1. **A `keys` table for LangGraph** (6-8 h) — unblocks everything else, because the adapter
   currently full-scans on every `get`/`put`/`delete`.
2. **Complete LangGraph + an adapter contract test** (1-2 d) — `list_namespaces`, `batch`,
   `aput`/`aget`/`asearch`/`adelete`, and the `search` signature fix.
3. **`ConsolidationOutcome`** (4-6 h) — the single change that fixes BUG-A1-02, the truncation
   loss, and the offline stub's positional retirement. **Highest leverage in this section.**
4. **LLM circuit breaker + error classification** (6-8 h) — a dead endpoint currently costs
   60 s × every chunk.
5. **A scheduler** (3-4 h) — the Glossary calls consolidation a "**Scheduled** background
   process". No scheduler exists.
6. **`--emit` / `--apply` human review** (6-8 h) — the only control that inspects unreviewed
   model output itself.
7. **Token-accurate budgeting** (4-6 h) — the shipped llama.cpp example cannot run a chunk.

---

## §6 · Sequencing — what must land together

Several of these are **not independent**, and shipping them apart breaks something that works
today.

| Bundle | Items | Why they must ship together |
|---|---|---|
| **B1** | `errors.py` + `predicates.py` + `ranking.py` | `predicates.py` owns the batching helper that fixes BUG-A1-08. Splitting it from `errors.py` means two modules both claiming the OMN-* table. |
| **B2** | `ef_search` 64→128 **+** the `baseline.py` gate fix (PERF-A2-10) | The corrected gate reads `min` instead of `max` and **will be red on the current tree** (768-d/100k = 0.359). Landing it without the config change leaves a permanently red pipeline. |
| **B3** | `tags` table migration **+** the `SCHEMA_VERSION` bump | `store.py:188` hardcodes `set_meta(...,"3")`; `db.py:6` = 3; **6 test modules** pin `== SCHEMA_VERSION` and one tests `SCHEMA_VERSION+1` rejection. They must land in one commit. |
| **B4** | `llm_base_url` validation **+** MCP sampling (I-12) | Validating the URL is the interim control; sampling is the structural removal. Doing sampling first makes the validation unnecessary. |
| **B5** | `index_keeper.py` **+** the `store.py` split | Splitting `store.py` without `index_keeper` forks the ARCH-02 crash window across two files. **This is the one hard ordering constraint in the whole plan.** |
| **B6** | `recall` result-shape change **+** `uat.md` update | The result dict is built twice (`recall.py:122-133` and `:190-201`); changing one and not the other is a silent inconsistency. |

### The one thing to not do
**Do not start the `store.py` decomposition without `index_keeper.py` existing first.** A5
already documented this; F3 independently reached the same conclusion. Every other item in this
document is safely reorderable. This one is not.
