# 3 · Advanced Features

Source: **C1** (memory arch) · **C2** (retrieval innovation) · **C3** (agent UX) · **C4** (competitive) · **C5** (resilience).

---

## §1 · Memory architecture (agent C1)

| ID | Feature | Effort | Value |
|---|---|---|---|
| A1 | **Bitemporal facts** — `valid_from_ms`/`valid_to_ms` + `as_of(ms)` recall + contradiction detector | 2-3 d | High |
| A2 | **Evidence-accumulating confidence** (Beta/log-odds) | 1-1.5 d | **Very high** |
| A3 | **Automatic skill induction** from episodic traces | 3-5 d | **Very high (differentiator)** |
| A4 | Scope tree (path-prefix column) replacing flat `agent_id` | 1.5-2 d | High |
| A5 | Counterfactual / rejected-alternative memory | 1-1.5 d | High |
| A6 | **Self-verifying memory** — evidence span via `recall --explain` | 1-1.5 d | High |
| A7 | Principled forgetting — expiry + `derived_from` lineage + cascade purge | 2-3 d | Med-High |
| A8 | Ambient capture with an accept-reject queue | 2-3 d | Med |

**A1** — `_DEDUP_COSINE=0.92` (`consolidate.py:23`) merges *paraphrases* but lets "deploy is
Tuesday" and "…Wednesday" land side by side, both live. Only `superseded_by` exists and it is
unreachable from the product surface.

**A2** — turns the LLM self-reported `confidence` from a stored assertion into a **prior**.
Model each fact as Beta(1+s, 1+f): the LLM sets `a0 = 1 + 4c`, `b0 = 1 + 4(1-c)` and **cannot
by itself write a confident fact**; corroboration from >=2 distinct episodes moves the posterior;
a contradiction increments `f` on the loser. Time enters **only** as a discount
`1/(1 + dt/tau)`, tau=30 d, on how much a new observation moves it — *not* as a score multiplier.
This is the defensible answer to the vault rejected "fixed decay religion".

**A3 (the differentiator)** — runs at the end of `run_consolidation` over already-materialised
episodes. **Pure Python, zero new model spend.** (1) Tokenize each episode tool/verb sequence.
(2) Count n-grams (n=3..6) over normalised verbs, arg *shapes* not values, plus a gapped-repeat
variant. (3) Require support >=3 across **>=2 distinct episodes** — distinctness stops one verbose
episode minting a skill. (4) Synthesise the regex and **hard reject** >256 chars, any nested
quantifier, anything failing to compile: inferred patterns are **attacker-influenced**, so the
lint is mandatory. (5) Write with `proposed=True`; recall skips it. (6) `promote_skill()`.
(7) Track `precision = hits/(hits+rejects)`, auto-retire below 0.6.
**Acceptance:** replay 200 episodes, measure precision@k against human labels. Below ~0.8 the
feature is worthless — **and that number must be published.**

**A6** — converts the trust model from "the DB said so" to "checkable in one call". The
structural answer to prompt-injection -> memory poisoning.

**A8** — every false positive is a durable, globally-retrievable, **attacker-plantable** fact.
Must go to a bounded proposal queue with a rate ceiling, never straight to `remember`.

**Recommended AGAINST:** (a) automatic TTL expiry of facts — confidence is a self-referential
posterior, not ground truth, and silent auto-deletion is unrecoverable; (b) free-text ambient
capture that auto-commits; (c) an LLM judge that auto-supersedes on contradiction — a
hallucinated "Wednesday" would silently retire a correct fact with no audit.

---

## §2 · Retrieval innovation (agent C2)

The paper's own mechanism line governs this section: **a reranker's value scales with
first-stage weakness, not window size.** The innovation surface is first-stage recall + fusion.

| ID | Feature | Effort | Expected delta |
|---|---|---|---|
| R2 | **Embedder-aware leg confidence** — gate `sparse_weight` on the stored `meta.embedder` | 1-2 h | offline default recall@5 **0.416 → ~0.97** |
| R3 | CombMNZ fusion (per-document leg agreement) | 3-5 h | +0.01-0.03 nfcorpus/fiqa |
| R4 | idf-gated double lexical leg (full-OR + top-m-idf) | 6-10 h | recall@100 +0.02-0.05 |
| R6 | Offline query expansion (trigram tokenizer + shipped synonym table) | 4-8 h | +0.01 on typo/abbrev queries |
| R7 | Two-stage exact f32 rerank, **scale-gated** | 6-12 h | .881/.896 @100k; **.967 @0.543 ms @1k** |
| R8 | MMR on the agent path only | 4-6 h | no nDCG change (not a qrels objective) |
| R1 | ~~Raise `k_fetch`~~ | — | **0 — REFUTED**, see doc 1 §0.4 |
| R9 | ~~Graph-aware retrieval~~ | — | **0, measured** |
| R10 | ~~Query-intent routing~~ | — | **0, unmeasurable** |

**R2 is the one high-value finding.** The measured failure (`comparative-results.md`: hybrid
recall@5 **0.416** vs dense-alone **0.968**) is *specifically* `hashing-v1` letting the lexical
leg outvote correct dense evidence — and the embedder identity is **already stored, already
enforced, already in `stats()`**. Repairable in 1-2 h with no new retrieval machinery.

**R9 rejected because the graph is empty.** `link()` has **zero callers**; `related()` is
unreachable from MCP. `edges_dst` already indexes the reverse direction, so this is not an
index-cost problem — it is a *data* problem. **Fix ARCH-11 first**; PPR is then 1 day.

**R10 rejected because there is no signal.** `query_log` defaults `False`, so no offline
training data exists. A classifier would be 3 hand-written regexes re-deriving the documented
0.7 ↔ ≤0.3 spread.

**Recommended AGAINST:** any CE reranker or ML-weighted sparse retrieval (needs a trained model,
violates local-first, and the paper refutes the outcome) · SQ8 as a *default* (2.4-5.2× over the
p50 gate) · BM25F (single-column external-content FTS makes it a storage rebuild for ±0.01).

---

## §3 · Agent-native experience (agent C3)

| ID | Feature | Effort | Agent benefit |
|---|---|---|---|
| C3-2 | New `recall` contract — an envelope, not a bare list | 1-1.5 d | ~35-40 tok/row of unusable floats removed; adds "was I truncated?" |
| C3-3 | `recall_batch` — N queries → 1 round trip | 1 d | 3-5 sequential recalls → 1; the saving is **LLM turns**, not DB time |
| C3-4 | Declarative `query` tool (structured filters, not SQL) | 1.5-2 d | The model never synthesizes SQL; also deletes the `SQLITE_FUNCTION` surface |
| C3-5 | Content-hash idempotency on `remember` | 1.5-2 d | A retry currently creates a duplicate that then **votes in fusion** |
| C3-6 | `unforget` | 4-6 h | Reverses a wrong `forget`; the row already survives (soft delete) |
| C3-7 | **Typed provenance** | 2-3 d | Lets an agent **cite and distrust** machine-written facts |
| C3-8 | Handshake instructions + actionable errors | 1-1.5 d | Kills bracket-parsing; a `remedy` field fixes most retries |
| C3-1 | ~~Tool-budget / lazy loading~~ | — | **Rejected** — it is a *client* capability, not a server one |

**The envelope.** `list[dict]` cannot express truncation, so an envelope is *required*, not nicer:
```json
{"hits":[{"id":42,"kind":"semantic","content":"...","created_at":"...",
          "relevance":0.87,"match":"hybrid",
          "provenance":{"origin":"user","confidence":null,"episodes":[]}}],
 "truncated": true, "total_candidates": 137}
```
`relevance` is min-max normalized **within the result set**; `match` tells the agent *why* it
surfaced — the one genuinely actionable bit that `rrf=0.0` was encoding.

**C3-7 the honest boundary.** `origin` ∈ {user, agent, consolidation, import, unknown},
`confidence` ∈ [0,1] or **null (unasserted, not 0)**, plus a derived
`distrustable = origin ∈ {consolidation, import} or confidence < 0.6`. **But `metadata` remains
writable, so a determined caller can still forge provenance inside it.** This is a trust signal
for the model's own reasoning, **not an enforcement boundary, and must not be sold as one.**

**Rejected:** lazy tool loading via resources/prompts (~1k tokens *once*, and it is a client
capability) · streaming/partial results (stdio JSON-RPC has no consumer; at k≤10 a recall is
<1.5 KB) · a session scratch tier (an LLM already has a context window; an ephemeral store
inside a *memory* server is conceptually off-key).

---

## §4 · Competitive positioning (agent C4)

### The one sentence the evidence supports
> **"SQLite with retrieval superpowers."**

The vault's own honest framing, and the data backs it: hybrid beats bm25s/pyserini on **all 4**
corpora, loses to dense-only only on scidocs. The honest position is also the defensible one —
claiming to be a database is a fight against qdrant/pgvector that a 4,195-line local binary
loses on brand recognition alone.

### Where the moats actually are
| Asset | Why it is a moat |
|---|---|
| **The published negative result** | A controlled study with an open eval harness and a clean "this does not help" finding. No competitor ships that. It converts research credibility into a trust position. |
| **"Never phones home"** | In 2026, for regulated deployments, a memory server with no egress is a **procurement** advantage — and it can be made *verifiable* (egress-audit mode, air-gap story, data-flow diagram). |
| **One home directory** | Most agent stacks need a memory SDK + a vector DB + an MCP server. Rows-are-truth means the index is reconstructible — most vector DBs **lose data** if the index is lost. Make that a `doctor` check and it becomes a provable guarantee. |
| **Reproducible quality** | A 4-corpus harness with human qrels and proper statistics. `omnidb-server bench` on the user's own corpus makes the corpus-dependent-weight finding operational. |

### What NOT to build
A database · a distributed system · a CE reranker · **multi-tenancy before the owner decides** ·
a hosted service · a vector engine in Go/Rust · proprietary extensions. Each is either a fight
you lose or a commitment the 1,200-line ambition already lost once.

---

## §5 · Resilience & scale (agent C5)

### R6 — the best effort/benefit ratio in the entire audit
**Online index rebuild, 0.5-1 day.** `rebuild_index` already ends in `self.index = fresh` —
**the pointer swap is atomic and free.** The entire cost is that the build loop runs under
`@_serialized`. Build the fresh index *outside* the lock, then take the lock for ~1 assignment.
Recall never sees a partial index, and it **fixes BUG-12 (unlink-before-construct) for free**
because you no longer unlink at all.

### The rest
| ID | Feature | Addresses | Effort |
|---|---|---|---|
| R1 | Chaos harness at 100k + an fsync'd ack log | "zero data loss is asserted, never measured" | 3-5 d |
| R2 | Append-only vector sidecar WAL (LSM) | PERF-06's O(N) persist | 5-8 d |
| R3 | `recall` on a WAL reader snapshot | `recall` is a write | 2-3 d |
| R5 | Priority queue + admission control | SEC-05's unbounded in-lock work | 2-3 d |
| R7 | A real SLO/gate framework | the soak gate currently FAILS | 3-4 d |
| R8 | Periodic read-only integrity auditor | the 5 markers cannot see content drift | 3-4 d |
| R9 | Canonical/seeded export | no reproducible home for tests | 1-2 d |
| R10 | Degradation matrix + runtime guards | 5 unhandled failure modes | 2-3 d |
| R11 | SQ8 + an `ef` ladder by live count | recall@10 0.395-0.519 @100k | 1-2 w |
| R4 | ~~Leader election / multi-writer~~ | — | **Rejected** |

### R4 rejected, and the reasoning matters
Leader election buys a local-first tool **nothing**: it cannot add writers (single-writer is the
point), it adds a lease file the OS lock already covers, and split-brain on `msvcrt` byte-range
locks is not a risk worth owning. **Document the rejection in the ADR** so nobody re-proposes it.

### R1 — the claim to make, stated narrowly
> *"For SIGKILL at any point in a 100k-row mixed workload, the reopened home passes
> `PRAGMA integrity_check`, contains every row whose `remember` returned before the kill, and
> reconciles in ≤ T s."* — RPO = 0 rows, RTO measured.

**Do NOT claim power-loss safety.** `db.py:93` sets `synchronous=NORMAL` in WAL, which per
SQLite's guarantee cannot corrupt but **can roll back unfsynced commits**. Power-loss RPO is
*argued*, not measured, and must be labelled as such.

### Failure modes currently unhandled (verified)
1. **Disk full mid-flush** — `fsync`/`save_index` raises inside `persist()` → propagates into
   the user's `remember()`. `index_dirty` correctly stays 1, but a full disk becomes a
   write-path exception. `doctor` warns at 64 MB and **nothing consults it**.
2. **Read-only FS / home unwritable at runtime** — `set_meta` raises uncaught.
3. **`busy_timeout` exceeded** — caught in `checkpoint()` only, **not on the write path**.
4. **Clock skew** — `recall.py:116` computes `age_ms = now_ms - created_ms` and feeds it to
   `exp(-age/τ)`; a backwards wall clock yields negative age and recency dominating the score.
   No clamp exists.
5. **`doctor` false-clean** — `_recovery_check` reports only 4 of the 5 markers; the 5th
   (`active_count != live_count`) is evaluated **only at open**. A sidecar deleted after a clean
   close leaves all 4 clear → `doctor` says "clean".

**Already handled (do not re-report):** missing hnswlib · missing FTS5 (hard refusal) · corrupt
or foreign sidecar with a stderr reason + brute fallback · backup snapshot self-healing via
forced `index_dirty=1` · unsupported schema version refused before the WAL pragma flips.

---

## §6 · The three highest-leverage features, ranked

C1–C5 produced ~30 candidate features. Ranked by leverage against the measured state:

### 1. R2 — embedder-aware leg confidence (1–2 h, and it is a repair, not a feature)
The measured failure in `comparative-results.md` is *specifically* `hashing-v1` letting the
lexical leg outvote correct dense evidence: hybrid recall@5 **0.416** vs dense-alone **0.968**.
The embedder identity is **already stored, already enforced, already surfaced in `stats()`**.
Gate `sparse_weight` on it and the default offline configuration recovers to **~0.97**.
**No new machinery, no new dependency, no new table.** Nothing else in this document has that
ratio.

### 2. A3 — automatic skill induction (3–5 d, the only true differentiator)
Pure Python, zero new model spend, and nobody in Mem0/Zep/Graphiti mines behavioural traces.
The trigger-evaluation path already exists and is already budgeted. **But it must ship with a
published precision@k** — below ~0.8 the feature is worse than nothing, and an inferred
`trigger_patterns` is an attacker-influenced regex reaching the SEC-05 surface.

### 3. R6 — online index rebuild (0.5–1 d, best effort/benefit in the entire audit)
`rebuild_index` already ends in `self.index = fresh`. The pointer swap is atomic and free; the
only cost is that the build loop runs under `@_serialized`. Move the build outside the lock and
recall never sees a partial index — **and BUG-A1-12 is fixed for free**, because you no longer
unlink before constructing.

**Deliberately not ranked high despite being genuinely valuable:** the `recall` envelope
(C3-2, 1–1.5 d) and typed provenance (C3-7, 2–3 d). Both improve agent trust, but neither
closes a measured regression, and both change a wire contract.

---

## §7 · Sequencing — the two hard constraints

| # | Constraint | Consequence |
|---|---|---|
| **1** | **A3 (skill induction) must not ship before the nested-quantifier lint (SEC-03 fix option 1)** | Induced patterns are **attacker-influenced** by construction. Generating them before the lint exists manufactures a new ReDoS surface. The lint is 1 h; do it first. |
| **2** | **A1 (bitemporal) must not ship before the scope tree (A4)** | Bitemporality without a scope column means every live-set query doubles its logic, and `valid_from`/`valid_to` must **never** be wired into `_SELECTIVE` or the whole read path breaks for a consumer that does not exist yet. |

Everything else in C1–C5 is freely reorderable. These two are not.
