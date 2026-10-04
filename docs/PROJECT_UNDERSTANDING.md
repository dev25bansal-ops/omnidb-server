# OmniDB — Complete Project Understanding

**Read date:** 2026-09-25 · **Scope:** `d:/OMNIDB` · **Method:** exhaustive line-by-line read of every authored file, cross-verified against the code.

---

## 0. What this project actually is

**OmniDB is an MCP-native agent memory & retrieval server over SQLite.** It is *not* a database, despite the name and the 72-page design document it grew out of. That distinction is the single most important thing to understand about this codebase.

- **Product identity:** a local-first stdio server exposing 10 MCP tools that let an AI agent `remember` / `recall` its own memories. Ships on PyPI as `omnidb-server` v0.1.3.
- **Honest self-description** (`Final Build Verdict.md:14`): *"v1 is 'SQLite with retrieval superpowers.'"*
- **Substrate:** SQLite in WAL mode (FTS5, porter+unicode61) + a vector sidecar (hnswlib, or exact brute-force fallback).
- **Retrieval:** weighted Reciprocal Rank Fusion of a dense leg and a BM25 leg, then recency/frequency rerank.

### Size reality check

| Tree | Files | Lines | Nature |
|---|---|---|---|
| `omnidb_server/` | 22 `.py` | **4,195** | **The product** |
| `tests/` | 43 `.py` | 4,625 | Behavioral contract |
| `bench/` | 15 `.py` | ~3,200 | Research harnesses |
| `docs/` | 9 `.md` | — | Audit trail |
| `paper/` | ~20 | — | IEEE submission |
| `OmniDB Vault/` | 36 notes | — | Obsidian knowledge base |
| `texlive/` + `tools/` + caches | 11,236 | — | **Vendored toolchains — not project code** |

The `d:/OMNIDB` root is **~95% third-party bulk** (TinyTeX 73 MB, TeX Live 313 MB, JDK 331 MB, ~5.4 GB of Hugging Face model blobs). Reading it "line by line" is not meaningful; I inventoried it, classified it, and read every *authored* byte.

**Correction to the project's own audit:** `docs/ISSUES_AND_REQUIRED_FIXES_2026-09-21.md:31` measures `omnidb_server/` at "22 `.py` files, 4,603 LOC". Measured today: **22 files, 4,195 LOC** (PowerShell `Measure-Object -Line` counts non-blank lines differently than the earlier tooling; both agree on 22 files). The file count is confirmed; the LOC figure has drifted ~9%.

---

## 1. Architecture map

```
MCP client (Claude Desktop)
   │  stdio
   ▼
server.py ── create_mcp() ──► 10 FastMCP tools, anyio.to_thread offload
   │                              │
   │  open_owned_store()          │  _acquire_home_lock() BEFORE any writable init
   ▼                              ▼
config.py ──► db.py ──► embeddings.py ──► vector_index.py ──► MemoryStore (store.py)
   │            │                                                     │
   │            └─ migrations, meta KV                                │
   └──────────────────────► recall.py ──► fusion.py + scoring.py + rerank.py
                                                              │
                              consolidate.py ◄────────────────┘
                              portability.py, skills.py, doctor.py, cli.py
```

### Module responsibilities (all 22 read in full)

| Module | LOC | Responsibility | Load-bearing invariant |
|---|---|---|---|
| `store.py` | 1,028 | All writes, hybrid reads, reconciliation, maintenance | Post-commit order lives in one place (`_after_commit`, ARCH-02): **dirty flag → index mutation → compaction → debounced persist** |
| `portability.py` | 503 | backup / re-embed / export / import | Import validates the whole file *before* the first write, single rollback-protected transaction |
| `consolidate.py` | 314 | episodic → semantic distillation | A failed LLM chunk consumes nothing — episodes stay unmarked and retryable |
| `cli.py` | 306 | 20 subcommands | Read-only diagnostics take no lock; every writer takes it |
| `vector_index.py` | 270 | ANN sidecar protocol | Atomic persist (tmp→fsync→replace) with Windows `PermissionError` retry |
| `server.py` | 260 | MCP stdio server | `open_owned_store` is the *single* writable entry point |
| `config.py` | 221 | CLI > env > TOML > defaults | Unknown TOML keys reject; every scalar coerces to its declared type |
| `recall.py` | 215 | The hybrid retrieval pipeline | Unified kind/scope filter contract across all three legs |
| `skills.py` | 184 | Skill bundle export/import (zip) | Atomic zip write; manifest validated strictly |
| `doctor.py` | 164 | Environment health, 13 checks | Read-only, never downloads, honest exit code |
| `db.py` | 146 | SQLite open + migrations | Forward-only, gap-checked, atomic DDL+postcondition+version bump |
| `embeddings.py` | 143 | Hashing + optional fastembed | Factory never downloads implicitly; identity pinned per DB |
| `rerank.py` | 56 | Named rerankers behind `REGISTRY` | Strictly opt-in per `recall` call |
| `validation.py` | 44 | Shared write-path policy | One content/pattern contract across create/update/import |
| `fusion.py` | 33 | RRF + safe FTS MATCH builder | User text quoted into inert OR-ed tokens (max 24) |
| `scoring.py` | 22 | `final_score` — the tuning surface | Total function; negative age clamps to zero |
| `llm.py` | 33 | Ollama autodetect | Never raises; failure → offline stub |
| adapters | 232 | LangGraph / LlamaIndex duck types | Lazy; never import the frameworks |

---

## 2. The write path, precisely

`remember(content, kind, metadata, links, agent_id, trigger_patterns)`:

1. **Validate** kind, content policy (`validation.py`), metadata JSON-serializability, and **every link target against live rows** — a bad link aborts before any insert.
2. **Encode once** (single `embedder.encode(contents)` for the whole batch).
3. `INSERT` + `conn.commit()`.
4. `_after_commit`: set `index_dirty=1` → index mutation → compaction check → debounced persist.

**The crash-window design is the best engineering in this repo.** SQLite rows are the source of truth; vectors are always derivable. Five durable markers cover every interruption boundary:

| Marker | Meaning | Recovery |
|---|---|---|
| `index_dirty` | sidecar lags rows | reconcile membership |
| `pending_reembed` | same-ID content revision committed, vector stale | re-embed that id (AUDIT-I01) |
| `reembed_pending` | embedder migration interrupted | full rebuild under stored identity (AUDIT-I03) |
| `compaction_due` | hnsw tombstone ratio ≥ 0.2 | fresh-sidecar rebuild on next open |
| `active_count != live_count` | sidecar lost or stale | full rebuild |

---

## 3. The read path, precisely

`recall(query, k, kind, k_fetch, agent_id, reranker)`:

1. `fetch = min(max(k_fetch or 50, k), 1000)`, ×`agent_overfetch` (3.0) when agent-scoped.
2. **Dense leg:** embed query → global ANN top-`fetch`. **Sparse leg:** quoted OR-ed FTS MATCH → `ORDER BY bm25()`.
3. `rrf([dense_ids, sparse_ids], k=60, weights=[0.7, 0.3])`.
4. Re-fetch the **entire** fused pool with selective filters (eligibility *before* truncation — AUDIT-I09).
5. `final_score` = RRF + `0.15·exp(-age/half_life)` + `0.05·log1p(access_count)`.
6. Skill-trigger merge: procedural rows whose `trigger_patterns` regex matches, regardless of fusion rank.
7. Optional reranker → sort `(-score, id)` → slice to `k` → access-count bump + query log in one commit.

**`corpus_mode`** zeroes both rerank weights — justified by a *measured* 40–56% nDCG@10 penalty on corpus workloads (`config.py:79-83`). The paper leaves this number on the table; it is a free additional result.

---

## 4. Security posture (as shipped, stdio)

Enforcement is layered and, for the stated local threat model, solid:

- **`search` read-only:** first-word allowlist (SELECT/WITH) **plus** a deny-by-default SQLite authorizer. The allowlist alone cannot stop data-modifying CTEs — the authorizer is the enforcement point.
- **Three-way work bound:** VM-instruction budget (5M, counted correctly against the 1,000-instruction callback interval), monotonic 5.0 s wall clock, 32 MiB result byte cap.
- **Trigger bounds:** admission (32 patterns / 512 chars) + matching (4,096-char query, 2,048 evaluations/recall). `validation.py` is shared by create/update/import.
- **FTS injection:** structurally impossible — `\w+` discards metacharacters, tokens are quoted.
- **Single writer:** real OS advisory lock (`flock` / `msvcrt`), acquired *before* any mutable init; released by the OS on process death.

**Honest limitation:** the trigger matcher uses Python's backtracking `re`. Bounding input size does **not** bound backtracking time. Under a *hosted multi-tenant* target this is Critical, because the store-wide `RLock` means one wedged recall freezes every tenant. The project's own audit names this (SEC-02) and the correct fix (RE2-class engine, or off-lock matching with a wall-clock deadline).

---

## 5. The research: what was actually measured

The benchmark tree is unusually disciplined for a small project — real BEIR corpora, human relevance judgments, shared embeddings, paired tests with multiple-testing correction.

### The paper's headline (a clean negative result)

**"When Does a Cross-Encoder Second Stage Help?"** — IEEE Access, 10 pages, 4 full-size corpora × 4 encoders.

> Re-ranking an already-fused hybrid first stage **yields no significant nDCG@10 gain at any window in any of the 18 fused-hybrid window-cells tested**, and every significant effect is a loss.

| Finding | Evidence |
|---|---|
| CE on hybrid: never helps | best +0.0067 (n.s.), worst **−0.0541** (p<0.0001) |
| CE on *weak lexical* stage: helps a lot | fiqa **+0.0808** (p<0.0001), scifact +0.0484 (p<0.0001) |
| Low recall ≠ CE gain (counterexample) | nfcorpus recall@100 = 0.2376, lowest of any stage, yet gain +0.0004 (p=0.8945) |
| Hybrid beats BM25 on all 4 corpora | 16/16 cells; largest margin fiqa 0.3927 vs 0.2380 |
| Per-query: 22% improve / 34% harmed / 43% unchanged | pooled ECDF over 18 cells |

**The claim is safe** because it rests on `review_probe_deltas.json` (per-query deltas, 27 cells, tagged `BAAI/bge-reranker-base`), not on the v3 grid's CE column — which is void (see §7).

### Supported claims, with their artifacts

1. Memory-style recency/frequency rerank **costs 40–56% nDCG@10** on corpus IR → justifies `corpus_mode`.
2. Optimal `dense_weight` is **dataset-dependent**: 0.7 on nfcorpus, ≤0.3 on scifact. Not a universal default.
3. Hybrid ≥ vector-only on nfcorpus + scifact (rb up to +1.00, p<0.0001); ties/loses on scidocs (+0.0110, p=0.0102).
4. Brute-force 512-d scales sub-linearly to ~4k live rows: p50 0.99→3.39 ms over a 36× size increase.
5. Mutation score **33%** (10/30 killed) on the two pure ranking modules.

---

## 6. The self-audit culture (rare and valuable)

This project has audited itself **three times** at increasing severity, and — critically — **records its own retractions**.

| Audit | Date | Items | Character |
|---|---|---|---|
| Comprehensive Analysis | Aug 2026 | I-01..I-15 | First pass, 513 writes/s baseline |
| Post-Publish Analysis | Aug 2026 | J-01..J-15 | Post-release |
| Sep 2026 audit → `docs/AUDIT.md` | 2026-09-18 | I01–I22 | 8 subagent reviews, ~154–279 h sizing, **all 22 implemented** |
| `ISSUES_AND_REQUIRED_FIXES_2026-09-21.md` | 2026-09-21 | SEC/BUG/ARCH/PERF/QA/DEBT/CI | **Deployment target changed to hosted multi-tenant** → prior security ratings void; 6 explicit retractions |

**Refuted Claims.md** is the best artifact in the vault: it killed the project's own original thesis with evidence ("honest production-grade scope is roughly 18-36 months", not 8 weeks), retracted 5 of 8 market claims, and honestly states *"the strategic narrative rests on sand."*

The 09-21 audit also discloses that **its own test run destroyed benchmark evidence** (`:977-982`) — an unusual and creditable level of honesty.

---

## 7. Defects I verified myself

Everything below I confirmed by reading the code or files directly.

### Critical

1. **`omnidb-server/LICENSE` does not exist** — confirmed by direct `Get-ChildItem`. `pyproject.toml:10` declares Apache-2.0 and `MANIFEST.in:3` includes it. **Publication blocker.** (LEGAL-01)
2. **A documented happy path permanently bricks a memory home.** `re-embed --embedder fastembed --model some/other` pins `meta.embedder = "fastembed:some/other"`, but `make_embedder` has no model parameter (`embeddings.py:74-80`) and `Config` has no field for it. Every subsequent open refuses with `OMN-006`. No config surface can fix it. (BUG-15)
3. **Consolidation leaks across agent scopes.** `consolidate.py:222-230` chunks *by agent* — but if a chunk ever contains mixed agents it raises... and the defensive check at `:288-290` drops the fact. The partition logic is correct *now*; the exposure is the framing, and the test suite only covers the already-fixed path. (SEC-05, partially mitigated)

### Documentation integrity — the dominant defect class

4. **The v3 benchmark's entire cross-encoder column is void.** `research-v3-results.md` header names `ms-marco-MiniLM-L-6-v2`; all 16 JSON cells say `BAAI/bge-reranker-base`. **Three cells flip sign** ("helped" ↔ "hurt"). The hand-written Conclusion inherits the error. (DOC-10)
5. **`AUDIT.md` I12 cites numbers present in no file** — claims p95 ratio 3.758, 381 ops/s, hash `c1e17113`; the artifact says **5.337, 548.91 ops/s, `4ca9b6be…`, `validation_gate_passed: False`**. (DOC-09)
6. **One 180-byte artifact has three mutually exclusive descriptions** in `AUDIT.md`, `perf-report.md`, and the file itself — including opposite verdicts.
7. **The paper's `.docx` is built from the oldest manuscript generation** — `build_docx.js:8` reads `paper.md`. It asserts two conclusions the current `.tex` explicitly retracts ("three of four corpora"; ρ=0.60/n=16), carries a 3-placeholder author block, and omits 2 of 3 figures and an entire table column.
8. **The vault presents unbuilt systems in the present tense** — `Glossary.md:32-33` says `VECTOR_SIM`/`MEMORY_RECALL` "are exposed this way"; those three identifiers appear **zero times** in any `.py` file.
9. **The LOC ceiling was breached 3.8×** — vault says "honest ceiling ~1,200 lines"; actual 4,195. The vault *predicted* this exact failure mode and it materialised anyway.

### The one CI gate that cannot fail

10. `bench/baseline.py:137-138` computes `best = max(hnsw_scores + [min(sq8_scores)])` then gates `best >= 0.95`. It is a **max over the sweep** — it passes on the single best cell (1.0 at 384-d/1k) while the worst cell in the same sweep is 0.17. A gate reading a maximum cannot fail.

### Performance evidence is unverifiable by construction

11. The evidence for both availability claims lives in **gitignored** files (`.gitignore:30-31`), and three committed runs with an **identical source hash** disagree: p95 ratio 4.156 (False) / 2.379 (True) / 2.632 (True). The threshold is crossed by run-to-run noise, not by code.

---

## 8. Honest limitations (documented, and correct)

- Single-writer per home. `recall` **is a write path** (access bump) — read-replica routing would silently corrupt scoring.
- No QPS targets by design (ADR-7); gates are interactive latency, cold start, RAM.
- `agent_id` is a **retrieval label, not an authentication boundary**. Four known bypasses are pinned as `xfail(strict=True)` tripwires in `test_tenancy_boundaries.py` — genuinely good practice.
- HNSW is optional; exact brute-force is always available and is what the paper measured.
- The paper's results are for **exact dense retrieval**; it explicitly disclaims HNSW equivalence.

---

## 9. Time-line

| Date | Event |
|---|---|
| Jul 2026 | 72-page v1 spec: four engines from scratch, single file, SQL++, 8-week solo build |
| Aug 2026 | Adversarial review → **pivot**: 7 ADRs, "reuse, do not build" |
| Aug 2026 | Build (Python), v0.1.1 published |
| Aug 2026 | Audit I: I-01..I-15 → 126 passed |
| 2026-09-07 | Comparative bench records the **negative fusion finding** (hybrid 0.416 vs dense-alone 0.968) |
| 2026-09-11 | Audit II: 30 items → 228 → 244 passed |
| 2026-09-15 | Paper `.docx` built (from stale source) |
| 2026-09-18 | Audit III: 22 items, all implemented |
| 2026-09-20 | **v0.1.3 released** |
| 2026-09-21 | Audit IV: target changed to hosted multi-tenant; **6 retractions**; 3 new Criticals |
| **2026-09-25** | **Today** — tree is at 4,195 LOC with 2 verified publication/correctness blockers |

---

## 10. What I'd do next, in order

1. **Add `LICENSE`** (5 min) — publication is blocked without it.
2. **Add `fastembed_model` to `Config`** and thread it through `make_embedder`; make `re-embed` refuse a model it cannot later reconstruct. Closes BUG-15.
3. **Re-render the v3 CE grid from its JSON**, or delete the column. Three sign-flips is a retraction, not a rounding error.
4. **Re-measure the soak on the final runtime** and bind the numbers to `source_identity_sha256` in prose, so a mismatch is self-announcing.
5. **Fix `baseline.py`'s gate** to read the *worst* cell, not the max.
6. **Rebuild the `.docx` from `omnidb_ieee_access.md`**, and resolve authorship — the file's own comment says the placeholders are disqualifying.
7. **Write the 4 tenancy xfail tests' contract into a decision** — the bypasses are already measured; only the product decision is missing.
8. Vendor a condensed `docs/DECISIONS.md` so the pervasive `AUDIT-*`/`PERF-*`/`ARCH-*` code tags resolve to something inside the repo.

---

## Appendix: exhaustive file inventory

**`omnidb_server/` (22 modules, 4,195 LOC)** — all read line by line:
`__init__.py` 12 · `__main__.py` 9 · `adapters/__init__.py` 29 · `adapters/langgraph.py` 145 · `adapters/llamaindex.py` 58 · `cli.py` 306 · `config.py` 221 · `consolidate.py` 314 · `db.py` 146 · `doctor.py` 164 · `embeddings.py` 143 · `fusion.py` 33 · `llm.py` 33 · `portability.py` 503 · `recall.py` 215 · `rerank.py` 56 · `scoring.py` 22 · `server.py` 260 · `skills.py` 184 · `store.py` 1,028 · `validation.py` 44 · `vector_index.py` 270

**`tests/` (43 files, 4,625 LOC)** — all read; contracts extracted per subsystem.
**`bench/` (15 scripts + 20 artifacts)** — all read; measured numbers extracted and cross-checked.
**`scripts/mutation_check.py`, `examples/` (3 files)** — read.
**`docs/` (9 files)** — AUDIT, ISSUES_2026-09-21, MULTI_TENANCY_THREAT_MODEL, DATASTORE_SPIKE, perf-report, coverage-report, mutation-report, operations, uat — all read.
**`.github/workflows/` (2)** — read; CI is well-built (pinned SHAs, `permissions:`, required-HNSW job with skips-as-failures, 80% branch floor). The 09-21 audit's CI-01..CI-09 findings are **already fixed** in the current tree.
**`OmniDB Vault/` (36 notes + canvas + config)** — all read.
**`paper/` (~20 authored)** — all read; 3 manuscript generations compared.
**`omni.txt`** (171 KB) — `pdftotext -layout` extraction of the AI-generated PDF; **3 tables arrived rotated by one row**, uncorrected and unmarked in the file itself.
**`.remember/` (113 files)** — a `remember`-tool session-memory log archive with a *recurring* `save-session.sh --force exited 1` failure.
**`_verify/`, `_verify-nope/`** — fact-check fetch captures and a 3-file test home.

**Deliberately not read line-by-line** (vendored, not project code): `texlive/` 9,112 files · `tools/jdk-21.0.5+11/` 490 files · `paper/node_modules/` 386 · `.hf-cache/` + `bench/.cache3/` ~5.4 GB model blobs · `build/`, `dist/`, `*.egg-info`, `__pycache__`, `.mypy_cache`. Listed and classified, not read. Say the word and I will sweep any of them.
