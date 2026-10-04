# GROUND TRUTH BRIEF — read this before any analysis

**Purpose:** 30 analysis agents work in parallel. This file is the single source of verified fact. Anything not in here must be verified from source before asserting it.

**Project:** OmniDB / `omnidb-server` v0.1.3 at `d:/OMNIDB/omnidb-server`
**Read date:** 2026-09-25 · Python 3.14.4 / Windows / numpy 2.x / hnswlib present

---

## 0. PRODUCT IDENTITY (do not get this wrong)

OmniDB is an **MCP-native agent memory & retrieval server over SQLite**. It is NOT a database, despite the name and the 72-page design doc it grew out of. The vault's own verdict: *"v1 is 'SQLite with retrieval superpowers.'"*

The July 2026 spec planned four engines from scratch (B+tree/MVCC/LSM/SQL++/TVFs) in 8 weeks. The August 2026 adversarial review **refuted that** and pivoted to "thin glue over reused engines." ADR-1: *reuse, do not build from scratch.* ADR-2: SQLite as sole substrate. ADR-4: SQL++ abandoned permanently.

---

## 1. MEASURED FACTS (verified 2026-09-25)

| Fact | Value | How verified |
|---|---|---|
| `omnidb_server/` | **22 `.py` files, 4,195 LOC** | PowerShell per-module line count |
| `tests/` | **43 files, 4,625 LOC** | same |
| `store.py` | 1,028 LOC (24.5% of runtime) | same |
| `.github/` location | `omnidb-server/.github/workflows/` — **NOT** at workspace root | `Get-ChildItem -Recurse -Force -Filter .github` |
| `LICENSE` file | **DOES NOT EXIST** | `Get-ChildItem omnidb-server\LICENSE*` → empty |
| `tests/test_tenancy_boundaries.py` | 4 `xfail(strict=True)` | read |
| Hypothesis imports | 3 test files import at module level; `hypothesis` only in `[dev]` extra | read |
| Vault LOC ceiling | "~1,200 lines" vs actual 4,195 = **3.5×** | vault + measured |

Per-module LOC: store 1028 · portability 503 · consolidate 314 · cli 306 · vector_index 270 · server 260 · config 221 · recall 215 · skills 184 · doctor 164 · db 146 · langgraph 145 · embeddings 143 · llamaindex 58 · rerank 56 · validation 44 · fusion 33 · llm 33 · adapters/__init__ 29 · scoring 22 · `__init__` 12 · `__main__` 9.

---

## 2. ARCHITECTURE (verified)

```
MCP client ──stdio──► server.py create_mcp()  [10 tools, anyio.to_thread]
                          │ open_owned_store() ── _acquire_home_lock() FIRST
                          ▼
config.py ─► db.py ─► embeddings.py ─► vector_index.py ─► MemoryStore(store.py)
                                     └─► recall.py ─► fusion/scoring/rerank
                       consolidate.py, portability.py, skills.py, doctor.py, cli.py
```

**Write path** (`remember`): validate kind/content/metadata/links → encode once → INSERT+commit → `_after_commit` = [dirty flag → index mutation → compaction check → debounced persist]. Order centralized (ARCH-02) so no new write method can get it wrong.

**Read path** (`recall`): `fetch=min(max(k_fetch or 50,k),1000)` ×`agent_overfetch`(3.0) if agent-scoped → dense ANN top-fetch + sparse FTS `ORDER BY bm25()` → `rrf(k=60, weights=[0.7,0.3])` → re-fetch ENTIRE fused pool with selective filters (eligibility BEFORE truncation, AUDIT-I09) → `final_score` = RRF + `0.15·exp(-age/half_life)` + `0.05·log1p(access_count)` → skill-trigger merge → optional reranker → sort `(-score,id)` → slice k → access bump + query log in one commit.

**Five durable recovery markers:** `index_dirty`, `pending_reembed` (same-ID content revision, AUDIT-I01), `reembed_pending` (interrupted embedder migration, AUDIT-I03), `compaction_due` (hnsw tombstone ratio ≥ 0.2), `active_count != live_count` (sidecar lost/stale). SQLite rows are the source of truth; vectors always derivable.

**Config precedence:** CLI > env `OMNIDB_*` > `omnidb.toml` (CWD then home, home wins) > dataclass defaults. ~28 config fields.

**10 MCP tools:** remember, recall, search, schema, related, update_memory, forget, forget_many, consolidate, stats. (The vault's `Implementation Status.md:17` still says 7 — stale.)

---

## 3. VERIFIED DEFECTS — these are CONFIRMED, do not re-litigate

### 3.1 LEGAL-01 — missing LICENSE (Critical, P0, 5 min)
`omnidb-server/LICENSE` does not exist. `pyproject.toml:10` declares `license = {text="Apache-2.0"}`; `MANIFEST.in:3` includes it. Built sdist advertises a license it does not carry. PyPI license field already populated for 0.1.x. **Publication blocker.** Prefer PEP 639 (`license = "Apache-2.0"` + `license-files`).

### 3.2 BUG-15 — re-embed --model bricks the home (Critical, P0, 0.5 d)
- `cli.py` `re-embed` accepts `--model` and succeeds, pinning `meta.embedder = "fastembed:some/other"`.
- `embeddings.py:74-80` `make_embedder`/`FastEmbedEmbedder` has **no model parameter** — always resolves to class default `BAAI/bge-small-en-v1.5`.
- `Config` has **no field** for the model name.
- Every subsequent normal open → `_check_consistency` raises `OMN-006`: *"database was built with embedder X but fastembed:BAAI/bge-small-en-v1.5 is configured"*. No config surface can fix it. Home reopens only by re-running `re-embed` (which CAN read stored name).
- Contradicts `README.md:184-185`. **Customer-triggerable self-inflicted total lockout from a documented happy path.**

### 3.3 ARCH-11 — supersede/link unreachable from product surface (High, P1, 2-4 h)
`MemoryStore.supersede` and `MemoryStore.link` have **zero in-package callers** — only tests. The vault markets "memory that updates instead of accreting" and Implementation Status advertises "a manual `supersede()` API." **An agent talking MCP cannot supersede anything**, so in deployment memory can only accrete. Plumbing + tests exist; only the entry point is missing.

### 3.4 BUG-13 — doctor always exits 0 (High, P1, 15 min)
`doctor.py` returns `{"ok": all(...)}`; `cli.py` `return 0 if report["ok"] else 4`. **This is ALREADY FIXED in the current tree** (verified `cli.py:133-137`). The 09-21 audit's BUG-13 is stale. Do not re-report as open.

### 3.5 BUG-16a — rebuild unlinks before constructing (Med, P2, 1 h)
`store.py rebuild_index` and `portability.py reembed_home` unlink the old sidecar BEFORE constructing the replacement. A constructor failure destroys the persisted index (recoverable from rows, but the file is gone and the new index exists only in memory).

### 3.6 BUG-16b — HnswlibIndex.update never grows capacity (Med, P2, 2 h)
`vector_index.py` `update()` (~175-178) does not grow `cap`; `add()` (180-186) does. `update_memory` calls `update()`. **An update-heavy workload on a full `cap` hits hnswlib capacity errors.** Real bug, read the code to confirm before reporting.

### 3.7 SEC-02 — one regex wedges the whole store (Critical under hosted target, P0, 4-8 h)
`recall.py:167 compiled.search(query)` runs Python backtracking `re` **while holding the store-wide `RLock`** (`store.py:158`, ~21 `@_serialized` sites). Bounds exist and are real: admission 32 patterns/512 chars (`validation.py:14-15`, create AND import), matcher query ≤4096 chars, ≤2048 evaluations/recall (`recall.py:29-30`). **But bounding input does NOT bound backtracking time.** `(a+)+$` (<512 chars) vs 4,095 `a`s + `b` is super-polynomial. One tenant + one regex = every subsequent op from every tenant queues forever; `anyio` worker pool fills; stdio server stops responding, no timeout, no error. The lock is the CORRECT fix for BUG-02 — this is its cost. Fix order: (1) screen nested quantifiers at import, (2) RE2-class linear engine, (3) match off-lock on a bounded pool with wall-clock deadline (note: the STORE lock cannot be interrupted, so the deadline must live outside it), (4) `--trust-triggers` for operator bundles.

### 3.8 SEC-05 — consolidation cross-scope leak (Critical, P0, 0.5 d)
The 09-21 audit read an OLDER `consolidate.py`. **Current code partitions chunks by `agent_id` first** (`consolidate.py:222-230`) and **refuses mixed-scope chunks defensively** (`:288-290` drops, `:235-237` raises). So the specific leak described is **largely mitigated**. Verify current state before reporting. Residual: the report shape, and that `_LiveSemanticCandidates` dedup pools are per-scope but the *unscoped* pool (`agent_id IS NULL`) is readable by everyone.

### 3.9 BUG-17 — test_j10 (High, P0, 2-3 h)
`tests/test_post_publish.py:172-204` — **partially improved since the audit.** It now passes `--output str(report)` to a tmp_path, so it no longer overwrites the checked-in artifact. Remaining defects: (a) asserts against `bench/soak-results.txt` which is **gitignored and untracked** — cannot pass in fresh CI; (b) accepts `returncode in (0,1)` so a harness crash-with-1 still passes; (c) `rss_start_mb`/`rss_peak_mb` keys are now correct; (d) **60.35 s of a 109.33 s suite = 54.9% floor**. Fix: delete subprocess, make `soak.py` a callable returning its report dict, gate behind `-m slow` / `OMNIDB_RUN_PERF`.

### 3.10 BUG-18 — suite runnable from exactly one CWD (Low, P2, 1 h)
`d:/OMNIDB/.pytest_cache/v/cache/lastfailed` = `{"omnidb-server/tests": true}` — a **bare directory key**, which pytest writes only for a collection/import error. So the last root-level invocation collected ZERO tests and died at import. No root pointer, no `tests/conftest.py`. Also `--import-mode=importlib` breaks 14 test modules relying on default prepend-mode `sys.path` injection.

### 3.11 QA-12 — free-string config with no enum was FIXED
`_EMBEDDER_SELECTORS`/`_VECTOR_INDEX_SELECTORS` + `validate_selector` now exist and are enforced. The 09-21 QA-12 is **stale**. Verify before reporting.

### 3.12 CI-01..CI-09 — ALREADY FIXED in the current tree
Verified `ci.yml`: `permissions: contents: read` present; all three actions **pinned to full SHAs**; `OMNIDB_RUN_PERF: "1"` set at `:36`; `hnsw-required` job enforces skips-as-failures via JUnit XML parsing (`:64-70`); `--cov-branch --cov-fail-under=80` at `:37`; `mypy` in dev extra; `mypy==1.19.1` pinned. `release.yml`: pypi action **pinned to SHA** `dc37677b…` v1.14.2, `permissions` scoped correctly. **Do not re-report CI-01 through CI-09 as open.** The 09-21 audit predates these fixes.

### 3.13 ARCH-08 — zero logging (Medium, P1, 1-2 d)
`grep getLogger|basicConfig omnidb_server/` → **no matches**. No logger, no correlation id, no tenant attribution, nothing to ship to a pipeline. This is a **prerequisite for every tenant-facing fix after it** — SEC-02 and SEC-05 are undebuggable in production without attribution. Closest thing to a log today: `consolidate.py` `chunk_errors.append(str(exc))`, printed to stdout by `cli.py:221-224`, and able to contain `llm_base_url`.

### 3.14 QA-22 — local paths in tracked files (Medium, P1, 3-4 h)
`bench/research-v3-results.json:31100` commits a traceback containing `C:\Users\dev25\AppData\Roaming\Python\Python314\site-packages\qdrant_client\...` — the only tracked file carrying the operator username. Nine bench scripts hardcode absolute paths: `research_bench_v3.py:455` `D:/OMNIDB/tools/jdk-21.0.5+11`, `:464` + `probe_rerank_window.py:49` `E:/omnidb-bench-tmp`, `sweep_rerank_nfcorpus.py:49` `.hf-cache`, `compute_probe_27.py:8,13`, `compute_review_stats.py:34,38`, `sweep_rerank_scidocs.py:42,175`, `sweep_rerank_scifact.py:53`. **Benchmarks are not reproducible on any other machine**; the scidocs section of `rerank-window-results.md` is not reproducible from tracked files at all (its JSON lives on E:). Also `examples/claude-desktop-config.example.json:15` ships `/Users/yourname/.omnidb`.

### 3.15 DEBT-11 — vendored toolchains inside the project tree (Medium, P2, 2-4 h)
`texlive/` 9,112 files (313 MB), `tools/jdk-21.0.5+11/` 490 files (331 MB), `TinyTeX-1-windows-v2026.09.exe` 73.7 MB, `.hf-cache/` + `bench/.cache3/` ~5.4 GB model blobs — all inside `d:/OMNIDB` with **no documented boundary**. Correctly untracked, but nothing states they are not project code, so every exhaustive review must re-derive the boundary by hand. `research_bench_v3.py:455` depends on the JDK being at exactly that path.

### 3.16 BUG-14 — five exception classes, no shared base (Medium, P2, 0.5-1 d)
`StoreError`, `ConfigError`, `DbError`, `VectorIndexError`, `EmbeddingError` are all direct `RuntimeError` children with no common ancestor, enumerated only at `cli.py:270-271`. A host mapping exceptions to JSON-RPC needs 5 `except` clauses and will miss the 6th. Compounding: `StoreError.__init__` bakes the code into the message (`f"[{code}] {message}"`) *and* keeps `self.code`, so callers must bracket-parse to recover the bare message.

### 3.17 QA-14 — "half-life" is a time constant (Low, P3, 15 min)
`scoring.py:26` computes `exp(-age/τ)`, whose actual half-life is `τ·ln2 ≈ 116 h` for configured `half_life_hours=168`. Both the README and the config name misdescribe the knob by ~44%. Faithful to the spec's error.

### 3.18 QA-16 — error-code gaps (Low, P2, 1 h)
Codes 001, 007-010, 012, 014-019 absent. `E_CONSOLIDATION_FAILED` (`store.py:47`) is **never referenced** — `consolidate.py:260` uses the literal `"OMN-013"`. `cli.py:293` passes a bare string literal instead of a named constant.

### 3.19 QA-20 (Low-Med, P2, 2 h)
- `fusion.py:28` `zip(rankings, weights)` silently drops legs on length mismatch, contradicting `:14`'s "must match".
- `recall.py:204-205` assumes metadata is a dict; a foreign home with JSON-array metadata raises `TypeError` inside recall.
- `langgraph.py:107` round-trips the entire framework value into `metadata["_value"]` → content stored **twice** (2× amplification, doubled exposure surface).

### 3.20 SEC-06 — bearer token + transcript over unvalidated URL scheme (High, P1, 1 h)
`consolidate.py:101-113` POSTs the **entire episode transcript** + `Authorization: Bearer {api_key}` to `{llm_base_url}/chat/completions`. Nothing in `config.py` rejects a non-https `llm_base_url`, so an operator typo (`http://internal-llm/…`) silently leaks credential + tenant content in cleartext. It is the only outbound data path. Second egress worth naming: `llm.py:16` probes `http://127.0.0.1:11434/v1/models` (loopback, gated off by default).

### 3.21 SEC-07 — zip members read with no decompressed-size limit (Medium, P1, 1 h)
`skills.py` `zf.read()` with **no bound on decompressed size** — a small bundle can inflate to GBs before any content cap applies. A memory DoS preceding the regex one. Fix: total-bytes budget + per-member cap checked BEFORE `read()`.

### 3.22 BUG-16c — doctor false-clean on corrupt marker (Med, P1, 30 min)
`doctor.py` `_recovery_check`: a corrupt `pending_reembed` value is caught, but the shape check lets a bad value pass. **Verify current code** — the 09-21 text says it swallows to `-1` then filters `> 0` so pending work reports clean. `test_doctor_truthfulness.py:83-95` asserts corrupt → `ok is False`. **Likely already fixed; verify before reporting.**

### 3.23 MT-02 — agent_id carries no authentication (Critical for hosted, P0, blocked)
Two independent sources agree this was never a boundary: `Goals and Non-Goals.md:40` lists multi-tenant isolation as a v2 non-goal; `Comprehensive Analysis (Aug 2026).md:73` says the `agent_id` column was CUT and later re-added as recall-scoping convenience. `edges`, `memory_fts`, `meta` have **no tenant column at all**. `agent_id` is nullable and `NULL` means visible-to-everyone — the permissive branch is the default.

**Four known, code-confirmed, untested bypasses** (pinned as xfail in `test_tenancy_boundaries.py`): (1) dense ANN leg is global, filtered post-hoc; (2) `related()` walks forged edges across tenants; (3) `stats()` enumerates all agents; (4) one shared `vectors.npz` holds all tenants' vectors.

### 3.24 MT-06 — every documented tenancy bypass has zero positive assertions
`MULTI_TENANCY_THREAT_MODEL.md:28-33` enumerates them. A search across all test modules finds **no test asserting any of these four behaviours in either direction** — none pins "this leak exists" and none guards "this must not widen." The xfails are the right instrument but only 3 of 4 patterns are covered and they don't assert current behavior. Turn the register into a mechanically tracked checklist.

### 3.25 OPS-01 — plaintext at rest, no key management, no retention (Medium, P1, 1-2 d)
Shipped posture is "use OS disk encryption" (`README.md:139,155-157`); SQLCipher deliberately unbundled. `queries.query` stores **verbatim recall/search text** and `omnidb-server query-log` dumps it in cleartext — tenant content in a log table with no export control. `forget` is soft delete; `purge` is manual; **nothing enforces a retention ceiling**.

---

## 4. VERIFIED DOCUMENTATION DEFECTS (dominant defect class)

### 4.1 DOC-10 — v3 CE column void in all 16 cells (High, P1, 2 h)
`research-v3-results.md:9` header names `cross-encoder/ms-marco-MiniLM-L-6-v2`. **Every** JSON cell is tagged `BAAI/bge-reranker-base` (16/16). Base hybrid cells match exactly → only the CE column drifted. **Three cells flip sign:** scifact/minilm md −0.0177 vs JSON **+0.0016**; nfcorpus/e5-small md **+0.0025** vs JSON **−0.0275**; fiqa/minilm md **+0.0085** vs JSON **−0.0064**. That is the difference between "the reranker helped" and "hurt." The hand-written Conclusion §2 inherits it. A third number-set lives in `research_bench_v3.py:80-83` matching neither. **Safe:** the paper's headline rests on `review_probe_deltas.json` + `rerank-window-results.md`, not this column.

### 4.2 DOC-09 / AUDIT.md I12 — cited numbers exist in no file (Critical for audit trail, P0, 0.5 d)
`docs/AUDIT.md:394` claims a **passing** 2-min run: p95 ratio **3.758** ≤ 4.0, **381 ops/s**, growth **629 ops/s**, recall p95 ≈1→**5.6 ms**, both bound to `source_identity_sha256: c1e17113…`. On disk: `soak-results-v2.txt` has `source_identity_sha256: 4ca9b6be…`, `ops_per_second: 548.91`, `p95_ratio: 5.337`, **`validation_gate_passed: False`**. `growth-results.txt` has `52de74cb…`, `1242.95 ops/s`, bucket p95 1.919→3.761 ms. **No file contains `c1e17113`, `3.758`, `381`, or `629`.** Verdict conflict: AUDIT + perf-report say PASS; artifact says FAIL. `DATASTORE_SPIKE.md:28` reports the correct verdict (gate fail) with 4 of 6 numbers also wrong.

**Structural defect:** the evidence for both availability claims lives in **gitignored** files (`.gitignore:30-31`), so it is unverifiable by construction, and the test suite can overwrite it.

### 4.3 One 180-byte artifact, three descriptions (High, P1, 0.5 d)
`bench/soak-results.txt` actual: ops **36,096**, cycles **4,512**, recall mean **7.93**, p95 **16.03**, first/last third **4.22→17.31** (ratio **4.103** → FAILS the 4.0 gate), RSS **37.3→89.9**, WAL **2.02**. `AUDIT.md:196` says "3.30→34.38, 10.42×, overall 32.10, RSS 36.9→61.9, WAL 1.88" — all six numbers wrong. `perf-report.md:48-54` says "21,280 ops, 2,660 cycles, 5.50, 11.84, 3.40→12.95, 37.1→68.8, 1.84" and concludes **"passes all three stability gates"** — the file it cites fails.

### 4.4 Three committed runs, one source hash, three verdicts (Medium, P2, 1 h)
`soak_stab.txt` / `soak_stab1.txt` / `soak_stab2.txt` at repo root: identical `workload_sha256 c85dd173…` AND identical `source_identity_sha256 52de74cb…`, yet p95_ratio **4.156 (False)** / **2.379 (True)** / **2.632 (True)**. The 4.0 threshold is crossed by **run-to-run variance, not code change**. Also these three escape the `bench/soak-results*.txt` ignore rule while a file named `.killed.bak` is permanent history.

### 4.5 perf-report.md recall figures a full generation behind (Medium, P2, 1 h)
`perf-report.md:31-38` quotes 384/10k/ef128 = **0.873**, 384/100k = **0.383**, 768/10k = **0.823**, 768/100k = **0.359**. `bench/results.md` actual: **0.888 / 0.519 / 0.854 / 0.395**. numpy moved 2.3.5→2.5.2 between report and artifact. `README.md:128` still says "HNSW at ef=128 (0.837)" vs **0.854**. `AUDIT.md:339` 100k row cites 0.281/0.240, P95 0.628/0.825 vs **0.378/0.278, P95 0.864/1.101**.

### 4.6 DOC-06 — ARCHITECTURE.md has six factual errors (Medium, P1, 2 h)
1. `:22` assigns `check`/`purge` to `portability.py`; both are on `MemoryStore` and `portability.py` defines **neither**. `:20` also credits them to `store.py` — double-credit plus an invention.
2. `:33` describes an "importer-only explicit-id hook"; `remember(_explicit_id=…)` has **zero callers** — `import_jsonl` writes its own INSERT.
3. `:24` lists four `cli.py` subcommands; there are **20**.
4. `:63` says "the top `max(4k,k)` fused ids are re-fetched" — the code re-fetches the **entire** fused pool and truncates after. `max(k*4,k)` is a no-op for k≥1.
5. `:105-107` describes the authorizer without noting **`SQLITE_FUNCTION` is allowed unconditionally** (`store.py:753`) — the actual residual risk of `search`.
6. **`corpus_mode` is documented nowhere in ARCHITECTURE.md** despite zeroing both rerank weights (justified by a measured 40-56% nDCG loss). It is the single most behavior-changing config flag and it is invisible.

### 4.7 DOC-18 — paper ships three generations; .docx is the oldest (High for artifact, P1, 2 h)
`build_docx.js:8` reads `paper.md` — the **oldest** generation. The `.docx` therefore asserts claims the current `.tex` explicitly withdraws: "three of four corpora" (retracted to "robust on two of the four"); ρ=**0.60 at n=16** (retracted to ρ=0.40 at n=4). Carries 3-placeholder author block + `author@example.org` + `[sponsor and grant number to be added]`. Omits the dataset-categorization table, the qdrant "Vector-only" column, Fig. 2 (headroom scatter) and Fig. 3 (ECDF). Hard-scales the image to 560×290 px, **vertically squashing ~26%** the current 1.43:1 PNG. Author block in the current `.tex:23-25` is still `A. Author` / `a.author@university.edu` with an in-file comment saying these "must not appear in a submitted manuscript."

Current `.tex` build is **clean**: 10 pages, zero errors, zero undefined refs/citations, **zero overfull hboxes**, 6 underfull (cosmetic, all from unbreakable `\texttt{}` model IDs). The stale `build/` copy has an 88pt overfull box the current build fixed.

### 4.8 DOC-04 / Glossary — unbuilt systems in the present tense (Medium, P2, 1 h)
`Glossary.md:32-33` says `VECTOR_SIM`/`MEMORY_RECALL` "are exposed this way" — those identifiers appear **zero times** in any `.py`, no `create_function`/`create_module` anywhere. `Architecture.canvas:15` repeats it. `Glossary.md:61` defines Consolidation as a "**Scheduled** background process" — **no scheduler exists**; `consolidate` is an opt-in tool + CLI subcommand, and the only background loop is WAL-checkpoint-plus-flush. `Glossary.md:18` "Rerank" describes float32-rescore-over-quantized-shortlist which exists **only in `bench/baseline.py`**; shipped rerankers are `none`/`lexical_overlap`/`recency_first`.

### 4.9 DOC-01 / DOC-02 — count drift and the 3.5× ceiling breach (Medium, P2, 1 h)
Test counts across notes: 32 → 126 → 203 → 224 → 228 → 244 → 127 → 34. File/LOC: "44 .py / ~1,985 LOC" (Post-Publish), "21 files / ~3,300" (Issues), "22 files / 4,603" (09-21), actual 22/4,195. Coverage: 79.5% → 85.3% → 86.8%. **`Comprehensive Analysis (Aug 2026).md:111` "Honest ceiling: ~1,200 lines doing one job well. Biggest risk is re-inflation into the original ambitions; keep cutting."** — and the from-scratch DB design was abandoned partly because it outgrew a solo maintainer's budget. The named risk materialised.

### 4.10 DOC-07 / DOC-12 — ADR citations point outside the repo (Low-Med, P2, 1 h)
`ARCHITECTURE.md:5-7` and `README.md:5` reference `../OmniDB Vault/`, and ADR 2/4/7 are cited in code-facing prose — but the vault is **not in the git repository** (`d:/OMNIDB/.git` = False; `omnidb-server/.git` = True). The far more numerous `AUDIT-*`/`PERF-*`/`ARCH-*`/`SEC-*`/`BUG-*`/`QA-*`/`J-*`/`DEBT-*` tags peppering the code are **not indexed in any top-level document**. `tests/test_audit_remediation.py:1-5` and `tests/test_enhancements_p1.py:1-4` cite vault paths as their stated reason for existing — a test file is the one artifact a clone is guaranteed to have, and its cited reason is guaranteed not to be.

### 4.11 DOC-11 — mutation-report integrity hash no longer matches (Low, P2, 30 m)
`docs/mutation-report.md` binds itself to `scoring.py 5d8f6e7f…` (still matches) and `fusion.py 5ae71f86…` (**current is `5d33d7e7…`** — file changed after the report, report not updated). This is the one place the repo declares how it should be invalidated, and the mechanism caught a real drift — **and it is unenforced: nothing runs the check.**

### 4.12 DOC-16 — market claims under-credited; two survivors still wrong (Medium, P1, 1 h)
`_verify/` is a 2026-08-21 fact-check sweep. Four of six direct fetches failed. The three successful Brave SERP pages were never re-read: `mordor_agentic_vec.html` **confirms the number and refutes the year** (live page: $0.57B 2026 → **$1.73B 2031**, 24.86% CAGR; the PDF says 2030 — wrong by one year); `brave2.html` **dates the Oracle announcement to 24 December 2025** (the PDF says March 2026 — wrong by three months). The two claims the vault rejects **stay rejected** (Reddit taxonomy, StackOne report). Anyone re-marketing from the PDF republishes one wrong year, one wrong date, two unverifiable citations.

### 4.13 DOC-17 — omni.txt is a lossy derivative (Medium, P2, 2 h)
`omni.txt` is `pdftotext -layout` of `OmniDB_v1_Design_Document.pdf`, not independent: title-page letter-spacing artifacts (`"U NI F I ED DATAB ASE ENG I NE"`), per-page footers, lengths matching to 0.06%. **Three tables arrived rotated by one row** (`:779-790`, `:807-831`, `:994-1003`) — the corruption is documented **in the vault**, not in `omni.txt`, so any reader of the extraction gets wrong values with no warning. The PDF carries `Author: Z.ai`, `Creator: Z.ai`, `Producer: http://z.ai` and a `NotoSerifSC-Regular` fallback in an all-English document — it is machine-generated — while its title page stamps "CONFIDENTIAL ENGINEERING REVIEW."

### 4.14 QA-18 / CHANGELOG shape (Low, P3, 30 m)
`### Fixed` appears **three times** inside `[0.1.3]` (`:10, :88, :209`); `### Added` three times; `## [Unreleased]` (`:214`) sits **below** the released section, inverting the chronological order claimed at `:5-6`.

### 4.15 DEBT-09 — SQLCipher "needs no code changes" is false (Low, P2, 1 h)
`README.md:153-165` claims a SQLCipher swap "needs no code changes." `db.py:87` and `server.py:179` both call `sqlite3.connect` directly (two edits), and `_home()` derived from `PRAGMA database_list` (`store.py:1092-1093`) would break.

---

## 5. VERIFIED PERFORMANCE FACTS

| Measurement | Value | Source |
|---|---|---|
| Cold start, fresh home | 26 ms | perf-report.md:19 |
| Single `remember` | 3.8 k writes/s | perf-report.md:20 |
| `remember_many(1000)` | ~29 k writes/s (~34 ms) | perf-report.md:21 |
| Recall p50, <1k rows | 0.37 ms | perf-report.md:22 |
| HNSW recall@10 @100k | **0.519** (384-d, ef128) / **0.395** (768-d, ef128) | bench/results.md |
| SQ8 shortlist+f32 rerank @100k | 0.881 / 0.896 at **59 / 131 ms** p50 | bench/results.md |
| Hybrid nDCG@10 (full corpora) | nfcorpus 0.3747 · scifact 0.7508 · scidocs 0.2075 · fiqa 0.3922 | research-v3-results.md |
| vs bm25s | omnidb wins all 4; fiqa 0.3922 vs 0.2380 | same |
| vs dense-only | **loses on scidocs** +0.0110 (p=0.0102) | same |
| Latency omnidb vs qdrant | omnidb is **1.6-2.0× slower** at every corpus (nfcorpus 8.1-9.0 vs 4.6-6.1 ms; fiqa 117-124 vs 97-110) | research-v3-results.json |
| CE inference cost | **274 s (nfcorpus) → 10,995 s (fiqa e5-small)** per cell — a 40× spread never mentioned in prose | same |
| Brute-force scaling 512-d | p50/p95: 0.99/1.92 (110→500) · 1.87/2.69 (1k→2k) · 2.93/3.76 (2k→4k) | growth-results.txt |
| Soak v2 (60 s) | 548.91 ops/s, recall p50 5.255 / p95 17.625, **p95_ratio 5.337 → FAIL** | soak-results-v2.txt |
| Mutation score | **33%** (10/30 killed) | mutation-report.md:43 |
| Coverage | 86.8% (2,433 stmts / 321 missed); `server.py` at **63.5%** — lowest of any substantive module | 09-21 audit |

**PERF findings:**
- **PERF-09 (High, P2, 3-5 d):** No quantized backend. `SQ8`/`int8` appear **nowhere** in `vector_index.py`; brute is full-float. The research note called this "the top gap" and it was never closed.
- **PERF-10 (High, P2, 4-8 h):** Idle RAM is O(corpus) per tenant. `.npz` is a zip container and **cannot be mmap'd**. At 768-d/100k that is ~300 MB resident, **multiplied by tenant count**.
- **PERF-11 (Med, P2, 0.5-1 d):** Retrieval quality capped by toy rerankers — `REGISTRY` holds only `none`/`lexical_overlap`/`recency_first`; no cross-encoder. Ironically the paper proves CE doesn't help on hybrid anyway — so the *right* answer may be better fusion, not rerank.
- **PERF-12 (Med, P2, 1-2 w):** `persist` is a full O(N) snapshot; the PERF-01 *policy* fix bounds frequency, not cost. `N × (4·dim + 8)` bytes per flush.
- **PERF-13 (Med, P2, 2 h):** `import_jsonl` encodes ALL content in one call (`portability.py`) with no chunking, unlike the other two bulk paths (512/256).
- **PERF-08′ (Med, P1, 4-8 h):** `agent_overfetch` shipped but **the recall-quality eval is pending** — mechanism exists, acceptance test missing.
- **ARCH-12 (Med, P2, 0.5 d):** `bench/baseline.py:137-138` computes `best = max(hnsw_scores + [min(sq8_scores)])` then gates `best >= 0.95`. It is a **max over the sweep** — passes on the single best cell (1.0 at 384-d/1k/ef128) while the worst cell in the same sweep is 0.17. **A gate reading a maximum cannot fail.**

---

## 6. VERIFIED TEST-SUITE FACTS

- **43 files, 4,625 LOC.** Suite last recorded at **271 passed / 4 skipped / 109.33 s**.
- `conftest.py` forces `embedder="hashing"`, `vector_index="brute"` — **every test is offline and deterministic.**
- `conftest.py:23-27` autouse session fixture strips all `OMNIDB_*` env vars.
- **Gates:** `OMNIDB_RUN_PERF=1` (perf_smoke module, remember_many 800/s), `OMNIDB_TEST_SEMANTIC=1` (downloads a model), `pytest.importorskip("hnswlib")` at 5 sites, `importorskip("mcp")` at 2 sites.
- **4 `xfail(strict=True)`** tenancy tripwires in `test_tenancy_boundaries.py`.
- **Mutation analysis: 33%.** `scripts/mutation_check.py` applies 30 hand-written mutations to `fusion.py`+`scoring.py`; **20 survived.** Root cause per `mutation-report.md:196-199`: *"neither pure function has a single direct unit test — everything flows through `store.recall`."* Killing tests are already written out in that report.
- **Weak assertions found:** `test_memory.py:27,60` conditionally guarded ordering claims; `test_chaos.py:114` `or reconciled_on_first_reopen` masks a real index/live mismatch; `test_post_publish.py:144-153` (J-06) asserts on `inspect.getsource` **text**, not behavior; `test_post_publish.py:280` named "collapses writes to one commit" but never counts commits; `test_audit_remediation.py:130` `assert any(...) or len(tail_ids)==5` trivially satisfiable.
- **Hardcoded 512** in ~15 test files for `BruteForceIndex(tmp/"vectors.npz", 512)` — if `HashingEmbedder.dim` changes these become silent dim-mismatch failures.
- `hypothesis` is a module-level import in 3 test files but lives only in `[dev]` → **collection error** on a minimal install.

---

## 7. WHAT IS ALREADY FIXED — DO NOT RE-REPORT

Verified in the current tree; the 09-21 audit predates these:

- **BUG-13** doctor exit code → `cli.py:133-137` returns `0 if report["ok"] else 4`. FIXED.
- **QA-12** config enum → `_EMBEDDER_SELECTORS`/`_VECTOR_INDEX_SELECTORS` + `validate_selector`. FIXED.
- **CI-01** perf gates armed → `ci.yml:36` `OMNIDB_RUN_PERF: "1"`. FIXED.
- **CI-02** `permissions:` → `ci.yml:8-9`. FIXED.
- **CI-03** pypi action pinned → `release.yml:55` SHA `dc37677b…`. FIXED. (Also all other actions SHA-pinned.)
- **CI-04** mypy pinned → `dev` extra has `mypy==1.19.1`.
- **CI-06** coverage floor enforced → `ci.yml:37` `--cov-branch --cov-fail-under=80`.
- **CI-07** hnsw-required job → `ci.yml:45-75` parses JUnit XML, asserts ≥4 cases and **zero skips**. (Weakness remains: it pins 4 hardcoded node ids rather than index invariants.)
- **CI-08** env sanitization → `conftest.py:23-27` autouse scrub.
- **BUG-16c** doctor corrupt-marker → `test_doctor_truthfulness.py:83-95` asserts `ok is False`. Verify but likely FIXED.
- **BUG-12** k/k_fetch clamp → `recall.py:15` `_MAX_RECALL_K = 1000`.
- **SEC-04 / QA-02** → both closed and tested.
- **All 22 AUDIT-I items** → `docs/AUDIT.md:375-411` records all implemented with regression tests.
- **DOC-01 count of 22 modules** → confirmed.

---

## 8. RESEARCH EVIDENCE — TRUST TIERS

**TRUSTWORTHY (reproducible, load-bearing):**
- `rerank-window-results.md` + `rerank-window-results.json` / `nfcorpus-rerank-window-results.json` / `scifact-window-results.json` / `review_probe_deltas.json` — per-query deltas retained, 27 cells, method-controlled, tagged `BAAI/bge-reranker-base`. **The paper's headline rests here and is safe.**
- `research-v2-results.md` — 5 datasets × 4 embedders, bootstrap CIs, Holm.
- `research-v3-results.md` **non-CE columns** — full-size corpora.
- `growth-results.txt`, `bench/results.md`.

**VOID pending re-render:** the `omnidb + cross-encoder` column of `research-v3-results.md` and Conclusion §2/§3 citing it.

**UNVERIFIABLE BY CONSTRUCTION:** everything in `AUDIT.md` §I12 remediation and `perf-report.md` §Stability soak — cited files are gitignored, and three copies that escaped the ignore rule disagree with each other.

**Best-evidenced claims in the repo:**
1. CE rerank of a strong fused hybrid is a net loss across 4 corpora × 2 embedders × 2-3 windows.
2. CE rerank of a *weak single-leg BM25* stage helps (fiqa +0.0808 p<0.0001; scifact +0.0484; scidocs +0.0088) — **but nfcorpus gives nothing (+0.0082, p=0.5040)**, so weakness is necessary, not sufficient.
3. Low recall alone does not imply a CE gain — nfcorpus bm25 recall@100 = 0.2376, lowest of any stage, yet n.s. Genuinely useful boundary result.
4. Memory-style rerank costs 40-56% nDCG@10 on corpus IR.
5. Optimal `dense_weight` is dataset-dependent: 0.7 (nfcorpus) → ≤0.3 (scifact); v1 sweep on scifact is monotone decreasing (0.3405 at w=0.3).
6. A weak embedder makes the lexical leg outvote correct dense evidence: dense-alone recall@5 = 0.968 but full hybrid = 0.416 (`comparative-results.md:23-30`).

---

## 9. ARCHITECTURAL REALITY

- **One writer per home, permanently** (ADR-2). Real OS advisory lock, acquired BEFORE any mutable init, released by OS on process death.
- **`recall` is a write path** (access bump + optional commit) — read-replica routing silently corrupts scoring. Easy to miss in a Postgres design.
- **`agent_id` is a retrieval label, not an auth boundary.** NULL = visible to everyone = the permissive default.
- **`edge`/`memory_fts`/`meta` have no tenant column.** Adding tenancy is a storage-contract rebuild, not an `ALTER TABLE`.
- **Python `sqlite3` cannot register virtual tables** → ADR-4's TVFs deferred to a Go/Rust port that has not happened.
- **The `store.py ↔ recall.py` cycle** is held apart only by function-body imports (`recall.py:10` reaches into `store` privates; `store.py` lazily imports `run_recall`).
- **`server.py` coverage 63.5%** — lowest of any substantive module, and the gap is structural: stdio tests run out-of-process so the in-process transport never executes under coverage. **In a hosted deployment the transport IS the product.**

---

## 10. RULES FOR EVERY AGENT

1. **Read the source before asserting.** Every finding needs `file:line`. If you cannot cite it, do not report it.
2. **Do not re-report anything in §7.** Verify first; the 09-21 audit is stale on several items.
3. **Quantify.** "Slow" is not a finding; "59 ms p50 at 100k, 40× the hybrid path" is.
4. **Effort estimates in hours**, with a range. Base them on the actual code you read.
5. **Distinguish measured from asserted.** `bench/*.md` = measured (but check §8 trust tier). Docs = asserted. Your own reasoning = inferred.
6. **Severity by business impact**, not by how interesting the bug is. Use Critical/High/Medium/Low.
7. **Priority P0/P1/P2/P3** where P0 = release blocker, P1 = next two weeks, P2 = this quarter, P3 = backlog.
8. **Respect the single-writer and local-first constraints.** A recommendation that breaks "no network listener" or "one process per home" must say so explicitly and price the trade-off.
9. **Prefer the smallest fix that makes the invariant enforceable.** The repo's own pattern is a named constant + a test.
10. **Be honest about uncertainty.** If something is unmeasured, say "unmeasured" rather than estimating a number and presenting it as data.
