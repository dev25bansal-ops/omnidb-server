# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — multi-tenant baseline harness (three defects)
Re-verifying the tenancy paper's Table IV against the harness that produces it
found three problems, each of which made the table unreproducible or
misleading.
- **Single un-warmed timing sample.** `_time` called each query once, so the
  Query ms column was dominated by page faults and cold-cache effects:
  re-running the same harness on the same host moved individual rows by
  **2–47×**. Query arms now take the median of 21 warmed calls (spread
  1.0–1.2×); builds stay single-shot, being non-idempotent.
- **The qdrant payload-filter arm timed an empty query.** Both tenants upserted
  ids `0..199` into one collection, so the second tenant's points *overwrote*
  the first's (same ids, different payload) and the querying tenant had **zero
  points** — 0 hits against 10 for every other system, in the row that exists
  to be the production analogue of the shared-index design. Ids are now
  disjoint per tenant; all six arms return 10 rows.
- **The two OmniDB arms returned 1 row against the others' 10.** They matched
  `"unique1"`, a token held by exactly one row, so they timed a 1-row lookup.
  They now match a term every row of the querying tenant carries.
- `boundary_in_file` corrected from `False` to `True` on the
  "OmniDB partitioned" row: it builds through the shipped per-tenant path, so
  each tenant gets its own FTS5 table and vector sidecar. The `False` belonged
  to the sibling "shared" arm, which sets it only after DROPping those tables.
- A `_time` bug caught while fixing the above: it called `fn()` once to capture
  the return value and then `repeat` more times, so `repeat=1` ran builds twice
  and four baselines dropped off the table on "table already exists".
- The paper's Table IV query times are regenerated from the corrected harness.
  Its "measured identically" framing is narrowed: the two OmniDB rows are
  lexically retrieved while the four external systems are vector search, so the
  table shares corpus and `k` but not index type.

### Fixed — MCP handshake advertised the wrong version
- `initialize` returned `serverInfo.version = "1.29.1"` — the version of the
  **mcp SDK package**, not of this server. `FastMCP` forwards no version to the
  low-level `Server`, whose `create_initialization_options` falls back to
  `pkg_version("mcp")`. Set explicitly on both transports (`server.py`,
  `http_server.py`); the handshake now reports `0.1.3`.
- Why it survived: the stdio conformance test asserted `serverInfo.name` and
  never `serverInfo.version`, so from a client's perspective this server's own
  version was unobservable. The test now asserts the version, and that
  assertion was confirmed to fail with the fix reverted.

### Changed — paper revisions (round 8, 2026-10-08)
- **Citation integrity.** The tenancy paper's bibliography defined eight
  entries and the prose cited one, leaving named claims about external systems
  unsourced: Qdrant (9 mentions), LanceDB (5), `sqlite-vec`, HNSW, BM25 and
  FTS5 were all discussed with no citation. Six citations added at first
  substantive mention, plus one in the addendum for the MCP query tool. Both
  documents now cite every entry they define (21/21); both compile with zero
  undefined citations.
- **Stale derivative drafts flagged, not silently shipped.** `paper.md`,
  `omnidb_ieee_access.md` and `omnidb_ieee_access.docx` are dated 2026-09-15
  and carry claims corrected since — most seriously the false touche-2020
  "near-degenerate judgments" rationale, the "bounds nDCG at the ceiling"
  wording, the 2,000-document capped pre-run, a per-cell rather than
  family-wide multiplicity correction (five significant losses where the
  current paper reports four), and a +CE column matching no committed
  artifact. `paper/DERIVATIVES.md` records these against the corrected
  positions and marks the `.docx` withdrawn until `paper.md` is brought
  current. `build_docx.js` consumes the stale `paper.md`, so re-running it
  as-is would republish them.
- `paper/check_citations.py` added: verifies every `\cite` resolves against a
  `\bibitem` across both `\input`-spliced documents, and reports entries that
  are defined but never cited.

### Changed — paper revisions (round 7, 2026-10-08)
- **touche-2020 exclusion rationale was factually wrong and is replaced.**
  Rounds 5 and 6 both asserted the collection's graded judgments were
  near-degenerate ("one or two relevant documents per query") as the reason to
  exclude it. The real qrels — `BeIR/webis-touche2020-qrels`, mirrored at
  `resolve/main/test.tsv` (earlier fetch attempts 401'd because the dataset was
  requested under the wrong name, `touche-2020`) — show **49 judged queries,
  every one with at least 6 relevant documents, median 19, across three graded
  levels**. The old claim is now explicitly flagged as false in the text. The
  exclusion stands, but on the two grounds that are true: 382k documents make a
  full embedding pass a ~10-hour job against a hardware budget that the other
  four corpora exhaust in minutes, and only 49 queries carry judgments, which is
  too few for the per-query paired tests the grid is built on. Both statements
  of the rationale (Sec. IV and Sec. VI) were stale and both are corrected.
- Table III (architecture-matched controls) widened from `table` to `table*`:
  the 8-column body was set at `\footnotesize` inside a single column and was
  cramped to the point of being hard to read. Now full-width, same font size.
- "All three arms run the same index" removed. It is not true as written — the
  sparse-only and dense-only arms necessarily execute different legs. The
  caption now says the arms share a corpus snapshot, preprocessing, and
  retrieval stack, and differ in which legs execute and at what fusion weights.
- "Bounds nDCG at the ceiling" replaced: the recall gap is a headroom
  *measurement*, not a bound on achievable nDCG. Prose now says first-stage
  recall limits which relevant documents are available for recovery.
- The 2{,}000-document capped pre-run is gone. It predated the switch to
  per-corpus full collections and no longer describes what was executed.
- "The lexical leg is neutralized" softened to a statement about relative
  weakness — the lexical leg contributes little on SciDocs — rather than an
  absolute claim that it stops working.

### Changed — paper revisions (round 6, 2026-10-08)
- **Table II now carries the headline evidence.** Added `Delta_CE` and the
  16-way family-corrected p for every cell, with the four survivors daggered
  (all losses). The central negative result is auditable from the table alone
  rather than requiring the prose.
- Conclusion: the recall-gap sentence rewritten to "first-stage recall limits
  which relevant documents are available for recovery; we use the
  hybrid-minus-BM25 recall gap as an empirical measure of that headroom, and we
  do not treat it as a bound on achievable nDCG".
- "Two-factor characterization" replaced throughout by "a measured
  recall-headroom factor plus an unmeasured model-corpus boundary hypothesis"
  (the second factor is not measured and is no longer presented as established).
- Matched-control table: "differ only in fusion weights" corrected - the
  executed legs necessarily differ; the configuration difference is which legs
  run and with what weights.
- Table values stated as means over each corpus's queries from per-query
  nDCG@10, paired within query.
- Section order: the matched-controls subsection now precedes Hybrid-vs-Vector-
  Only, so its table is introduced before it appears.
- Figure 2 caption: the arm choice is stated as a fixed convention (one x-axis
  definition) rather than a post-hoc pick, and the panel is labelled
  descriptive at n=4.
- "The deepening is systematic" given a countable basis (all eight hybrid arms
  move down or flat).
- touche-2020 exclusion: kept qualitative (no per-query statistic asserted) -
  the collection's qrels are not reachable from the mirror used for the other
  four corpora, so no number is claimed for it.
- ECDF percentages marked approximate; the 2026-09-30 editorial note marked
  resolved.

### Changed — paper revisions (round 5, 2026-10-08)
- **Engine count reconciled.** The methodology said "six engines" while the
  results added a seventh (matched sparse-only). Resolved as an explicitly
  separate experiment: the grid keeps six engines and four secondary
  comparisons; `omnidb sparse-only` is a controlled case study in its own
  subsection with its own table, its own family (two contrasts per corpus,
  Holm), and a stated comparison unit (per-corpus, paired within query).
- **New subsection + table** for the matched controls: sparse-only, dense-only,
  and hybrid nDCG@10 across four corpora, with hybrid-minus-single-leg deltas
  and Holm p-values. Controlled comparisons (varying only fusion weights inside
  one stack) are now distinguished from external implementation baselines.
- Figure 2 caption: removed a false "stronger on every corpus" claim (MiniLM
  exceeds BGE-base on SciDocs).
- CE section: replaced a two-marginal-CI example — not a CI on the paired
  difference — with the paired-test statement.
- Conclusion and mechanism: "bounded by recall" replaced by "first-stage recall
  limits which relevant documents are available to recover"; the headroom
  quantity is now defined explicitly as H_w = recall_hybrid@w - recall_BM25@w.
- Retitled the window section "No Tested Window Improves the Second Stage";
  "replicates" -> "corroborates ... under a second re-ranker and varying
  candidate windows".
- touche-2020 exclusion re-justified on an intrinsic property of the collection
  (near-degenerate graded judgments) rather than on an earlier protocol's cap.
- Notation: Equation (1) uses k_RRF; Figure 1 caption uses K; "single-seed"
  removed from the abstract and the limits sections.

### Added
- Graph retrieval leg (`omnidb_server.graph_leg` + `store.graph_ranking` + the
  recall fusion branch): a bounded, both-endpoints-fenced adjacency expansion
  feeds a max-discounted-path ranking, fused as a third RRF leg. Ships off
  (`graph_weight=0.0`). The weight behaves as a switch within the tested range:
  a controlled synthetic mechanism test (60 memories, 5 queries) showed nothing
  below ~0.2 and 8 rows/query at 0.875 precision at 0.3; real BEIR corpora put
  additions above the 0.10–0.30 interval and grade them near zero precision
  under a qrels-derived graph, while an 80/20 held-out construction finds a
  small gain on nfcorpus only (+0.0019 nDCG@3). The 0.0 default is an
  engineering precaution, not a measured optimum.
- Unified query surface: `MemoryStore.query(spec)` and the MCP `query` tool
  return one ranked list from all four capabilities in one call; rows carry
  `sources` provenance; a spec smuggling `tenant` is rejected.
- Mutation-gate coverage for `graph_leg.py` (8 mutants). All 8 are now killed:
  the finalised-check mutant was first declared equivalent ("only wastes heap
  pops"), which is false under a hard pop budget — a wasted pop is one fewer
  node finalised. Fuzzing found output differences on 124/300 random multi-path
  graphs, the declaration was withdrawn, and
  `test_the_finalized_check_is_load_bearing_under_a_hard_budget` pins the
  minimal counterexample. Gate total: 38 mutants, 37 killed, 1 equivalent.

- Tenant observability for operators: `MemoryStore.tenants()` returns the home's full tenant roster with live/soft-deleted/superseded/total counts and per-kind breakdown. `omnidb-server tenants` exposes it as a read-only CLI command; this is the single-tenant-operator view, deliberately not an MCP tool (scoped tenants cannot enumerate other tenants via the server).
- Multi-tenant streamable-HTTP transport: `omnidb-server serve-http` plus a
  per-request bearer-key authenticator (`omnidb_server.tenancy`,
  `omnidb_server.http_server`; ADR-08). Each key resolves to a tenant; no tool
  accepts a `tenant_id` argument.
- `omnidb-server tenant-key` to mint one API key and print its config entry.
- Per-tenant FTS5 table lifecycle (`omnidb_server.tenant_schema`): create/drop
  atomically with a registration marker, one `fts_<tenant>` table per tenant.

### Changed

- Schema v5: the FTS5 `memories_ai/ad/au` triggers are replaced by explicit
  in-transaction FTS sync in the store write path (ADR-08 / W3-3).
- Schema v4: `tenant_id` column on `memories` and `queries`, defaulting to the
  single tenant so pre-tenant homes keep working unchanged.
- `vector_index` now converts a corrupt HNSW/tombstone sidecar to
  `VectorIndexError` (F31), so "this index is unusable" always reaches the
  documented brute-force fallback.

### Fixed

- Duplicate, unguarded HNSW index construction in `HnswlibIndex.__init__`
  (each open built the graph twice and masked the F31 guard).

### Security

- `serve-http` fails closed at startup with no tenant credentials, and a
  request with an absent/malformed/unknown key is rejected with 401 (no
  default-tenant fallback).

## [0.1.3] — 2026-09-20

<!-- 0.1.3 was assembled from four separate remediation batches, each of which
     had pasted its own Added/Changed/Fixed/Security blocks under this heading. They
     are merged here into one block per category, entry order preserved within each.
     60 entries total: 21 Added, 16 Changed, 21 Fixed, 2 Security. -->

### Added

- `bench/soak.py --workload growth --growth-cap N`: a separate monotonic
- `omnidb-server doctor` gained a read-only `recovery` check that surfaces
- Operations docs now state the lock-free diagnostic set vs lock-requiring
- `schema` MCP tool and CLI command (E-06): tables, column notes, and
- Shell-facing `omnidb-server recall` and `remember` commands (E-07) — the
- One-line serve startup banner on stderr: home, live memories, embedder,
- `omnidb-server config` (E-15): the effective configuration with each
- Chunked consolidation with `dry_run` (E-12): extraction runs one LLM call
- `omnidb-server backup DEST [--keep N]` (E-13): hot, crash-consistent
- `omnidb-server re-embed --embedder fastembed [--model ...] [--yes]`
- Query-log observability: opt-in `query_log` config records every recall/search
- Skill bundles: export/import/list procedural-skill memories as zip bundles
- Framework adapters for LangGraph and LlamaIndex wiring OmniDB in as agent
- `doctor` and `init` CLIs for diagnosing a home and bootstrapping one.
- Ollama autodetect: with `ollama_autodetect` on, consolidation probes the
- `py.typed` marker shipped in the package.
- `cold_start_ms` reported in `stats()`, plus a `OMNIDB_COLD_START_BUDGET_MS`
- README 10-second test section; performance report at `docs/perf-report.md`;
- Fusion property tests (hypothesis) covering the pure scoring surface.
- Concurrent-serve test covering multi-process contention on one home.
- LLM-load responsiveness test asserting slow LLM calls cannot stall the stdio

### Changed

- Post-commit write ordering lives in one place (`MemoryStore._after_commit`,
- `ef_search` is a config knob (`OMNIDB_EF_SEARCH`, PERF-07), honored by hnsw
- Agent-scoped recall over-fetches the dense leg by `agent_overfetch`
- Wheel ships only `omnidb_server` (DEBT-02): tests/docs/bench/scripts stay
- mypy gate is green and runs in CI (QA-05); store/recall/consolidate/server
- Sidecar persistence policy (PERF-01): the time-based flush no longer dumps
- Brute-force index stores vectors in a capacity-doubling buffer (PERF-02):
- Recall's skill-trigger scan is two-phase (PERF-03): match on
- Recall commits once (PERF-04): the access bump and the query-log row share
- Query-log retention (PERF-05): `query_retention_rows` (default 100k) prunes
- Home lock is a real OS-level advisory lock on every platform — flock on
- `purge()` hard-deletes superseded-only rows too (they were previously
- `recall` clamps `k`/`k_fetch` (max 1000) instead of crashing on SQLite's
- A corrupt hnsw sidecar now says why on stderr before falling back to the
- Version and docs sync: pyproject/package version aligned to 0.1.3;
- Server WAL-checkpoint loop goes through `MemoryStore.checkpoint()` so it

### Fixed

- `pyproject.toml`: removed the unregistered `Framework :: MCP :: Server`
- `reembed_home` now acquires the home lock BEFORE `db.connect` (which runs
- Content updates encode before the SQL mutation and record a durable
- The home lock is acquired before any writable initialization in the server
- Backup snapshots carry `index_dirty=1`; re-embedding publishes identity
- JSONL import preserves original ids and edge endpoints, validates every
- `search` budgets VM instructions (not callbacks), adds a monotonic deadline
- Trigger patterns are bounded at admission (count/length) and at matching
- Batch inserts capture each row's actual id; recall filters share one
- CI runs a required HNSW job with skips-as-failures, collects branch
- Consolidation marked every fetched episode as consumed while distilling at
- One oversized episode blocked the whole consolidation queue forever
- A failed `remember_many` (e.g. duplicate explicit id) left partial rows in
- Concurrent MCP tool calls no longer hang or crash: tool bodies share one
- The documented `OMNIDB_INDEX` environment variable was read by nothing
- `related()` no longer returns superseded memories.
- `supersede` rejects a new_id that is itself already superseded, closing
- Schema migrations are now atomic (DDL + `user_version` bump in one
- LangGraph adapter `put` is now an upsert (previously duplicated rows and
- Both framework adapters scope reads and writes to an `agent_id`
- Stale foundation test asserted schema v2 after the queries table moved the

### Security

- Consolidation facts carry `origin: "consolidation"`, `source_episodes`, and
- `remember` bounds content at `max_content_chars` (default 100k, SEC-04) and

## [0.1.3] — expanded notes

### Added

- `update_memory` tool: edit a memory's content (re-embeds and reindexes) and/or merge metadata.
- `forget_many` batch soft delete with per-id `forgotten`/`missing` reporting.
- Agent namespacing: `agent_id` column (migration v2, indexed) on write; recall, the FTS leg, the
  trigger scan, and skill triggers all scope by it; unscoped queries still see everything, and
  `stats()` breaks live counts down per agent (top 20).
- Procedural skill triggers: `trigger_patterns` regexes surface matching procedural memories in any
  recall regardless of fusion rank (`skill_trigger: true` metadata); invalid imported patterns are
  skipped, not fatal.
- Reranker seam: named rerankers behind an `omnidb_server.rerank` REGISTRY, opt-in per recall call.
- Maintenance and portability surface: `check()` integrity report (FTS integrity, index drift,
  orphan edges, WAL bytes, soft-deleted count), `purge()` hard delete with FK cascade and optional
  `VACUUM`, whole-home export/import.
- TOML configuration layer: `omnidb.toml` in CWD and home directory under CLI > env > file >
  defaults precedence; unknown keys rejected; scalar coercion to declared field types.
- Forward-only schema migrations tracked by `PRAGMA user_version` with gap detection; v1 databases
  upgrade in place without data loss.
- Test suites: hypothesis property tests for the pure scoring surface, a BFS-oracle fuzz test for
  the graph walk, chaos/crash-window reconciliation tests, a vector re-baseline benchmark gate
  (`bench/baseline.py`), and a hand-rolled mutation analysis script for the ranking modules
  (`scripts/mutation_check.py`).
- GitHub Actions CI: lint + tests across Ubuntu/Windows/macOS on Python 3.12 and 3.14, plus a
  benchmark gate job.
- `OMN-*` error codes carried by `StoreError`, extended beyond the spec where it was silent.
- Richer `stats()`: per-kind and per-agent counts, edges, embedder identity, index kind/count/
  dead_ratio, pending-compaction flag, last reconciliation time, WAL size, and all file sizes.
- `took_ms` timing field on mutating and reporting responses.

### Changed

- Vector-sidecar persistence is debounced: dirty-flag + op-count/time thresholds replace a flush
  per write; PASSIVE WAL checkpoint rides along with forced flushes.
- Hashing embedder vectorized: cached feature buckets and bulk NumPy accumulation replace
  per-text Python loops (same vectors as v1, materially faster bulk import).
- All MCP tools execute via `anyio.to_thread.run_sync` so slow embeds or LLM calls cannot stall
  the stdio event loop.
- Consolidation split into sanitize -> transcript budget -> optional LLM extraction -> fact
  validation -> source marking, with a ~24k-token episode budget per run and dedup-aware marking;
  episodes beyond budget stay retryable.

### Fixed

- WITH-wrapped data-modifying CTEs (`WITH t AS (...) DELETE ...`) bypassed the SELECT/WITH keyword
  allowlist; a deny-by-default SQLite authorizer now enforces read-only execution (the allowlist
  alone was never sufficient).
- Cyclic link graphs exploded recursive-CTE traversal; visit tracking plus a client-supplied hops
  clamp (1-8) makes `related` cycle-safe, verified against a BFS oracle.
- hnswlib deletion tracking: mark-deleted labels are mirrored to a persisted sidecar set so active
  counts, dead-ratio compaction triggers, and post-reopen query results are correct.
- Clock-skew overflow: negative memory ages (created_ms in the future) clamp to zero instead of
  inflating recency scores.
- Non-ASCII BM25: tokenization is Unicode-aware end to end (porter+unicode61 FTS5, `\w` match
  builder), keeping accented and non-Latin terms searchable.
- Windows sidecar persistence: atomic replaces retry through transient `PermissionError` from AV /
  indexer file locks instead of failing writes.
- Consolidation with an LLM configured no longer consumes episodes when the LLM call fails — they
  stay unmarked so a later run retries them; offline stub output is validated identically.

### Security

- `search` authorizer denies every action except reads (and the single FTS5-required pragma),
  closing the data-modifying-CTE hole.
- Runaway queries abort at a 5M VM step budget via a progress handler.
- Untrusted text sanitized before storage and before entering LLM prompts: control characters
  stripped, lengths capped, extraction prompt marks transcript content as data, never instructions.
