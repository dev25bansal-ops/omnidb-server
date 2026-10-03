# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
