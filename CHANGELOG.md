# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] — 2026-09-20

### Fixed

- `pyproject.toml`: removed the unregistered `Framework :: MCP :: Server`
  trove classifier, which failed `python -m build` metadata validation and
  blocked the 0.1.3 wheel/sdist build (the stale 0.1.0 wheel predated it).
- `reembed_home` now acquires the home lock BEFORE `db.connect` (which runs
  migrations/WAL pragmas), closing the last writable-init-before-ownership
  path (AUDIT-I02); the pre-ownership connection is also closed on every
  early-failure path (AUDIT-I20). `examples/demo.py` uses the shared
  `open_owned_store` helper.

### Added

- `bench/soak.py --workload growth --growth-cap N`: a separate monotonic
  scaling experiment (live set grows without forgets; every recall tagged
  with the live count, reported per size bucket, no pass/fail gate) closing
  the I12 protocol gap; evidence in `bench/growth-results.txt`.
- `omnidb-server doctor` gained a read-only `recovery` check that surfaces
  pending healing work (`index_dirty`, `pending_reembed`, `reembed_pending`,
  `compaction_due`) so operators can see what the next open will reconcile
  from rows; informational only, never fails the report.
- Operations docs now state the lock-free diagnostic set vs lock-requiring
  commands, the 1 s maintenance loop, dirty-marked backups, enforced
  downgrade rejection, and in-place `re-embed`; historical coverage/perf
  reports are labeled with their post-remediation successors.

### Fixed (Sep 2026 audit remediation, `docs/AUDIT.md`)

- Content updates encode before the SQL mutation and record a durable
  `pending_reembed` revision in the same transaction; startup reconciliation
  heals same-ID content/vector divergence at every interruption boundary (I01).
- The home lock is acquired before any writable initialization in the server
  and every writable CLI command; rejected openers perform zero persistence;
  lifecycle cleanup releases each resource in its own `finally` (I02, I20).
- Backup snapshots carry `index_dirty=1`; re-embedding publishes identity
  atomically behind a durable migration marker, so interrupted migrations
  (including same-dimension model swaps) rebuild consistently (I03).
- JSONL import preserves original ids and edge endpoints, validates every
  record before the first write, and rolls back completely on rejection (I04).
- `search` budgets VM instructions (not callbacks), adds a monotonic deadline
  and a result byte cap, and cleans up handlers on all exits (I05).
- Trigger patterns are bounded at admission (count/length) and at matching
  (query size, per-recall evaluations) (I06, I21).
- Batch inserts capture each row's actual id; recall filters share one
  kind/scope contract and filter before truncation; torn v2 migrations heal
  per-column; future schema versions are rejected; consolidation facts carry
  per-chunk provenance and agent scope; recovery runs in bounded batches;
  idle dirty vectors flush within `persist_max_s`; `forget_many` builds its
  membership set once; config coercion is strict (I07–I19, I21).
- CI runs a required HNSW job with skips-as-failures, collects branch
  coverage with an 80% floor, and the soak harness is steady-state with
  revision-bound metrics (I12, I14).

### Added

- `schema` MCP tool and CLI command (E-06): tables, column notes, and
  copy-runnable example queries for the `search` escape hatch.
- Shell-facing `omnidb-server recall` and `remember` commands (E-07) — the
  real hybrid retrieval path, runnable by humans for debugging.
- One-line serve startup banner on stderr: home, live memories, embedder,
  index kind, cold start, pid (E-07).
- `omnidb-server config` (E-15): the effective configuration with each
  value's source (`default` / toml path / `env:OMNIDB_*`), `llm_api_key`
  redacted. Config values are range-validated on load and fail with a named
  key instead of misbehaving downstream.
- Chunked consolidation with `dry_run` (E-12): extraction runs one LLM call
  per `consolidation_chunk_episodes` (default 10); a failed chunk is
  reported and retried alone while successful chunks commit; dry-run
  previews what would be written (MCP tool param and `--dry-run`).
- `omnidb-server backup DEST [--keep N]` (E-13): hot, crash-consistent
  physical backup via SQLite's online backup API plus the vector sidecars,
  into timestamped snapshot directories with rotation.
- `omnidb-server re-embed --embedder fastembed [--model ...] [--yes]`
  (E-01): migrate an existing home to a new embedder in place — re-encodes
  all live rows into a fresh sidecar and re-pins the stored identity last,
  so every crash window heals through reconciliation. This replaces the
  "switching embedders requires a fresh home" limitation.

### Fixed

- Consolidation marked every fetched episode as consumed while distilling at
  most 50 facts; in offline-stub mode episodes 51+ were permanently skipped
  without ever being distilled. Marking now covers exactly the consumed
  episodes (the full transcript for the LLM path, the distilled prefix for
  the stub).
- One oversized episode blocked the whole consolidation queue forever
  (`_fit_episode_budget` broke instead of skipping); oversized episodes are
  now skipped and reported as `episodes_skipped`.
- A failed `remember_many` (e.g. duplicate explicit id) left partial rows in
  an open transaction that the next unrelated operation committed, with no
  vectors in the index. All write paths now roll back and raise a typed
  `[OMN-005]`/`[OMN-000]` error.
- Concurrent MCP tool calls no longer hang or crash: tool bodies share one
  sqlite connection, and `search`'s connection-global authorizer/progress
  handler could deadlock a concurrent write (observed as a multi-minute
  hang) while concurrent batch writes crashed on the id read-back. Every
  public store operation now runs under a store-wide reentrant lock.
- The documented `OMNIDB_INDEX` environment variable was read by nothing
  (the config layer only derives `OMNIDB_VECTOR_INDEX`); it is now honored
  as an alias, with the native variable winning when both are set.
- `related()` no longer returns superseded memories.
- `supersede` rejects a new_id that is itself already superseded, closing
  the A→B→A cycle that made both rows vanish from the live set.
- Schema migrations are now atomic (DDL + `user_version` bump in one
  transaction); a crash mid-upgrade no longer bricks the home, and homes
  already bricked by the old runner heal on the next open.
- LangGraph adapter `put` is now an upsert (previously duplicated rows and
  `get` kept returning the stale value).
- Both framework adapters scope reads and writes to an `agent_id`
  (`"langgraph"` / `"llamaindex"`); LlamaIndex `clear()` no longer deletes
  the entire store across all agents and kinds.

### Changed

- Post-commit write ordering lives in one place (`MemoryStore._after_commit`,
  ARCH-02): dirty flag first, then the index mutation, then compaction
  scheduling, then the debounced persist. All five write paths (remember_many,
  update_memory, forget_many, supersede, bulk import) share it, so a future
  write method cannot get the crash-window order wrong.
- `ef_search` is a config knob (`OMNIDB_EF_SEARCH`, PERF-07), honored by hnsw
  opens and rebuilds down to a floor of 16.
- Agent-scoped recall over-fetches the dense leg by `agent_overfetch`
  (default 3.0, PERF-08) so one dominant agent cannot crowd a scoped recall
  below k; the cap keeps placeholders bounded.
- Wheel ships only `omnidb_server` (DEBT-02): tests/docs/bench/scripts stay
  in the sdist, so installing no longer creates top-level `tests`/`docs`
  packages that collide on sys.path.
- mypy gate is green and runs in CI (QA-05); store/recall/consolidate/server
  internals are typed, and the lenient `[tool.mypy]` config is documented.

### Security

- Consolidation facts carry `origin: "consolidation"`, `source_episodes`, and
  `confidence` metadata so LLM-distilled content is auditable and filterable
  via `search` (SEC-01); `consolidation_min_confidence` (default 0.0) drops
  facts the model itself rates below a floor. README gained a threat-model
  section covering the poisoning residual risk.
- `remember` bounds content at `max_content_chars` (default 100k, SEC-04) and
  rejects non-string content with typed errors.

### Changed

- Sidecar persistence policy (PERF-01): the time-based flush no longer dumps
  the whole vector index once per slow write — it now requires a minimum of
  pending ops (`persist_min_ops`, default 8) and an interval scaled by the
  last flush's own duration, with `persist_max_s` (default 5 s) as the
  durability ceiling. Sustained single-write throughput no longer collapses
  at large index sizes.
- Brute-force index stores vectors in a capacity-doubling buffer (PERF-02):
  single-row adds amortize to O(1) instead of copying the whole matrix.
- Recall's skill-trigger scan is two-phase (PERF-03): match on
  (id, trigger_patterns) with a bounded compiled-regex cache, then fetch full
  rows only for matches; procedural content no longer moves on every recall.
- Recall commits once (PERF-04): the access bump and the query-log row share
  a single transaction, and a log-free recall with no results writes nothing.
- Query-log retention (PERF-05): `query_retention_rows` (default 100k) prunes
  the `queries` table periodically while logging; the enable flag is plain
  store state instead of a module-level id(conn) set (ARCH-03).
- Home lock is a real OS-level advisory lock on every platform — flock on
  POSIX, msvcrt byte-range on Windows (ARCH-05). It releases on process
  death, stale files are harmless, `serve` propagates its lock-conflict exit
  code (3) through the CLI, and `doctor` probes the lock instead of checking
  file existence.
- `purge()` hard-deletes superseded-only rows too (they were previously
  un-purgeable and invisible in every report — DEBT-01); `check()` reports
  a `superseded` count.
- `recall` clamps `k`/`k_fetch` (max 1000) instead of crashing on SQLite's
  variable limit (BUG-12); unknown reranker names raise typed `[OMN-002]`
  errors (QA-02); non-JSON-serializable metadata is rejected typed and early.
- A corrupt hnsw sidecar now says why on stderr before falling back to the
  brute-force backend (ARCH-04).
- Version and docs sync: pyproject/package version aligned to 0.1.3;
  operations guide lock-file name corrected to `.lock`.
- Server WAL-checkpoint loop goes through `MemoryStore.checkpoint()` so it
  cannot interleave with the search authorizer.

### Added

- Query-log observability: opt-in `query_log` config records every recall/search
  into a `queries` table (schema v3), with a `query-log` CLI dump and
  `query_stats()` aggregation (counts, avg/max latency by kind, top queries).
- Skill bundles: export/import/list procedural-skill memories as zip bundles
  (`manifest.json` + memories) via `omnidb_server.skills`.
- Framework adapters for LangGraph and LlamaIndex wiring OmniDB in as agent
  memory.
- `doctor` and `init` CLIs for diagnosing a home and bootstrapping one.
- Ollama autodetect: with `ollama_autodetect` on, consolidation probes the
  local Ollama endpoint (1.5 s budget) and adopts the first listed model.
- `py.typed` marker shipped in the package.
- `cold_start_ms` reported in `stats()`, plus a `OMNIDB_COLD_START_BUDGET_MS`
  stderr tripwire on slow opens.
- README 10-second test section; performance report at `docs/perf-report.md`;
  operations guide at `docs/operations.md`; tag-driven PyPI release workflow
  with a PR dry-run build job.
- Fusion property tests (hypothesis) covering the pure scoring surface.
- Concurrent-serve test covering multi-process contention on one home.
- LLM-load responsiveness test asserting slow LLM calls cannot stall the stdio
  event loop.

### Fixed

- Stale foundation test asserted schema v2 after the queries table moved the
  schema head to v3.

## [Unreleased]

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
