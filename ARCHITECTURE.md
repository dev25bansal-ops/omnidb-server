# Architecture

OmniDB is an MCP-native agent memory and retrieval server over SQLite (WAL mode + FTS5) with an
hnswlib vector sidecar. It is deliberately thin glue over reused engines — no own storage engine,
no bespoke planner. Decision history lives in `../OmniDB Vault/` (start at
`04 Verdict and Plan/Final Build Verdict.md` and `Architecture Decision Record.md`; ADR numbers
below refer to that record).

## Module map

| Module | Responsibility | Key invariant |
| --- | --- | --- |
| `config.py` | Layered settings: CLI arg > `OMNIDB_*` env > `omnidb.toml` (CWD, then resolved home; home wins) > dataclass defaults | Unknown TOML keys reject with `ConfigError`; every scalar coerces to its declared field type |
| `db.py` | Opens `omnidb.db` (WAL, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`), runs migrations, hosts the `meta` KV helpers | Forward-only `MIGRATIONS` list tracked by `PRAGMA user_version`; a version gap or a NEWER-than-supported version aborts startup with upgrade guidance (AUDIT-I17); torn v2 column additions are healed independently and schema postconditions are verified before the version bump (AUDIT-I10); FTS5 availability probed at connect |
| `embeddings.py` | `Embedder` base plus two impls: offline deterministic `HashingEmbedder` (unique-feature hashing, dim 512) and optional `FastEmbedEmbedder` (`semantic` extra) | The factory never downloads implicitly (`auto`/`hashing` are offline-safe); embedder name/dim pin into `meta` and mismatch refuses open (`OMN-006`) |
| `vector_index.py` | ANN sidecar behind one protocol: `HnswlibIndex` (inner-product space on L2-normalized vectors == cosine; deletions are mark-deleted tombstones) and exact `BruteForceIndex` fallback; `open_vector_index` selects by config and stored `vec_kind` | All persistence is atomic (tmp file, fsync, `os.replace`) with a Windows `PermissionError` retry loop; `active_count` excludes tombstones |
| `graph_leg.py` | Pure math for the graph retrieval leg: `graph_seed_masses` (seed -> restart mass on the RRF rank kernel) and `graph_expand` (max-discounted-path best-first over a capped adjacency) | Bounded by pop budget, not graph size; per-seed quotas prevent hub starvation; deterministic (id tie-break); seeds excluded from output. Mutation-gated alongside fusion/scoring |
| `query.py` (module-level: `store.query` and the MCP `query` tool) | The unified surface: one ranked result from all four capabilities — text+semantic legs fused via `run_recall`, graph reachable by `graph_from`, relational filters `kind`/`agent_id`, exact `filter_metadata`, `min_score` post-filter | A spec carrying a `tenant` key is rejected loudly; `tenant` comes from the auth context like every other tool. Rows carry `sources` (leg provenance) so a graph-surfaced row is attributable. |
| `fusion.py` | `rrf` (Reciprocal Rank Fusion, smoothing k=60) and `fts_match_query` (safe FTS5 MATCH builder) | User text is quoted into inert OR-ed tokens (max 24 terms); Unicode-aware `\w` matches the FTS5 `unicode61` tokenizer |
| `scoring.py` | Pure `final_score`: RRF base + exponential recency decay + log-frequency; the entire tuning surface | Total: finite output for any finite input; negative age (clock skew) clamps to zero; property-tested directly |
| `rerank.py` | Named rerankers behind `REGISTRY`; `apply_reranker(name, query, results)` reorders a fused candidate list | Strictly opt-in per `recall` call; runs after fusion and trigger merge, before the final slice |
| `store.py` | `MemoryStore`: all writes, hybrid reads, index reconciliation, compaction scheduling, maintenance (`check`/`purge`), stats, `OMN-*` errors | Inserts: SQL commit -> `index_dirty=1` -> index mutation -> debounced persist. Content updates encode BEFORE SQL and record the id in a durable `pending_reembed` revision queue inside the same transaction, so a same-ID update torn by a crash still heals (AUDIT-I01); recovery fetches/encodes in bounded batches (AUDIT-I13) |
| `consolidate.py` | Distills episodic rows into semantic facts: sanitize, transcript budget, optional OpenAI-compatible LLM extraction (offline stub otherwise), fact validation, source marking | A failed LLM call consumes nothing — episodes stay unmarked and retryable (`OMN-013` reported) |
| `portability.py` | Integrity audit (`check`: FTS integrity, index drift, orphan edges, WAL bytes), `purge` of soft-deleted rows (FK-cascaded, optional `VACUUM`), whole-home export/import | The home directory (db + WAL + sidecars) stays copyable as a unit; imports preserve original ids and edge endpoints, validate every record before the first write, and roll the whole transaction back on rejection (AUDIT-I04); backups carry `index_dirty=1` so restores reconcile vectors from rows (AUDIT-I03) |
| `server.py` | FastMCP server (stdio via `serve`, plus the shared tool surface + `open_owned_store` single writable entry point): tool schemas wrapping `MemoryStore`; the home lock is acquired BEFORE `build_stack` assembles config -> connection -> embedder -> index -> store; `open_read_only_store` serves diagnostics without the lock and with zero persistence | AUDIT-I02/I20: a conflicting opener is refused before any mutable initialization, and every acquired resource releases in its own `finally`; the background loop also enforces the idle persistence deadline (AUDIT-I16) |
| `tenancy.py` | Tenant identity: `TenantRegistry` hashes bearer keys and resolves them to tenants; `TenantKey`/`new_key` mint credentials | Keys are stored as SHA-256 digests, compared with `hmac.compare_digest` over every candidate (no early exit); tenant ids are validated to `[A-Za-z0-9_-]`, max 64 chars |
| `http_server.py` | Streamable-HTTP MCP transport (`serve-http`): `TenantTokenVerifier` resolves the bearer token to a tenant claim carried on the access token | Fails closed at three levels: startup refuses a credential-less registry, a bad token is a 401 (not a 500), and a tool that cannot resolve a tenant raises rather than running unscoped; no tool accepts a `tenant_id` argument |
| `tenant_schema.py` | Per-tenant FTS5 table lifecycle: `register_and_create`/`drop_tenant` plus the registration marker that is the reachability criterion | Create/drop and the `meta` marker commit in one transaction, so a half-created tenant can never be reachable; table names are `fts_<tenant>` over the validated id |
| `cli.py` | 21 subcommands: `serve`, `serve-http`, `tenant-key`, `stats`, `compact`, `consolidate`, `check`, `schema`, `recall`, `remember`, `backup`, `re-embed`, `config`, `query-log`, `purge`, `export`, `import`, `doctor`, `init`, `skill`, `query-stats` | `--home` binds identically before or after the subcommand; typed errors print and exit 2 |

## Write path

`remember(content, kind, metadata, links, agent_id, trigger_patterns)`:

1. Validate `kind` and pre-validate every link target against live rows — a bad link aborts before
   any insert, so a rejected write leaves no partial memory behind.
2. Encode content once; INSERT the row (metadata as JSON, `created_ms`, optional `agent_id`,
   `trigger_patterns`, and an importer-only explicit-id hook preserving supplied timestamps and
   access counts); `conn.commit()`.
3. Set `meta.index_dirty = 1`, then add the vector to the index. Commit-before-index is the
   deliberate ordering that makes the crash window recoverable (see below).
4. Links insert into `edges` (`INSERT OR IGNORE`, PK `(src, dst, label)`, FK `ON DELETE CASCADE`);
   `link` refuses endpoints that are missing or soft-deleted.
5. Persistence is debounced: indexed mutations increment the pending-operation counter. A
   flush is due when `persist_every_ops` (default 64) is reached; or at least `persist_min_ops`
   (default 8) are pending and elapsed time reaches `max(persist_every_s, 4 * last_flush_s)`
   (base default 0.25 s); or pending state reaches `persist_max_s` (default 5 s).
   **AUDIT-I18:** count and maximum-age triggers deliberately override adaptive time scheduling
   to limit post-crash reconstruction work, not to guarantee constant amortized write cost.
   Each snapshot writes the entire index: O(N) at fixed dimension/graph degree, with brute
   raw arrays occupying about `N * (4 * dim + 8)` bytes. A large batch may cross the count
   threshold in one transaction; it is a flush trigger, not a hard recovery-row cap.
   `flush()` forces the write and issues a PASSIVE WAL checkpoint (a reader-blocked checkpoint is
   tolerated, not an error). SQLite rows remain the recovery source of truth.
6. `forget`/`forget_many` set `deleted_ms` (soft delete, id-deduplicated batches, missing ids
   reported); `supersede` sets `superseded_by`. Both drop the ids from the index and schedule
   compaction when the hnsw tombstone ratio crosses `dead_compact_ratio` (0.2). `update_memory`
   edits flow through the FTS triggers (full-text stays synced automatically) and re-embed only
   when content actually changed.

## Read path

`recall(query, k, kind, k_fetch, agent_id, reranker)`:

1. Fetch depth is `max(k_fetch or 50, k)`. Two independent legs run: dense — embed the query and
   take ANN top-`fetch`; sparse — quote tokens into an OR-ed `MATCH` expression, rank with
   `bm25()`, and filter to live rows plus `kind`/`agent_id` in SQL.
2. `rrf([dense_ids, sparse_ids], k=60)` fuses the rankings; the top `max(4k, k)` fused ids are
   re-fetched with the same selective filters (the index may briefly lag the database).
3. `final_score` adds configurable recency decay (`exp(-age/half_life)`) and log-frequency
   (`log1p(access_count)`) on top of the weight-free RRF base.
4. Skill-trigger merge: every live procedural row whose `trigger_patterns` regex matches the query
   enters the pool regardless of fusion rank, tagged `skill_trigger: true` in metadata; invalid
   imported regexes are skipped rather than failing recall.
5. An optional named reranker (`rerank.REGISTRY`) reorders the merged pool; final order is
   `(-score, id)`, sliced to `k`.
6. Returned ids get an access bump (`access_count += 1`, `last_accessed_ms`).

`search(sql)` is the relational escape hatch: first-word allowlist (SELECT/WITH), a deny-by-default
SQLite authorizer, and a VM step budget (details under Security-relevant invariants below).

## Crash window and recovery invariants

- The only tear point is between the SQL commit and the persisted sidecar. Reconciliation-on-open
  fires only when dirty-or-mismatched: `compaction_due = 1` (scheduled rebuild takes priority),
  else a `reembed_pending` migration marker (an interrupted re-embed rebuilds vectors under the
  stored identity, covering same-dimension model swaps — AUDIT-I03), else `index_dirty = 1`, else
  a non-empty `pending_reembed` revision queue (AUDIT-I01), else sidecar active count != live row
  count (which also covers sidecar loss). A clean open performs zero re-embedding.
- Reconciliation diffs indexed ids against live SQL ids and re-embeds the union of missing ids
  and pending revisions in bounded batches (AUDIT-I13); stale ids are dropped; the healed sidecar
  persists and only then do the flag and revision queue clear. Rows are the source of truth —
  vectors are always derivable, so the sidecar is eventually durable by reconstruction, not just
  by flushing.
- Compaction is deferred, never inline: hnsw tombstones accumulate until `dead_ratio >=
  dead_compact_ratio`, which flags `compaction_due`; the flagged rebuild happens at the next open
  (fresh sidecar, chunked re-encode, tombstone-free index) or on demand via `omnidb-server compact`.

## Concurrency stance

One writer per home directory, permanently (ADR-2): SQLite WAL admits a single writer and the
sidecar is last-writer-wins, matching one server process per agent session. Inside the process,
sync store calls are offloaded to anyio worker threads so a slow embed or LLM consolidation cannot
stall the stdio event loop. Cross-process contention degrades gracefully via `busy_timeout=5000`
and PASSIVE checkpoints. Performance gates are interactive latency, cold start, and RAM — not QPS
(ADR-7); the measured baseline and its recall@10 gate live in `bench/baseline.py`.

## Security-relevant invariants

- `search` is enforced read-only by a SQLite authorizer (deny-by-default; only the `data_version`
  pragma FTS5 internals require is exempt). The keyword allowlist alone cannot stop
  data-modifying CTEs — the authorizer is the enforcement point.
- `search` bounds work three ways (AUDIT-I05): a VM-instruction budget (the progress handler
  counts callbacks of 1,000 instructions each, so the declared 5,000,000-step budget means five
  million instructions), a monotonic wall-clock deadline, and a byte cap on materialized result
  cells; authorizer and handler are cleared on every exit path. A VM-callback deadline cannot
  interrupt a long-running scalar function mid-call — hard wall-clock guarantees would require
  process isolation, which this local channel deliberately does not add.
- Procedural trigger patterns are bounded at admission (count and per-pattern length,
  AUDIT-I06/I21 in `validation.py`, enforced by creation AND import) and at matching (query-size
  and per-recall evaluation caps in `recall.py`). Python's backtracking engine has no in-thread
  cancellation that can be relied on, so the recall path refuses oversized input rather than
  abandoning a worker thread.
- Runaway queries abort at a 5M-step budget via a progress handler.
- Untrusted text is neutralized before storage or prompting: FTS MATCH syntax is quoted inert,
  consolidation strips control characters and caps lengths, and the extraction prompt declares the
  transcript data, never instructions.

## Errors

`StoreError(message, code)` carries the `OMN-*` taxonomy; the table lives at the top of
`store.py` (`OMN-002` type, `OMN-005` constraint, `OMN-006` embedder/vector-dim mismatch,
`OMN-011` aborted runaway query, `OMN-013` consolidation failure, `OMN-020` read-only violation,
`OMN-021` not found, among others). SQLite failures classify onto the same codes; the CLI prints
typed errors and exits 2.

## Extension seams

- New embedders: subclass `Embedder` (`name`, `dim`, `encode`). Identity pins per database;
  `omnidb-server re-embed --embedder ... --yes` migrates an existing home in place (AUDIT-I22:
  a different embedder no longer requires a fresh home).
- New rerankers: register in `rerank.REGISTRY`; reachable from `recall(reranker=...)` with no
  changes to SQL or index code.
- Schema changes: append `(version, ddl)` pairs to `MIGRATIONS` in `db.py`; `user_version` gating
  keeps upgrades forward-only and gap-checked.
- Native-port TVFs: `VECTOR_SIM`/`MEMORY_RECALL` as table-valued functions stay deferred to the
  Go/Rust port (ADR-4) — CPython's sqlite3 cannot register virtual tables; the tools are the
  hybrid interface until then.
- Alternate ANN backends: implement the add/remove/update/query/indexed_ids/active_count/persist
  protocol and wire into `open_vector_index`.
