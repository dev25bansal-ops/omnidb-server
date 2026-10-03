# OmniDB — MCP-native agent memory & retrieval server

A local-first, single-binary-feel **agent memory and retrieval server over SQLite**, exposed to AI
agents through the Model Context Protocol. This is the revised v1 that came out of the August 2026
research + adversarial review (see `../OmniDB Vault/`): an MCP memory/retrieval server, not a database.

- **Substrate**: SQLite in WAL mode (`synchronous=NORMAL`, `busy_timeout=5000`) — one directory on disk,
  copyable as a unit (`omnidb.db` + WAL sidecars + vector index file). The single-file mandate was dropped.
- **Retrieval**: BM25 (FTS5, porter tokenization) fused with dense vector search via Reciprocal Rank Fusion,
  then a gentle recency/frequency rerank. Hybrid retrieval runs in-process; SQLite table-valued functions
  (`VECTOR_SIM` / `MEMORY_RECALL`) need a native extension and are the documented upgrade path for the
  future Go/Rust port.
- **Vectors**: hnswlib (M=16, efC=200) inner-product on L2-normalized vectors == cosine, persisted as a
  sidecar file; exact brute-force fallback always available. Deletions are mark-deleted.
- **Memory model**: plain relational tables (`episodic | semantic | procedural`) — no bespoke LSM, no fixed
  decay religion (weights are config). Consolidation distills episodic → semantic via a configurable LLM,
  with an offline stub when none is set.

## 10-second test

```bash
pip install omnidb-server
omnidb-server stats
# -> {"live": 0, "by_kind": {}, ...}
omnidb-server check
```

Or with MCP: point Claude Desktop at `python -m omnidb_server serve` per
[`examples/claude-desktop-config.example.json`](examples/claude-desktop-config.example.json).

## Quickstart

```bash
pip install -e ".[hnsw,semantic,dev]"   # hnsw/semantic extras optional; falls back gracefully
pytest
omnidb-server serve                     # MCP stdio server on $OMNIDB_HOME (default ~/.omnidb)
python -m omnidb_server serve           # equivalent
python -m omnidb_server stats
```

### Claude Desktop / any MCP client

See [`examples/claude-desktop-config.example.json`](examples/claude-desktop-config.example.json)
— it pins `OMNIDB_EMBEDDER=hashing` (offline-safe, no download). To opt into real
(fastembed) semantics set `OMNIDB_EMBEDDER=fastembed` plus a local `HF_HOME`
override so the ~100MB model downloads once and never again. Minimal version:

```json
{
  "mcpServers": {
    "omnidb": {
      "command": "python",
      "args": ["-m", "omnidb_server", "serve"]
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `remember(content, kind, metadata, links)` | persist a memory (episodic/semantic/procedural), optional graph links |
| `recall(query, k, kind, k_fetch)` | hybrid BM25+dense recall with recency/frequency rerank; pool floors at `k` |
| `update_memory(memory_id, content, metadata)` | edit content (re-embeds + reindexes via FTS trigger) and/or merge metadata |
| `search(sql)` | read-only SELECT/WITH escape hatch against SQLite |
| `schema()` | tables, column notes, and copy-runnable example queries for `search` |
| `related(memory_id, max_hops, label)` | undirected graph walk (recursive CTE, cycle-safe, hops clamped 1-8) |
| `forget(memory_id)` / `forget_many(ids)` | soft delete one / many |
| `consolidate(hours)` | episodic → semantic distillation |
| `stats()` | counts, edges, embedder/index info, dead-ratio, all file sizes |

Error messages carry `OMN-*` codes from the original spec's taxonomy (`[OMN-020]` read-only violation, `[OMN-021]` not found, `[OMN-006]` dimension mismatch, `[OMN-011]` timeout, ...).

## Configuration

Precedence: **CLI flags > environment > `omnidb.toml` > defaults**. The TOML file is looked up in
the CWD and the data directory (field names mirror the config dataclass):

```toml
half_life_hours = 168
rrf_k = 60
k_fetch = 50
persist_every_ops = 64      # vector-sidecar fsync debouncing
dead_compact_ratio = 0.2    # hnsw tombstone ratio triggering rebuild-on-open
```

## Operations

```bash
omnidb-server recall "deployment checklist" --home D:/data --k 5   # run the real hybrid recall from a shell
omnidb-server remember "note text" --kind episodic --agent-id a1   # persist from a shell
omnidb-server schema --home D:/data            # tables, notes, example queries (also an MCP tool)
omnidb-server config --home D:/data            # effective config + where each value came from
omnidb-server backup D:/backups --keep 3       # hot, crash-consistent snapshot (DB + vector sidecars)
omnidb-server re-embed --embedder fastembed --yes   # migrate every memory to a new embedder, in place
omnidb-server consolidate --dry-run --hours 48 # preview what consolidation would store
omnidb-server compact --home D:/path/to/data   # force full vector-index rebuild
python -m omnidb_server stats                  # includes pending_compaction, reconciled_at, WAL size
omnidb-server tenants                          # list every tenant in the home with live/soft-deleted/superseded counts (operator view, read-only)
omnidb-server check                            # FTS integrity, index drift, orphan edges, WAL size
omnidb-server purge [--vacuum]                 # hard-delete soft-deleted rows; --vacuum reclaims space
omnidb-server export D:/backup/memories.jsonl  # dump all memories + edges to JSONL
omnidb-server import D:/backup/memories.jsonl  # restore from a JSONL dump
omnidb-server serve-http --host 127.0.0.1 --port 8765  # opt-in multi-tenant HTTP MCP (loopback; refuses no-credential start)
omnidb-server tenant-key                     # generate one API key + the config entry for serve-http
omnidb-server doctor                         # environment health report as JSON (exit 4 on failure)
omnidb-server init                           # create a home dir with a config template + sample memory
omnidb-server skill export NAME out.zip      # bundle memories tagged metadata["skill"]=NAME
```

Multi-tenant `serve-http` maps a bearer key to a tenant (no tool takes a `tenant_id` argument);
single-tenant stdio runs as the implicit `default` tenant. See
`docs/adr/ADR-08-multi-tenant-rebuild.md` and `docs/MULTI_TENANCY_THREAT_MODEL.md`.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `OMNIDB_HOME` | `~/.omnidb` | data directory |
| `OMNIDB_EMBEDDER` | `auto` | `auto`/`hashing` are offline-safe (never download); `fastembed` opts into real semantics (`[semantic]` extra, ~100MB model on first run). Embedder identity is pinned per database. |
| `OMNIDB_INDEX` | `auto` | `auto` \| `hnsw` \| `brute` |
| `OMNIDB_HALF_LIFE_HOURS` | `168` | recency half-life for rerank |
| `OMNIDB_RECENCY_WEIGHT` / `OMNIDB_FREQ_WEIGHT` | `0.15` / `0.05` | rerank weights (RRF base is weight-free) |
| `OMNIDB_K_FETCH` | `50` | candidates fetched per leg before fusion |
| `OMNIDB_RRF_K` | `60` | RRF smoothing constant |
| `OMNIDB_MAX_SEARCH_ROWS` | `500` | row cap for the `search` tool |
| `OMNIDB_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | unset | OpenAI-compatible endpoint used by `consolidate`; stub when unset. With one configured, a failed call consumes nothing (episodes stay retryable) |

## Benchmarking

```bash
python bench/baseline.py   # writes bench/results.md
```

Measures hnswlib (M=16, efC=200) recall@10 vs exact brute force across an ef sweep at
384/768 dims on clustered synthetic data (1k/10k/100k), plus the SQ8 experiment.
Findings so far: at 768-dim/10k, SQ8+rerank (0.947) beats raw HNSW at ef=128 (0.837);
at 100k the clustered-synthetic distribution is strongly pessimistic for graph ANN
(tight clusters create hub structure), where quantized scoring + f32 rerank stays
robust (0.90) but at full-scan cost (~120ms). Treat as a floor, not a forecast —
re-run against your real corpus before drawing conclusions.

## Threat model

OmniDB runs stdio by default — one agent session per process, no network
listener. `omnidb-server serve-http` adds an opt-in multi-tenant transport: a
streamable-HTTP MCP listener (loopback by default) that verifies a per-request
bearer key and resolves it to a tenant, and refuses to start with no
credentials. If you enable it, that listener is an inbound surface you are
exposing — bind it accordingly. In both transports the only *outbound* call is
to the OpenAI-compatible endpoint *you* configure for consolidation
(`llm_base_url`). Data at rest is plaintext SQLite plus the vector sidecar —
use OS disk encryption for the stolen-disk threat model (see below).

Residual risk to know about: consolidation sends your episode transcript to
that endpoint and writes the model's returned facts into long-term memory.
Those facts are machine-generated and unreviewed — a prompt-injected episode
can therefore poison future recall. Facts carry `origin: "consolidation"`,
`source_episodes`, and the model's `confidence` in their metadata so they are
auditable and filterable via the `search` tool; set
`consolidation_min_confidence` to drop facts the model itself rates below a
floor, and `max_content_chars` bounds how much a single `remember` call can
store. Turn `consolidate` off (simply never call it) if your threat model
cannot tolerate unreviewed model output.

## Encryption at rest

OmniDB is local-first and writes only to `OMNIDB_HOME`; it phones home to no one. For
data-at-rest protection we recommend OS-level disk encryption (BitLocker / FileVault /
LUKS) — that covers the stolen-disk threat model without adding a dependency or a key
you must manage.

A SQLCipher swap is documented but deliberately not bundled: encrypting the SQLite file
would require `pysqlcipher3`, whose Windows wheels for Python 3.14 are effectively
unobtainable, so shipping it as an extra would break installs rather than protect them.
If your threat model needs per-file encryption beyond full-disk, compile SQLCipher and
point this package's connection at it — the store layer is plain `sqlite3` DB-API and
needs no code changes.

## Demo video
> 90-second walkthrough: `python examples/demo.py` — GIF coming soon. Until then run the demo locally per Quickstart.

## Limitations (v1, deliberate)

- Single-writer process per home directory (SQLite WAL semantics), enforced by an OS
  advisory lock: every writable opener (server and CLI) acquires the lock BEFORE any
  mutable initialization, so a conflicting process is refused without touching the DB or
  sidecars (AUDIT-I02). Read-only diagnostics (`stats`, `schema`, `export`, `query-log`,
  `query-stats`) take no lock and persist nothing, and coexist with a running server. A
  crash between a content commit and the sidecar write is healed by startup
  reconciliation (AUDIT-I22: implemented, not deferred) — on open, indexed ids and
  pending content revisions are diffed against live rows and missing/stale vectors are
  re-embedded from the rows, which remain the source of truth.
- The Python `sqlite3` module cannot register virtual tables, so SQL++-style TVFs arrive with the
  native port; until then hybrid queries go through the tools.
- Hashing embedder matches surface forms, not paraphrases — install `fastembed` for real semantics.
  Embedder identity is pinned per database; `omnidb-server re-embed --embedder fastembed --yes`
  migrates an existing home in place.
- No QPS targets: this serves one agent session locally. Gates are interactive latency, cold start,
  and RAM at scale (re-baselined empirically at true embedding dimensions, not SIFT-128 folklore).
