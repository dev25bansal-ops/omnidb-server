# Datastore Spike: Backend Candidates for Hosted Multi-Tenancy

Status: **inconclusive by design — and the reason is measurable.**
Companion to `MULTI_TENANCY_THREAT_MODEL.md`, which must be resolved first.
Baseline reference: commit `6c99e9f`.

## Verdict up front

Postgres + pgvector is the only candidate that can satisfy the isolation requirements
without a structural workaround. But **this repo contains no data that can size that
decision.** Every benchmark present measures a single-user, single-writer process. There
is no multi-tenant number, no concurrency number, and no QPS target anywhere:
`README.md:186` states "No QPS targets" outright. Recommending a backend on this
evidence would be a guess wearing a citation.

---

## Measured facts

| Metric | Value | Source |
|---|---|---|
| Tables | 5: `meta`, `memories`, `memory_fts` (FTS5), `edges`, `queries` | `db.py:10-76` |
| Vector dimension | 512 fixed for `HashingEmbedder`; fastembed bge-small = 384; pinned in `meta.vec_dim`, mismatch is a hard failure. Not a config knob | `embeddings.py:41,74`; `store.py:190,199-212` |
| Embeddings in SQL? | **No** — no vector column exists. Sidecar only; rows are the source of truth for reconstruction | `db.py`, `vector_index.py:266-267` |
| Largest real corpus tested | fiqa 57,638 docs — hybrid recall p50 117–124 ms, p95 169–181 ms | `bench/research-v3-results.json` |
| Smaller corpora | scidocs 25,657: 70–73 ms · scifact 5,183: 20–22 ms · nfcorpus 3,633: 8.1–9.0 ms | same |
| Brute-force 512-d scaling | p50/p95 ms: 0.99/1.92 (110→500 rows) · 1.87/2.69 (1k→2k) · 2.93/3.76 (2k→4k) | `bench/growth-results.txt` |
| 60 s soak, 110 live rows | 549 ops/s · recall p50 6.70 / p95 12.99 ms · RSS 39.6→69.4 MB · WAL peak 2.27 MB · **788 flushes = 5.84 s of 60 s** · p95 creep ratio 4.39 (gate fail) | `bench/soak-results-v2.txt` |
| HNSW recall@10 | 1.0 @1k → 0.888 @10k → 0.519 @100k (384-d, synthetic clustered) · 768-d @100k: 0.395 | `bench/results.md:12-32` |
| Quality vs baselines (nDCG@10) | hybrid beats bm25s/pyserini on all 4 sets (nfcorpus 0.3747 vs 0.3047; fiqa 0.3922 vs 0.2380) but **loses to dense-only on scidocs** (+0.0110, p=0.0102) | `research-v3-results.md` |
| Cross-encoder rerank on hybrid | never helps: best +0.0067 (n.s.), worst −0.0541 (p<0.0001) | `rerank-window-results.md:176-197` |
| Fusion weights | per-corpus: ~0.7 (nfcorpus) to ≤0.3 (scidocs) | `research-v3-results.md` |
| Caps | `k_fetch`=50 · max recall k 1000 · `max_search_rows`=500 · content 100k chars · query retention 100k rows · search 5 s / 5M VM steps / 32 MiB · hops clamp 1–8, `LIMIT 200` · FTS tokens ≤24 · `agent_overfetch` 3.0 | `config.py:72-100`; `recall.py:15,7`; `store.py:32-33,822`; `fusion.py:34` |
| Trigger budget | 32 patterns × 512 chars admitted; ≤2048 pattern evals per recall; queries >4096 chars skip triggers | `validation.py:14-15`; `recall.py:29-30,141` |
| Consolidation | 60 s httpx timeout per chunk; ≤500 episodes / 24k transcript tokens; chunks of 10; ≤500 facts per run; dedup pool 500 @cos 0.92 | `consolidate.py:17,22,53,112,146,23` |
| Perf gates | 1200 writes, ≥250 w/s, recall p50 <25 ms, p95 <80 ms — **skipped unless `OMNIDB_RUN_PERF=1`**, and CI never sets it | `tests/test_perf_smoke.py:23-28`; `.github/workflows/ci.yml:22-33` |

## What a backend must preserve

1. **BM25 rank identity, not just hits.** Retrieval depends on `MATCH` + `ORDER BY bm25()`
   with porter+unicode61 (`db.py:29-34`, `recall.py:76-83`), and `fusion.py:10-31`
   consumes *ranked id lists*. Postgres `ts_rank_cd` is not `bm25()`; the ordering will
   differ. `db.py:98-100` currently refuses to open without FTS5.
2. **Filtered ANN.** The live-row predicate `deleted_ms IS NULL AND superseded_by IS NULL`
   plus kind/agent must apply *inside* the vector search (`store.py:25`,
   `recall.py:78-82`). `agent_overfetch` exists only because today's global-ANN-then-filter
   degrades scoped recall.
3. **Recall is a write path.** It bumps `access_count`/`last_accessed_ms` and commits
   (`recall.py:218-233`). Read-replica routing silently breaks scoring. This is easy to
   miss in a Postgres design.
4. **Regex scan over procedural rows** in Python (`recall.py:144-171`) — needs an indexed
   scalar scan or `pg_trgm` equivalent, and it is the ReDoS surface (A10).
5. **Recursive undirected graph walk with cycle guard** (`store.py:830-854`). Real
   feature, **zero latency data**.
6. **Embedder identity pinning and re-embed migration** (`store.py:230-235`;
   `portability.py:117-267`). Same-dimension swaps are currently undetectable
   (`AUDIT-I03`).
7. **Migration discipline**: refuses newer schema, gap check, in-transaction
   postcondition verification (`db.py:107-143`).

## What is not known

No multi-tenant or per-process concurrency measurement exists — only single-writer plus
one two-process lock test (`tests/test_concurrent_serve.py`). No QPS target. The 0.519
recall@100k figure is ANN-only, on synthetic clustered data, and **never passed through
`recall()`**; the v3 bench substituted `BruteForceIndex`
(`bench/research_bench_v3.py:272`). Nothing above 57,638 docs or 4,000 live memories was
ever tested. The "qdrant" rows are `QdrantClient(path=...)` **local file mode**
(`research_bench_v3.py:363`) — a networked vector DB has never been measured here, and
even its local numbers swung 97→390 ms across processes. No real-LLM consolidation
latency (tests stub at 300 ms). No graph latency, no cost, no per-tenant disk footprint,
no backup/restore timing, no multilingual quality.

## Candidate assessment

**Postgres + pgvector — conditionally suitable.**
Satisfies concurrency, filtered ANN, tenant-partitioned index, recursive CTE, and RLS.
The sidecar's purpose is reconstruction rather than durability (`config.py:79-85`), so
removing it is a gain, not a loss. Open risks: BM25 rank parity is untested and probably
not exact; per-tenant fusion-weight tuning is a real operational burden given the
0.7↔≤0.3 spread; `pgbouncer` transaction pooling will leak `app.tenant_id` unless
`SET LOCAL` in an explicit transaction is used.

**Postgres + external vector DB — fails on evidence.**
Adds a second system whose consistency the current design already solves for free
(SQLite WAL + `_reconcile_index` + `pending_reembed` dirty flags). Dual-write atomicity
between rows and vectors becomes a new failure mode owned by us. A networked deployment
has zero measurements in this repo. Justified only if corpus size escapes pgvector's
practical range — which is exactly the thing nobody has tested.

**Per-tenant SQLite on object storage — worst fit for a hosted service.**
Closest to current semantics, and per-tenant footprint is genuinely tiny (69 MB RSS,
2.3 MB WAL at 110 live rows). But the unit of tenancy is a home directory guarded by
`flock` (`server.py:197-232`), which forces one process per tenant: N tenants = N
processes. Object storage breaks the `os.replace` atomicity the sidecar relies on
(`vector_index.py:237-246` exists specifically for Windows AV lock contention), and the
sidecar rewrite is O(N) per flush — already 5.84 s of a 60 s soak at 110 rows, against a
gate it failed (p95 creep 4.39). SQLite-over-FUSE plus WAL is a well-known corruption
story. Defensible for self-hosted-at-tiny-scale; not a hosting architecture.

## Experiments required before deciding

Each is small, and each unblocks a specific choice:

1. **Concurrency ceiling.** Same corpus, 1 / 8 / 64 concurrent clients, `recall` p95 and
   error rate, SQLite-WAL vs Postgres. *Unblocks: whether SQLite could ever serve a tenant.*
2. **BM25 rank parity.** Run `recall()` on the four real corpora against Postgres text
   search; compare rank correlation and nDCG@10, not just overlap. *Unblocks: whether the
   retrieval quality in `research-v3-results.md` survives the migration at all.*
3. **pgvector at real dimension, filtered.** 512-d, tenant-filtered, 100k and 1M rows,
   recall@10 and p95 — through `recall()`, not the raw index. *Unblocks: pgvector vs external DB.*
4. **Graph latency.** `related`/`link` p50/p95 at realistic density, SQLite vs recursive CTE.
   *Unblocks: whether the graph feature survives.*
5. **Flush cost.** `access_count` write amplification under concurrency.
   *Unblocks: whether recall must stop being a write path.*
6. **Enable the perf gates in CI** (`OMNIDB_RUN_PERF=1`) — they already encode the SLO
   nobody is measuring against.

Experiment 2 is the highest-risk one: if BM25 parity fails, this is not a backend swap
but a retrieval-quality regression, and the whole migration premise needs revisiting.

## Provisional recommendation

Target **Postgres + pgvector**, treat **external vector DB as unjustified until
experiment 3 shows pgvector escaping its range**, and **reject per-tenant SQLite for
hosting**. Do not commit to it until the tenant definition in
`MULTI_TENANCY_THREAT_MODEL.md` §9 is answered and experiments 1–3 have numbers.
