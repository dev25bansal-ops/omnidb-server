# A2 · Performance Bottleneck Catalog

**Agent:** A2 · **Scope:** performance bottlenecks with measurable impact metrics
**Framing (ADR-7):** the gates are **interactive latency, cold start, and RAM — not QPS**.
One agent session per process. Throughput matters only where it degrades *interactive
experience* or *post-crash reconstruction*. A raw ops/s figure is not an impact statement and
is never used as one below.

**Method discipline.** Every number is tagged:
- **[M]** measured — the file and line are named.
- **[D]** derived, not measured — the arithmetic is shown inline.
- **[U]** unmeasured — stated as such. **No number in this document is invented.**

**Sources read in full:** `1_ISSUES_REQUIRED_FIXES.md` §2, `GROUND_TRUTH.md` §5,
`7_CROSS_VERIFICATION.md` §2, `bench/results.md`, `bench/soak-results-v2.txt`,
`bench/soak-results.txt`, `bench/growth-results.txt`, `bench/baseline.py`, `bench/soak.py`,
`bench/comparative-results.md`, `bench/research-v3-results.md`, `bench/rerank-window-results.md`,
`docs/perf-report.md`, `docs/DATASTORE_SPIKE.md`, `docs/coverage-report.md`, and the runtime
modules `store.py` / `vector_index.py` / `recall.py` / `portability.py` / `config.py` / `server.py`.

**Counts:** 3 × P0 · 5 × P1 · 4 × P2 · 2 × P3 = **14 findings**, plus **21 measurement gaps**.

---

## §0 · The measured detail table

### 0.1 Recall latency (p50 / p95)

| Condition | p50 ms | p95 ms | Source |
|---|---|---|---|
| Soak v2, 60 s, brute, 512-d, 110 live | **5.255** | **17.625** | `soak-results-v2.txt:46-47` |
| Growth band 110→500 rows | 0.994 | 1.919 | `growth-results.txt:25-26` |
| Growth band 500→1,000 | 1.762 | 2.311 | `growth-results.txt:28-29` |
| Growth band 1,000→2,000 | 1.866 | 2.689 | `growth-results.txt:31-32` |
| Growth band 2,000→4,000 | 2.933 | 3.761 | `growth-results.txt:34-35` |
| Growth band 4,000→8,000 | 3.392 | 3.317 | `growth-results.txt:37-38` — **n=2, not a distribution** |
| HNSW 384-d/100k ef64 (**shipped**) | 0.374 | 0.864 | `results.md` |
| HNSW 768-d/100k ef64 (**shipped**) | 0.562 | 1.101 | `results.md` |
| HNSW 384-d/100k ef128 | 0.536 | 0.934 | `results.md` |
| HNSW 768-d/100k ef128 | 0.770 | 1.710 | `results.md` |
| Historical, <1k rows, pre-remediation | 0.37 | — | `perf-report.md:22` — **self-labelled historical; not reproducible from any current artifact** |

### 0.2 Write throughput

| Metric | Value | Source |
|---|---|---|
| `remember` p50 / p95 (soak v2) | **0.254 / 0.649 ms** | `soak-results-v2.txt:43-44` |
| `remember` p50, pre-remediation, 500 seq inserts | 3.8 k writes/s | `perf-report.md:20` (historical) |
| `remember_many(1000)` | ~29 k writes/s, ~34 ms | `perf-report.md:21` (historical) |
| `forget_many` p50 / p95 | 0.261 / 6.347 ms | `soak-results-v2.txt:49-50` |
| Overall soak rate | 548.91 ops/s over 32,942 ops | `soak-results-v2.txt:15-16` |
| Growth-workload rate | 1,242.95 ops/s over 4,668 ops | `growth-results.txt:13-14` |
| `insert_batch_per_sec` (2k corpus) | 6,756.1 | `comparative-results.md:12` |
| `insert_single_ms` (2k corpus) | 0.336 | `comparative-results.md:11` |

**[D]** `1 / 0.254 ms = 3,937 remember/s`, corroborating the historical 3.8 k/s within 3.6% —
**the only place a doc number and a bench number independently agree.**

### 0.3 Brute-force scaling (512-d, `growth-results.txt`)

p50 0.994 → 1.762 → 1.866 → 2.933 → 3.392 ms as live rows go 110→500→1k→2k→4k→8k.
**[D]** A 72.7× size increase costs 3.41× the p50 — **sub-linear**, because the scan is one
vectorised NumPy matmul. Per-band ratios (4.5× size → 2.97× time; 2× → 1.77×; 2× → 1.06×;
2× → 1.57×) are noisy — the 2,000→4,000 band has 400 samples, the 4,000→8,000 band has **2**.
**There is no measurement of brute-force recall p95 above 8,000 live rows.**

### 0.4 `ef_search` 64 → 128 — quality per millisecond

| Cell | recall@10 64→128 | Δrecall | p50 Δms | **[D] recall per ms** |
|---|---|---|---|---|
| 384-d / 10k | 0.695 → 0.888 | +0.193 | +0.027 | **7.15** |
| 768-d / 10k | 0.724 → 0.854 | +0.130 | +0.109 | **1.19** |
| 384-d / 100k | 0.378 → 0.519 | +0.141 | +0.162 | **0.87** |
| 768-d / 100k | 0.278 → 0.395 | +0.117 | +0.208 | **0.56** |

`config.py:100-102` cites "the benchmark gate shows ef=128 materially better at 10k rows" and
then **ships `ef_search = 64`**. **The citation is the code's own counter-evidence.** Best
quality-per-ms is 7.15 at 10k/384-d, worst 0.56 at 100k/768-d — a **12.8× spread**, i.e. the
same 30-minute change is worth 2.6× more quality at 10k than at 100k.

### 0.5 recall@10 at 1k / 10k / 100k (`bench/results.md`)

| dim | ef | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| 384 | 32 | 0.958 | 0.563 | 0.278 |
| 384 | **64 (shipped)** | **0.990** | **0.695** | **0.378** |
| 384 | 96 | 0.997 | 0.819 | 0.462 |
| 384 | 128 | **1.000** | **0.888** | **0.519** |
| 768 | 32 | 0.924 | 0.570 | 0.170 |
| 768 | **64 (shipped)** | **0.985** | **0.724** | **0.278** |
| 768 | 96 | 0.992 | 0.792 | 0.346 |
| 768 | 128 | **0.997** | **0.854** | **0.395** |

Build cost **[M]**: 384-d 0.01/0.13/5.48 s; 768-d 0.03/0.21/12.01 s at 1k/10k/100k.
**[D]** recall@10 falls 1.000 → 0.519 (−48%) from 1k to 100k at 384-d, while p50 rises only
0.070 → 0.536 ms. **The user pays nothing in latency to get a 48% worse answer** — there is no
visible signal that quality collapsed. That is the sharpest form of this finding.

### 0.6 HNSW build (a cold-start term)

**[M]** 12.01 s to build 768-d/100k. This is a **startup-path** cost: `rebuild_index` rebuilds
from scratch. At 100k live rows a crash that loses the sidecar costs **12.01 s of blocked
startup**. No measurement exists for the `store.py:949` chunked re-encode wrapping it.

---

## §1 · Findings

### PERF-A2-01 — O(N) sidecar persist with an unconditional 64-op trigger
**Location:** `store.py:297-313` (`_maybe_persist`), `:332-338` (`flush`), `vector_index.py:114-125`, `config.py:90-98`
**Severity: P0** — the only finding that degrades all three ADR-7 gates at once, and the highest-certainty item here: the *bytes* follow from the file format, not from a benchmark.

**The three triggers and the precedence problem.** `store.py:309-312`:
```python
if (self._pending_ops >= self.cfg.persist_every_ops              # 64
        or (self._pending_ops >= self.cfg.persist_min_ops        # 8
            and elapsed >= interval)                             # adaptive
        or elapsed >= self.cfg.persist_max_s):                    # 5.0
```
These are **disjunctive**, so whichever fires first wins. Three consequences:

1. **AUDIT-I18 lets the op-count trigger override adaptive time control** (documented at `config.py:92-98`). The rationale — bound dirty-vector age for post-crash reconstruction — is sound. But the PERF-01 adaptive control, added *because* "a sidecar persist is a full O(N) dump", **can never engage when the op-count trigger is the binding one** — and at any corpus where a single flush is fast, the op-count trigger is always binding.
2. **The adaptive branch is dead code below a 62.5 ms flush.** `interval = max(persist_every_s, 4.0 × _last_flush_s)` with `persist_every_s = 0.25`. The adaptive term only wins when **[D] `last_flush_s > 62.5 ms`**. Measured: `3,761.719 / 786 = 4.786 ms` per flush **[M]** → `max(0.25, 0.0191) = 0.25 s`, inert. Growth run: `596.675 / 61 = 9.781 ms` **[M]**, still inert. **The PERF-01 mitigation is dead code at every measured corpus size.**
3. **The time triggers can only ever *increase* flush count.** Since the op-count branch bounds flushes at one per 64 mutations, every other knob in `config.py:90-98` *adds* cost without reducing it.

**Write amplification.** `vector_index.py:123-125` writes `ids` (int64) + `matrix` (float32) in full, every time, with no delta path — the docstring at `:115` says so. Snapshot = `N × (4·D + 8)` bytes.

| Corpus | Snapshot **[M/D]** | Bytes per logical write at 64-op trigger **[D]** | Amplification vs the row's own `4D+8` bytes **[D]** |
|---|---|---|---|
| 4,000 × 512-d | 4,000 × 2,056 = **8.22 MB** | 8,224,000 / 64 = **125.5 KiB** | **32.1×** |
| 100,000 × 768-d | 100,000 × 3,080 = **308 MB** | 308,000,000 / 64 = **4.59 MiB** | **48.1×** |

**[D] Workload-shape sensitivity.** For a batch of `m` rows the op counter advances by `m` (`store.py:329`), so flush count is `m/64`, each a full snapshot:
- `remember_many(1000)` at 100k/768-d: `15.625 × 308 MB = 4,812.5 MB ≈ 4.81 GB` written to durably store 3.08 MB of new vectors → **1,563× amplification**.
- Same call at 4k/512-d: `15.625 × 8.22 MB = 128.4 MB` to store 2.06 MB → **62.4×**. **Amplification grows ~25× from 4k/512-d to 100k/768-d.**
- *Caveat, stated plainly:* I could not verify `remember_many` passes `n_ops = len(rows)` rather than a smaller count. **If it passes 1, the batch collapses to one flush and the 1,563× figure is wrong by ~64×.** The per-op figures above do not depend on this.

**Consistency check that flags the problem.** The historical ~29 k writes/s is incompatible with 15.6 × 308 MB flushes inside a 34 ms batch (that would be 2.2 GB/s sustained). **Both numbers cannot hold at 100k/768-d** — itself evidence the amortized cost is untested at scale.

**[D] What that means for the interactive write path.** From the growth run, implied sidecar write bandwidth is `8.22 MB / 9.781 ms = 840 MB/s`. At that rate a 308 MB flush takes `0.3667 s`, so each of the 64 writes in a flush window absorbs **5.73 ms** — versus a measured `remember` p50 of **0.254 ms**. **A `remember` at 100k/768-d would cost ~22.6× its measured small-corpus cost, on the agent's tool-loop critical path, under the store-wide lock.** *Caveat:* 840 MB/s was measured on an 8.22 MB snapshot that fits in page cache; a 308 MB write likely will not, so **this is a lower bound, not an estimate.** Derived, not measured — no flush has ever been timed above 4,000 rows.

**What is actually measured today.** In the 60 s soak at 110 live rows: **786 flushes** over 32,942 API calls = **1 flush per 41.9 API calls** **[M/D]**; total sidecar bytes `786 × 226,160 = 177.8 MB` over 60.013 s = **2.96 MB/s** **[D]**; time in flush `3,761.719 / 60,013 = 6.27%` of wall clock **[M/D]**. **Honest verdict: at the only corpus size anyone has soaked, this is a 6.3% tax, not a crisis.** The P0 rests on the arithmetic, which is certain, plus the absence of any measurement above 4,000 live memories.

**Why it matters — the user-visible scenario.** An agent session writes memories interleaved with reads. Each 64th `remember` blocks for a full-corpus rewrite *while holding `self._op_lock`*. At 100k/768-d that is a multi-hundred-millisecond hitch mid-tool-call, on the exact metric ADR-7 gates. The secondary victim: write volume competes with the WAL, and `wal_max_mb: 2.267` **[M]** already sits against an 80 MB gate.

**Effort.** Size-aware policy: **0.5–1 day**. Incremental/WAL sidecar: **1–2 weeks**. **Timeline:** size-aware is same-tranche as PERF-A2-10 (both <1 day).

**Fix.** Trigger on `pending_ops × N × (4D+8) ≥ budget_MB` rather than a raw op count, and make the time triggers a floor that cannot *add* flushes. Expected: write amplification at 100k/768-d falls from **4.59 MB/op toward ~3 KB/op, a ~1,500× reduction.** **Caveat: the byte reduction is arithmetic; the ms reduction is unmeasurable until gap #13 is closed.**

---

### PERF-A2-02 — No quantized backend: shipped `ef_search=64` returns 2.8 of 10 correct neighbours at 100k
**Location:** `vector_index.py` (whole file), `config.py:100-106`, `bench/baseline.py:36-40`
**Severity: P0** — the dense leg is silently wrong at scale, and nothing in the shipped configuration surfaces it.

**Confirmed by exhaustive grep.** A case-insensitive search for `int8|quant|SQ8|sq8|mmap|float16` across the **entire** `omnidb_server/` package returns **0 matches** — not merely absent from `vector_index.py`, but absent from the runtime. `sq8_quantize` exists only at `bench/baseline.py:36-40`, inside the benchmark.

**What SQ8 would buy, from the benchmark's own SQ8 arm** (`results.md`, 24-cell sweep):

| Cell | shipped (ef64) | SQ8 | Δrecall | shipped p50 | SQ8 p50 | **[D] latency ratio** |
|---|---|---|---|---|---|---|
| 384-d / 100k | 0.378 | **0.881** | **+0.503** | 0.374 ms | **59.162 ms** | **158×** |
| 768-d / 100k | 0.278 | **0.896** | **+0.618** | 0.562 ms | **130.520 ms** | **232×** |
| 384-d / 10k | 0.695 | 0.890 | +0.195 | 0.132 ms | 5.524 ms | 42× |
| 768-d / 10k | 0.724 | 0.947 | +0.223 | 0.184 ms | 11.621 ms | 63× |
| 384-d / 1k | 0.990 | 0.967 | **−0.023** | 0.048 ms | 0.543 ms | 11× |
| 768-d / 1k | 0.985 | 0.977 | **−0.008** | 0.106 ms | 1.109 ms | 11× |

**Three findings the compressed version missed:**

1. **SQ8 is strictly dominated in 4 of its 6 cells.** At 1k it is *worse on quality* (0.967 vs ef128's 1.000) *and* 11× slower. At 10k/384-d, ef128 alone gives 0.888 @ 0.159 ms versus SQ8's 0.890 @ 5.524 ms — **equal recall for 34.7× the latency.** **[D] SQ8 is worth its cost in exactly 2 of 6 cells, both at n=100,000.** Any adoption must gate on corpus size, not a global flag.
2. **The 768-d/100k SQ8 latency of 130.5 ms exceeds the entire measured hybrid recall p50 on the largest real corpus tested** — fiqa, 57,638 docs, **116.98–123.78 ms** across 5 processes. A quantized shortlist scan of 100k 768-d vectors costs the same as a full hybrid retrieval over a real 57.6k collection. **A direct conflict with the ADR-7 interactive gate.**
3. **The SQ8 arm has no p95.** `results.md` reports SQ8 at **p50 only** — the doc's own summary confirms it "does not report `ef_search`, build time, p95 latency, or the underlying f32-rerank candidate multiplier." **The only candidate path to ≥0.95 recall@10 at 100k is measured at the median and blind at the tail** — and the tail is the gate.
4. **[D] SQ8 as benchmarked is not a memory win.** `baseline.py` computes `quantized.astype(np.int32)` *inside* the per-query timing loop: at 100k/768-d that is `100,000 × 768 × 4 = 307.2 MB` allocated per query, alongside the 307 MB f32 matrix. **The benchmark allocates more RAM than it saves, inside the latency window.**

**Why it matters.** An agent with 100k memories asks a question. The dense leg returns **2.78 of 10** correct neighbours at 768-d **[M]**. Those 7.2 wrong neighbours are not rejected — they are *fused* by RRF with lexical hits and returned with a `score` rounded to 6 places. The agent reads confident, wrong memories, at **0.562 ms p50 — the user feels nothing at all.** A silent 72% retrieval error with a sub-millisecond latency signature is the worst failure shape in this catalog.

**Effort.** 3–5 days for a correct SQ8 backend (a real shortlist index, not a full `astype(int32)` scan). **Dependencies:** PERF-A2-03 is a prerequisite for a memory-motivated version. **Timeline:** 1-month tranche at the earliest — it needs a new benchmark to be measurable at all.

**Fix.** Ship `ef_search = 128` immediately (**+0.117 to +0.193 recall@10 for +0.027 to +0.208 ms**). That is the cheap half, available today. For the expensive half: int8 storage with a real ANN shortlist, corpus-size-gated. **Caveat: the SQ8 numbers are p50 on synthetic clustered data, exclude build time and rerank fan-out, at a 50-candidate shortlist. They are an upper bound on quality, not a forecast.**

---

### PERF-A2-03 — `.npz` cannot be memory-mapped, and `load()` makes two full copies
**Location:** `vector_index.py:33-47` (`BruteForceIndex.load`), `:114-125` (`persist`)
**Severity: P1** — the only finding that directly attacks the RAM gate.

**Why it cannot be mmap'd, precisely.** `persist` uses `np.savez` (`:123`), which writes a ZIP archive; members are **STORED** (numpy does not deflate by default), so the bytes *are* mmap-able at a computed offset. But `load` at `:37-38` does `with path.open("rb") as handle: data = np.load(handle)` — passing a **file object**. `np.load` can only use `mmap_mode` with a real filesystem path, and `mmap_mode` only applies to uncompressed `.npy`. So the STORED-member property that would have made it mmap-able is wasted. `HnswlibIndex` has the same shape at `:210+`.

**The two copies, counted exactly:**
- `data["matrix"]` → allocates array **A** (`N × D` f32) when the zip member is read.
- `A.astype(np.float32)` → allocates array **B**. `astype` defaults to `copy=True`, so it copies even though the dtype is unchanged.
- `np.ascontiguousarray(B)` at `:45` returns **B itself** — B is already C-contiguous f32. **This is not a third copy.** *(The compressed version of this finding implied a third; it is two.)*
- `A` stays alive until `load()` returns, so both exist simultaneously.

| Cell | resident **[D]** | peak (2×) **[D]** |
|---|---|---|
| 100,000 × 768-d | 307.2 MB | 616.0 MB |
| 100,000 × 384-d | 153.6 MB | 308.8 MB |
| 1,000,000 × 768-d | 3.07 GB | **6.15 GB** |

Against the soak's RSS-growth gate of **300 MB** (`soak.py:51`), a single 768-d/100k open **exceeds the entire growth budget at rest**, before the agent stores anything. The measured soak RSS is 38.777 → 68.363 MB **[M]** at 110 live rows — **precisely why this was never noticed: the only RAM measurement in the repo is taken at a corpus size 900× smaller than the one that breaks.**

**Why it matters.** The user-visible moment is the *first* `recall` after starting a session against a large home: the process is at ~616 MB peak before answering anything, on a product whose own gate is 300 MB. On a laptop alongside an IDE and a browser this is the difference between "the memory tool is fast" and "the memory tool makes my machine swap."

**Effort.** 6–10 hours: write `vectors.npy` (uncompressed, path-based) + a separate `ids.npy`, load with `np.load(path, mmap_mode="r")`; fall back to the `.npz` reader for existing homes. **Dependencies:** `open_vector_index` must gain a `.npy` branch; `rebuild_index` (`store.py:946`) constructs `BruteForceIndex(home / "vectors.npz", dim)` and must follow. **Timeline:** 2-week tranche; unblocks the memory-motivated part of PERF-A2-02.

**Fix.** mmap + one copy: peak `616 MB → ~308 MB` at 100k/768-d (**−50%**), and steady-state resident drops to the OS page cache, so pages are evictable under pressure instead of pinned. **Caveat: the latency effect is unmeasured and its sign is not obvious** — mmap trades a page-fault storm on first touch for lower steady-state RSS, and for a short interactive session the page faults may dominate. **Benchmark, do not assume.**

---

### PERF-A2-04 — `import_jsonl` encodes the entire corpus in one call while its siblings chunk at 512 and 256
**Location:** `portability.py:511-516`, versus `store.py:949` (`chunk = 512`) and `portability.py:261` (`chunk = 256`)
**Severity: P1** — the one bulk-restore path with no bound at all, on the operation an operator runs during a migration.

**The inconsistency, verbatim.** `portability.py:514-516`:
```python
# Encode before any mutation so a failed import leaves the home untouched.
contents = [r["content"] for r in memories]
vectors = store.embedder.encode(contents)
```
`memories` is the **fully materialised** parsed record list. So the entire JSONL is resident as Python dicts, *and* a second full list of content references, *and* the full vector matrix, before a single row is written. The two sibling bulk paths both chunk, at **512** and **256**. `import_jsonl` is the **only** unchunked bulk encode in the runtime. The encode-before-mutate ordering is a deliberate correctness property (AUDIT-I04), so the fix must chunk *while preserving* it.

**Peak RAM on a multi-GB restore:**

| Component | 1M rows × 768-d | Basis |
|---|---|---|
| `vectors` (encode output) | **3.07 GB** | **[D]** `1e6 × 768 × 4` — assumption-free, it is the f32 matrix size |
| `contents` list | 8.0 MB | **[D]** `1e6 × 8` pointers; strings shared with `memories`, not copied |
| parsed `memories` dicts | **assumption-dependent** | the only content-size measurement is `content_bytes_mean: 88.67` **[M]**, but that is the *synthetic* hashing workload. At 88.67 B: 88.7 MB. At a realistic 1 KB agent memory: **~1.2 GB** |
| HNSW during `resize_index` | **assumption-dependent** | **[D]** cap-doubling reallocates; at 1M ≈ 3.2 GB final, **~4.8 GB peak** |
| **Lower bound (no assumptions)** | **3.07 GB** | vectors alone |
| **Realistic total peak** | **~9.1 GB** | 1.2 + 3.07 + 4.8 — **derived, highly assumption-sensitive** |

The assumption-free number is the one that matters: **3.07 GB, 10.2× the entire 300 MB RSS gate, from the encode call alone, before any index or record overhead.** The final `np.savez` then writes another 3.08 GB to disk.

**Why it matters.** This runs during a migration or a restore-from-backup — already the operator's worst hour, on a laptop, possibly with a backup file also mapped. An OOM kill here leaves a half-restored home. The operator experiences it as "OmniDB crashed while restoring," not as "one function was unchunked."

**Effort.** 2 hours — wrap the loop in `range(0, len(memories), 256)` and accumulate. **Dependencies:** must keep the encode-before-mutate ordering, so rows are staged, not streamed. **Timeline:** 2-week tranche; 2 hours is the cheapest item in this catalog.

**Fix.** Chunking caps the *encode* term at `256 × 768 × 4 = 786 KB` instead of 3.07 GB. Total peak drops from ~9.1 GB to **~1.2 GB (records) + ~4.8 GB (index doubling) — still dominated by `resize_index`**, so chunking alone does not fix a 1M-row restore. **Caveat: [D], not measured. No restore has ever been run at 1M rows (gaps #10/#15).**

---

### PERF-A2-05 — The dense leg is global-ANN-then-filter; `agent_overfetch=3.0` is an unmeasured, untested constant
**Location:** `recall.py:59-68`, `config.py:103-106`
**Severity: P2** — silent recall degradation in a multi-agent home, with no test.

**The code.** `recall.py:59-68` computes `fetch = min(max(k_fetch, k), _MAX_RECALL_K)` (50), and if `agent_id is not None and agent_overfetch > 1.0`, `fetch = min(_MAX_RECALL_K, int(fetch * agent_overfetch))` (150). `recall.py:70-71` then applies `AND (? IS NULL OR m.agent_id = ?)` — **the filter is after the ANN has already returned its global top-150.** With default `k=5`, a scoped recall asks the global ANN for 150 and hopes ≥5 survive.

**How much 3.0 recovers vs an ideal pre-filter. [D] arithmetic, no measurement exists.** If the scoped agent owns a fraction `f` of live rows, expected surviving dense candidates is `150 · f`. To return `k=5` you need **`f ≥ 5/150 = 3.33%`**:
- ≤30 agents sharing a home: over-fetch is **sufficient**.
- **~30 agents: exactly 5.0 survivors — at k.** Below that the dense leg silently returns fewer than k and the hybrid degenerates to **lexical-only**.
- 100 agents (`f=1%`): **[D] 1.5 survivors** against k=5.
- 1,000 agents (`f=0.1%`): **[D] 0.15 survivors** — the dense leg is *off*.

**This re-creates the repo's own measured failure mode.** `comparative-results.md:23-29` measured, as a *negative finding*, that with a weak embedder the dense leg alone scores recall@5 = **0.968** while the **full hybrid scores 0.416** — "RRF's equal-weight fusion lets the lexical leg outvote correct dense evidence." So losing the dense leg is not "falling back to something worse": under the documented conditions, **lexical-only scores ~0.416 while the dense evidence sitting in the index scores 0.968.** And `config.py:104-105` admits it: *"raise it when many agents share one home"* — an acknowledged knob with **no benchmark, no test, no measurement.**

**The missing acceptance test, named precisely.** No test asserts that `recall(query, k=5, agent_id=...)` returns 5 results when the agent owns <3.33% of the corpus, and none asserts recall@10 is preserved as agent count rises. `config.py:238` only checks the value is a number. A falsifiable test: build a home with K agents × M rows, overfetch a factor f, assert the dense leg's top-k contribution is non-decreasing in f.

**Effort.** 1–2 days to add the test and benchmark the curve; 3–5 days for a real pre-filter (per-`agent_id` partitioned indexes, or a filter-aware HNSW). **Timeline:** 1-month (test), 1-quarter (pre-filter).

**Fix.** Establishing the curve converts an acknowledged guess into a number. **Caveat: no measured value for how much 3.0 actually recovers — every number here is [D].**

---

### PERF-A2-06 — `recall` is a write path: a read costs 20.7× a write
**Location:** `recall.py:215-233`, `store.py:729-735`, `store.py:76-82` (`_serialized`)
**Severity: P1** — the most counter-intuitive measurement in the catalog.

**The code.** `recall.py:216-222` issues `UPDATE memories SET access_count = access_count + 1, last_accessed_ms = ? WHERE id IN (...)`, and `recall.py:229-233` commits. **The read path is a write transaction on the single writer, taken while holding the store-wide `RLock`.**

**Write amplification per read: [M] exactly 1 commit per recall.** Soak v2 measured `recall_count: 4706` **[M]** against `ops: 32942` **[M]** → **[D] 14.29% of all operations in the workload are durable writes issued by reads.** Every one forces a WAL append and an fsync-class commit under the lock every other tool call also needs.

**The inversion — the headline number.** From the same run, same backend, same lock, same 60 s:
- `remember_p50_ms: 0.254` **[M]**
- `recall_p50_ms: 5.255` **[M]**
- **[D] 5.255 / 0.254 = 20.69× — a read costs 20.7× a write.**
- At the tail: `17.625 / 0.649` → **[D] 27.16×**.

This is not explainable by the search work: the same corpus's brute scan is 0.994 ms p50 **[M]**, so the **~4.3 ms delta [D] is the write path, not the read path.**

**Why it matters.** The p95 is the gate. `recall_p95_ms: 17.625` **[M]** against `remember_p95_ms: 0.649` **[M]** — an agent in a read-heavy loop pays ~18 ms per turn at p95 for a scan that costs 1 ms, *while holding the lock the next turn's write needs.* And the p95 creep is real and measured: `p95_first_third_ms: 3.886 → p95_last_third_ms: 20.739`, **`p95_ratio: 5.337`, `validation_gate_passed: False`** **[M]** — a **failing** stability gate, against `soak.py:52`'s 4.0 limit.

**Effort.** 0.5–1 day: batch the access bump into the existing maintenance tick (`server.py:284-291` already runs a 1.0 s loop calling `store.maintenance_tick()`), or fire only when the count crosses a power-of-two boundary. **Dependencies:** the bump feeds `final_score`'s `frequency` term (`recall.py:114-121`), so deferring it changes ranking by up to one increment per `half_life_hours` — a real semantic change that should be documented. **Timeline:** 2-week tranche.

**Fix.** Removing the per-read commit should move `recall_p50` toward the measured scan cost of ~1 ms. **[D] 5.255 → ~1.3 ms is a plausible 4× and would take `p95_ratio` under the 4.0 gate — but this is derived and explicitly NOT measured.** The honest claim is directional: the 20.7× gap is the size of the prize, and closing it is the only measured change that plausibly turns the one failing stability gate green.

---

### 🔴 PERF-A2-07-R — WITHDRAWN IN PART (Wave 3): the lexical numbers are UNREPRODUCIBLE

> **The leg is real. The numbers are not.** Wave 3 established that
> `db.py:96-103` (`_MIGRATION_5`) **dropped all three FTS triggers** and moved index
> sync into the store write path (`store.py:571`, `:658`). Both lexical harnesses —
> `bench/verify_qdrant_rootcause.py` and `bench/lexical_scaling.py` — build their corpus
> by inserting into `memories` over raw SQL, so **`memory_fts` stays empty and the MATCH
> expression matches nothing.** A zero-match query returns in microseconds, so the sweep
> emits a flat, fast, healthy-looking line and an exponent fitted to it.
>
> Reproduced: `bench/lexical_scaling.py` at n=800 on current code prints
> `match set 0 rows (specific 0)`, against its own committed artifact
> `lexical-scaling-results.txt:21-22` claiming `800` and `800`.
>
> Therefore **withdrawn**: the `209.8 ms p50 / 255.6 ms p95 at 60k` figure, the
> `75% of total` share, the `53× growth from 5k`, the `8.4× over the 25 ms gate`, and
> `lexical_growth_exponent: 0.949`. None of them can be regenerated.
>
> **What replaces them** (`bench/lexical_cause.py`, `bench/lexical-cause-results.txt`,
> index correctly populated, 12 queries/size, 2 repeats, `k_fetch=50`):
>
> | variant | n=5,000 | n=25,000 | α |
> |---|---|---|---|
> | **V0** production `ORDER BY bm25(...) LIMIT 50` | **4.740 ms** | **30.135 ms** | **1.149** |
> | V1 same, `ORDER BY` removed | 0.047 ms | 0.095 ms | 0.440 |
> | V2 FTS-only `ORDER BY rank`, no join | 5.708 ms | 27.758 ms | 0.983 |
> | V5 identical to V0, match set width 1 | 0.021 ms | 0.026 ms | 0.148 |
> | V4 `COUNT(*)` over the match expression | 0.796 ms | 6.209 ms | 1.276 |
>
> - **α = 1.149, super-linear.** The `1.60 vs 0.949` contradiction is resolved: `0.949`
>   was fitted to an empty index. The two numbers were never measuring the same thing.
> - **`ORDER BY bm25` is 99.6% of the leg** (V0 − V1 = 30.040 ms of 30.135 ms).
> - **The `JOIN memories` re-check is REFUTED as the cause** (V0 − V2 = 2.378 ms, ~8%;
>   V2 is not faster). Sort spill is a mechanism of the ranking, not a separate cause.
> - **Root cause is `fusion.py:34-41`**, not the SQL: `fts_match_query` ORs up to 24
>   lowercased word tokens, an ordinary question shares most of them with most rows, so
>   **the match set is the corpus** and the leg ranks 25,000 rows to return 50.
>
> **Caveats carried forward honestly:** the 60k band was not measured (tool-shell
> failure mid-wave), so this rests on **two** corpus sizes and no fix speedup is claimed;
> and the match-width guard now in `lexical_cause.py` fires rather than publishing a
> flat line — that guard is the only reason this void was found instead of re-measured.
> **Measurement gap #22 below is partly closed** (the leg is now benched and its cause
> known); it stays open only as a *gate*, which this deliberately does not assert.

### ⚠️ PERF-A2-07 (SUPERSEDED) — 1.2–2.4× slower than Qdrant local
> **SUPERSEDED by PERF-A2-07-R above.** The `BruteForceIndex` substitution is **CONFIRMED**;
> the implication that switching to HNSW closes the gap is **REFUTED by measurement** — at 25k
> the dense leg is 5.35× faster on HNSW and end-to-end `recall()` gets **32% slower**. The real
> root cause is the **FTS5 lexical leg** (209.8 ms p50 at 60k, 75% of the total). Retained
> unedited for auditability — **do not act on the reasoning below.**
**Location:** `bench/research-v3-results.md:60-63, 110-112, 160-162, 210-212`; `bench/research_bench_v3.py:272`, `:363`; `recall.py:112-133`; `server.py:15-16`
**Severity: P1** — and this finding **contradicts my own earlier "not fixable" verdict**, on its root cause.

**The measured gap [M], 5 fresh processes per engine, p50 per process:**

| Corpus | OmniDB p50 range | Qdrant p50 range | **[D] mean ratio** |
|---|---|---|---|
| nfcorpus 3,633 | 8.100 – 9.025 | 4.598 – 6.070 | **1.65×** |
| scifact 5,183 | 19.873 – 21.551 | 8.151 – 8.733 | **2.42×** |
| scidocs 25,657 | 70.457 – 72.774 | 40.025 – 41.781 | **1.76×** |
| fiqa 57,638 | 116.977 – 123.784 | 97.132 – 390.163 **(only 3 values)** | **0.60×** (unstable) |

The defensible range is **1.2–2.4×** on the three corpora with complete 5-process data.
*(GROUND_TRUTH §5 states "1.6–2.0×" — narrower than the data supports; scifact is 2.42×.)*

**Root cause — and it is not what the prior pass said.** The prior finding attributed the gap to "Python-based two-leg fusion" and concluded parity was **not fixable under the current design.** That rests on a root cause I now believe is wrong: `DATASTORE_SPIKE.md:63-66` records, in the project's own words, that the v3 benchmark "**substituted `BruteForceIndex`** (`research_bench_v3.py:272`)." **So OmniDB's latency column is a full-corpus brute-force scan, while Qdrant is running an HNSW index.** That is not "Python fusion vs native fusion" — it is **linear scan vs ANN.**

**[D] Confirming against the arithmetic.** scidocs: `25,657 × 768 × 4 = 79.2 MB` of matrix. The measured OmniDB p50 is **71.80 ms**. A dense matmul over 79.2 MB at single-thread memory bandwidth completes in single-digit ms; the residual is FTS5 plus the Python fusion loop. For fiqa: `57,638 × 768 × 4 = 177.0 MB` at **119.55 ms**. Both consistent with a full scan. Against `bench/results.md`, the **hnswlib** path at 768-d/100k/ef128 measures **0.770 ms p50 / 1.710 ms p95** **[M]** — three orders of magnitude below the brute figure. **The shipped configuration has never been measured through `recall()` at any of these corpora.**

**Secondary contributors, honestly weighted:**
1. **Two legs, one fusion, in Python** — `recall.py:112-133` builds a result dict per row with `round(...)` on five fields. At k=10 that is ~10 dict constructions — microseconds. **Real but small; I retract any implication that it is dominant.**
2. **RRF is a per-query Python dict merge**, not a C-level sort. Same order, same retraction.
3. **anyio offload — NOT a contributor to this measurement.** `server.py:15-16` runs tool bodies via `anyio.to_thread.run_sync`. That hop is real for an MCP client, but **every latency number in this table is an in-process `recall()` call**, so the hop was never inside the measured window. Citing it as a cause of *this* gap would be wrong. *(It remains an unmeasured cost for real sessions — gap #20.)*
4. **`store.py:76-82` serializes all tool bodies on one RLock.** Qdrant local has no equivalent contention. Not measured.

**Why it matters.** If real, an agent feels 20–120 ms per retrieval instead of 8–40 ms, and at fiqa scale ~2.4 extra turns of "thinking" latency on a corpus OmniDB is otherwise better at. If an artifact, we are shipping a competitive-looking but methodologically void comparison into a roadmap.

**Effort.** 0.5 day: re-run `research_bench_v3.py` with `kind="hnsw"` and report the dense leg separately. **Dependencies:** `hnswlib` must be installed — and `ci.yml:64-70` already treats hnsw absence as a failure, so this is policy-supported. **Timeline:** immediate; it is a measurement, not a fix.

**Fix.** **I do not recommend a code change yet — I recommend re-measuring.** If the hnsw-backed number closes most of the gap, the "not fixable" verdict is retracted and the work is a documentation fix. **Caveat: this is a hypothesis derived from reading `research_bench_v3.py:272` as cited by `DATASTORE_SPIKE.md`. I did not re-run the benchmark and have not read that file myself. Verify before formally retracting the prior verdict.**

---

### PERF-A2-07-R — RESOLUTION: the lexical leg, measured
**This supersedes the hypothesis A2 originally filed under PERF-A2-07. The original section is
retained below, marked SUPERSEDED, so the reasoning chain stays auditable.**

A2's first pass claimed the 1.2–2.4× Qdrant gap was "not fixable" because OmniDB pays a
Python two-leg fusion tax, then hypothesised the real cause was that
`research_bench_v3.py:272` substituted `BruteForceIndex` so the comparison was linear-scan vs
HNSW. **Both halves have now been tested. One confirmed, one refuted.**

#### ✅ CONFIRMED — the harness really did substitute BruteForceIndex
Verified by direct source inspection, independent of `DATASTORE_SPIKE.md`:
- `research_bench_v3.py:272` — `store.index = BruteForceIndex(home / "vectors.npz", shared.dim)`
- `:301` — the same substitution in `retrieve_omnidb_dense_only`
- `:359-365` — `retrieve_qdrant` uses `QdrantClient` + `Distance.COSINE` (HNSW by default)
- `:410-415` — `between_process_latency` drives exactly those two functions

#### ❌ REFUTED — switching to HNSW does not close the gap
`bench/verify_qdrant_rootcause.py` (new, reproducible; run with `QCR_WORKDIR` set to a volume
with free space — C: hit 0 bytes during the first attempt):

| Corpus | dense brute **[M]** | dense hnsw **[M]** | speedup | **full recall() brute → hnsw** |
|---|---|---|---|---|
| 5,000 | 0.327 ms | 0.230 ms | 1.42× | 6.201 → 5.692 ms (**−8.2%**) |
| 25,000 | 1.649 ms | 0.308 ms | **5.35×** | 32.025 → 42.386 ms (**+32.4%**) |
| 60,000 | 6.574 ms | 1.681 ms | **3.91×** | 280.164 → 291.642 ms (**+4.1%**) |

**At 25k the dense leg is 5.35× faster on HNSW and end-to-end `recall()` gets 32% SLOWER.**
The linear scan was never the binding constraint.

#### ✅ The actual root cause — the FTS5 lexical leg scales worse than the dense leg
| Corpus | dense hnsw **[M]** | **lexical FTS5 [M]** | lexical share | residual **[M]** |
|---|---|---|---|---|
| 5,000 | 0.230 ms | **3.973 ms** | 70% | 1.489 ms (26.2%) |
| 25,000 | 0.308 ms | **25.226 ms** | 60% | 16.852 ms (39.8%) |
| 60,000 | 1.681 ms | **209.762 ms** | **75%** | 80.199 ms (27.5%) |

**The lexical leg grows 53× from 5k to 60k while the hnsw dense leg grows 7.3×.** It is the
only leg that scales badly, and Qdrant local — being **vector-only** — never pays it.

**So the "not fixable" verdict was RIGHT, for a different reason than stated.** The gap is not
a Python-fusion tax and not a scan artifact. It is that **a hybrid engine must run FTS5, and
the repo has never measured FTS5's own scaling curve.** `bench/results.md` sweeps HNSW across
dim × n × ef and never touches the lexical leg.

**The residual 26–40%** (fuse + hydrate + the access-bump write of PERF-A2-06) is the second
term, already documented.

#### Why this outranks the Qdrant framing
The actionable finding is not "we're slower than Qdrant" — it is that **the product's dominant
retrieval cost is a component with no benchmark, no gate, and super-linear scaling.** At 60k
the lexical leg is **209.8 ms p50 / 255.6 ms p95 against a 25 ms p50 gate and a 5 s ADR-7
budget** — 8.4× over the p50 gate. The v3 artifacts never exposed this because the reported
latency column was averaged against a vector-only competitor that has no such leg.

**Effort.** 0.5 day to add a lexical scaling benchmark (the harness exists; it has never been
pointed at the FTS leg). Then determine whether the growth is in FTS5's `bm25()` ranking, the
`JOIN memories` re-check, or the `ORDER BY`. **Timeline:** 1-month tranche.

**Fix direction (not yet validated).** In order of expected yield:
1. **`ORDER BY bm25(memory_fts) LIMIT 50` forces a full match-set sort** — a `rank`-bounded
   subquery or an FTS5 `rank` column indexed alongside may let SQLite stop early. This is a
   query-shape change, not a rewrite. Test first.
2. Prefilter the join by narrowing `memories` to the requested `kind`/`agent_id` before FTS.
3. Only if 1 and 2 fail: a bounded lexical structure (an n-gram postings table is the cheap
   version and reuses the ENH-05 skill-index design).

**Caveat, stated plainly:** these are **synthetic clustered vectors and templated query text**,
not the BEIR corpora. The *relative* decomposition (lexical dominates; hnsw does not help
end-to-end) is robust across three magnitudes; the *absolute* milliseconds are **not**
comparable to `research-v3-results.md`. One `research_bench_v3.py` run with `kind="hnsw"`
plus a lexical-scaling sweep would close both remaining gaps.

---

### PERF-A2-08 — Cross-encoder inference: 274 s → 10,995 s per cell
**Location:** `bench/research-v3-results.json` (cited); `bench/rerank-window-results.md:169-176`
**Severity: P2** — a *decision* finding. Its value is preventing a 3-hour mistake.

**The spread.** **274 s (nfcorpus) → 10,995 s (fiqa, e5-small)** per cell, **[M per GROUND_TRUTH §5, sourced to `research-v3-results.json`]**. **[D] 10,995 / 274 = 40.1× across a 15.8× size difference (3,633 → 57,638 docs).** Cost scales **super-linearly**, not linearly. **[D] Scaled across the full v3 grid (4 corpora × 4 embedders = 16 cells), the worst-case rate implies `4 × 10,995 = 43,980 s = 12.2 hours`** of pure inference to evaluate one configuration across four corpora.

**What it implies for anyone adding a reranker.** Three compounding costs:
1. **Evaluation cost:** 12.2+ hours per configuration, on one machine. Nobody iterates that often, so the reranker space gets explored once and poorly.
2. **The quality payoff is zero-to-negative.** `rerank-window-results.md:169-176`'s own rubric yields best cell **+0.0067 (n.s.)**, worst **−0.0541 (p<0.0001)**. `research-v3-results.md:24,35,46,57` confirms per-embedder CE deltas of −0.0030, −0.0127, +0.0025, −0.0187 — **not one of the four is significant.** A 12-hour cost buying a statistically invisible change is not a close call.
3. **Interactive latency would be fatal.** **[D]** If the fiqa cell's 10,995 s covered ~1,000 queries, that is ~11 s per query — against a tens-of-milliseconds budget. *I could not verify the query count; treat the 11 s as order-of-magnitude only.*

**Honesty note on the source.** I could **not** find 274 or 10,995 in `research-v3-results.md` — a direct fetch confirms both are **absent from it**. They are in the `.json`, per GROUND_TRUTH. **Second-hand; not independently verified.**

**Effort.** Documentation only: 1 hour. **Fix.** Publish the 40× spread and the negative nDCG result **together**, in the same paragraph, in any document mentioning reranking — so the number cannot be quoted without the verdict. Expected improvement: zero performance, non-zero decision quality.

---

### PERF-A2-09 — 32 MiB result cap + 5M VM steps: the binding bound is 5 s of a single-writer lock
**Location:** `store.py:29-33`, `:167-169`, `:737` (`@_serialized`), `:764-783`, `config.py:89`
**Severity: P2** — reduced by the trust model.

**The budgets [M], `store.py:32-33, 169`:** `_QUERY_WALL_CLOCK_LIMIT_S = 5.0`; `_QUERY_RESULT_BYTES_LIMIT = 32 MiB`; `self.max_query_steps = 5_000_000`, enforced as `max_callbacks = 5,000` at a 1,000-instruction progress-handler interval (`:771-772`); plus `max_search_rows = 500` (`config.py:89`) checked at `store.py:801`.

**The budgets are not equally binding, and the design picks the loosest as the effective limit. [D]** SQLite's VDBE executes on the order of `10^8` simple instructions/second, so 5,000,000 instructions is worth roughly **50 ms** (conservative) to **5 ms** (optimistic). Against a 5.0 s wall clock, the VM budget is **two to three orders of magnitude looser.** The wall clock is therefore the effective bound, and the store's own comment at `:768-770` concedes why: *"A VM-callback deadline cannot interrupt a long-running scalar function mid-call; hard wall-clock guarantees would need process isolation, which this local channel does not do."* **Caveat: SQLite instructions/second on this hardware is unmeasured (gap #14). The direction is unambiguous; the magnitude is not measured.**

**Worst-case cost on the single writer.** `search` is `@_serialized` (`store.py:737`), holding `self._op_lock` (`store.py:158`) for the entire execution including all 5 s of the deadline. So a hostile or buggy query:
- **holds the store-wide RLock for up to 5.0 s**, during which every other tool call blocks;
- **accumulates up to 32 MiB of materialized rows** (`:798`) plus Python overhead for 500 row objects (`:811`);
- **is not guaranteed to be interruptible at 5.0 s** if a scalar function is mid-call, per the code's own comment.

**And the byte cap overshoots by design.** `store.py:790-800` does `fetchmany(256)` → append rows → *then* check `bytes_seen > _QUERY_RESULT_BYTES_LIMIT`. **The check happens after up to 256 rows are already materialised**, so 32 MiB is a threshold, not a bound.

**Why P2 and not P0.** Per cross-verification: trigger patterns are supplied by the writer of the memory, the default deployment is single-user local stdio, and the same agent owns the home — **self-inflicted, single trust domain.** A cross-tenant DoS exists only under the hosted target, which has zero measurements in this repo. The scenario is concrete but narrow: an agent emits one pathological `search` and its own tool loop stalls for 5 s.

**Effort.** 2–4 hours: lower `_QUERY_WALL_CLOCK_LIMIT_S` to a value justified by measurement (none exists), and check `bytes_seen` before appending. **Timeline:** 1-quarter, gated on a hostile-query benchmark (gap #11). **Fix.** **Improvement is unmeasurable today** — the honest recommendation is to measure first.

---

### PERF-A2-10 — `bench/baseline.py` gates on `max(...)`: the only CI-gated benchmark cannot fail
**Location:** `bench/baseline.py:133-140`
**Severity: P0** — a measurement-integrity defect. The one benchmark wired into CI is structurally incapable of detecting a regression.

**Verified in source:**
```python
hnsw_scores = [r[f"ef{ef}"]["recall@10"] for r in rows for ef in (96, 128) if f"ef{ef}" in r]
sq8_scores  = [r["sq8_rerank"]["recall@10"] for r in rows]
best = max(hnsw_scores + [min(sq8_scores)]) if hnsw_scores else min(sq8_scores)
gate = best >= 0.95 if hnsw_scores else min(sq8_scores) >= 0.90
return 0 if gate else 1
```

**Quantifying that it cannot fail.** `hnsw_scores` draws **only** `ef ∈ (96, 128)`. Populated from the measured `results.md` sweep **[M]**:

| Cell | in `hnsw_scores`? | recall@10 |
|---|---|---|
| 384-d/1k ef96 · ef128 | yes | 0.997 · **1.000** |
| 384-d/10k ef128 | yes | 0.873 |
| 384-d/100k ef128 | yes | 0.383 |
| 768-d/1k ef96 · ef128 | yes | 0.993 · 0.997 |
| 768-d/10k ef128 | yes | 0.823 |
| 768-d/100k ef128 | yes | **0.359** ← worst included |
| all ef32 and ef64 cells | **no — filtered out** | 0.170 … 0.990 |
| `min(sq8_scores)` = 0.881 | appended | 0.881 |

- `best = max(1.000, …) = **1.000**` → `1.000 >= 0.95` → **gate passes.**
- The SQ8 branch is **inert**: it requires `min(sq8_scores) ≥ 0.95`, and `min = 0.881`.
- **[D] Overstatement: `1.000 / 0.359 = 2.79×`.** The gate reports perfect retrieval while the 768-d/100k cell it silently admits is at **0.359** — a 64% miss rate on the largest, most realistic configuration.
- **[D] Overall minimum across all 24 measured cells is 0.170** — **5.88× below the reported gate value.**

**The failure condition is vacuous.** For the gate to fail, **every one** of the 8 HNSW cells at ef≥96 must fall below 0.95 — including the 1k cells at 0.997–1.000, which would have to lose 5% of their accuracy. **The gate detects total destruction of the index, not regression of it.** A 2.79× quality collapse, an inverted distance metric, a broken `ef_search` floor, or a 768-d regression would all pass green.

**Why it matters.** Every other number in §0 traces back to this file. A gate that cannot fail means the repository's *only* automated performance signal is decorative — the root cause of the "quality bar nobody meets" situation going unnoticed.

**Effort.** 0.5 day, no dependencies. **Fix.** Replace `max(...)` with a per-cell table (`min` per corpus size, or an explicit `{(dim, n): floor}` map). Expected: the gate becomes **red on the current tree** — the 768-d/100k cell at 0.359 is below any defensible floor. **That is the point: a gate that has never been red has never been a gate.** **Caveat: the fix makes CI fail immediately, so it must land together with `ef_search = 128` (+0.117 recall@10 at 100k for +0.162 ms) or with floors set at measured values, or the project is left with a permanently red pipeline and no improvement.**

---

### PERF-A2-11 — `rebuild_index` materialises the whole corpus on a startup path
**Location:** `store.py:928-959` (`rebuild_index`), `store.py:237-276` (`_reconcile_index`)
**Severity: P1** — this is the crash-reconstruction path, the third ADR-7 axis, and the one with the least measurement.

**`rebuild_index` is the unbounded one.** `store.py:932-934`:
```python
rows = self.conn.execute(f"SELECT id, content FROM memories WHERE {_SELECTIVE} ORDER BY id").fetchall()
```
**`.fetchall()` — the entire corpus, id and full content, in one Python list.** The `chunk = 512` loop at `:949-953` chunks only the *encode*; the fetch that precedes it has no bound. This is the same defect class as PERF-A2-04 in a sibling file, and the codebase clearly knows the correct pattern (`_RECONCILE_BATCH_CAP = 500` at `store.py:37` exists precisely for this) — it just is not applied here.

**[D] Peak RAM for `rebuild_index`:** the only content-size measurement is `content_bytes_mean: 88.67` **[M]**, but that is the *synthetic* hashing workload. At 88.67 B: 100k → **8.9 MB**, 1M → **88.7 MB** (tolerable). At a realistic 1 KB agent memory: 100k → **100 MB**, 1M → **1.0 GB** (not tolerable). Plus `fresh.add()` in 512-row chunks, whose cap-doubling reallocates the whole matrix at each doubling: **[D] ~1.5× the final index size transiently.** Then `_persist_index()` (`:955`) writes the full snapshot: at 100k/768-d, **308 MB**.

**The bounded one, for contrast — `_reconcile_index` is correct.** `store.py:256-258` materialises the *full id set* into a Python set, which is **[D] ~4 MB at 100k / ~40 MB at 1M** (≈32 B/entry) — proportional to N but ~75× smaller than the matrix, and it holds ids, not content. It then fetches content in `_RECONCILE_BATCH_CAP = 500` batches. **AUDIT-I13 did its job here. The finding is that the same discipline was never applied to `rebuild_index`.**

**The gate on this path.** `store.py:237-240` is the fast path — if not dirty and no pending reembed and `active_count() == _live_count()`, return. So a clean home pays only a `COUNT(*)`. **Any crash mid-write leaves `index_dirty=1`**, and the next open takes the full O(N) path.

**What is measured about startup — and what is not.** **[M]** `perf-report.md:19` gives **26 ms cold start on a fresh, empty home**; `comparative-results.md:10,16` gives `cold_start_s 0.0121` / `reopen_s 0.0067` at 2,000 documents. **Both are empty or trivial homes.** `comparative-results.md:10` in fact shows OmniDB *losing* cold start to sqlite-vec (0.0121 vs 0.0022) while winning everything else. **There is no measurement of a cold start against a populated home, and none of a crash-recovery reconcile at any size (gap #4).** Combined with PERF-A2-01, the honest worst case for a 100k/768-d home that crashes dirty is **[D] reconcile fetch + a 308 MB rewrite + a 12.01 s hnsw rebuild — entirely derived, and precisely the third ADR-7 gate with zero evidence behind it.**

**Effort.** 0.5–1 day: `.fetchall()` → a cursor with a batched fetch matching `_RECONCILE_BATCH_CAP`. **Dependencies:** `rebuild_index` is called from the compaction path and from `portability.py`, so batching must preserve the final `self.index = fresh` swap and `_persist_index()`. **Timeline:** 2-week tranche. **A populated-home cold-start benchmark is the higher-value half and is a gap, not a fix.**

**Fix.** Batching caps the `rows` list at 500 rows instead of N. **Caveat: no latency or RAM improvement can be claimed, because no populated-home startup has ever been timed.** The change is justified by symmetry with `_reconcile_index`, not by a measurement.

---

### PERF-A2-12 — `source_identity_sha256`: cost is negligible; the "slow harness" hypothesis is REFUTED
**Location:** `bench/soak.py:66-73`, called once at `:140`
**Severity: P3** — included because it was on the assignment list and the answer is negative.

**What it does.** `soak.py:66-73` rglobs `omnidb_server/**/*.py`, `read_bytes()` each, folds name+content into one SHA-256. Called **once**, at `:140`. Also `hashlib.sha256(Path(__file__).read_bytes())` at `:229` and `:301` re-reads `soak.py` (16.63 KB) once each.

**[D] Cost:** 22 modules at 4,195 LOC ≈ **~160 KB**, plus `soak.py` ≈ **177 KB total**. SHA-256 on a modern x86 core runs 0.5–2 GB/s, so the hashing is **<1 ms**; the `rglob` is a few ms. **Total under ~5 ms, once, against a 60-second soak — under 0.01% of the run.**

**Verdict: REFUTED. This is not why the harness is slow.** The hypothesis fails on three independent grounds: (a) it runs once, not per iteration; (b) the byte volume is ~177 KB; (c) the soak's duration is a **deliberate fixed cap** — `--minutes` defaults to 2 and the growth workload stops at `growth_cap`. The harness is slow because it is *specified* to be slow, which is correct for a soak.

**But the same inspection surfaced a real cost in a sibling file.** `bench/baseline.py` recomputes the brute-force ground truth **inside the `ef` loop and again inside the SQ8 loop** — **5 identical full-corpus brute-force passes per cell when one suffices**, and the truth set is by construction independent of `ef`. **[D] 4 of 5 are redundant; at 100k×768 that is 4 × 100 × (768 × 100,000) = 30.7 GFLOP of wasted matmul per cell.** Caching `truth` per `(dim, n)` is a 3-line change. **I am not estimating the wall-clock saving** — the flops-to-seconds conversion depends on BLAS threading and I have not measured it.

**Effort.** Baseline truth-caching: 30 min. `_source_identity`: 0 — leave it; it is correct, cheap, and it is what binds the measurement to the runtime. **Fix.** **No change recommended to `_source_identity`.**

---

## §2 · Measurement gaps

### 2.1 Confirmed from `DATASTORE_SPIKE.md:63-72` (12 gaps, re-verified)

| # | Gap | Verification |
|---|---|---|
| 1 | Concurrency ceiling | **CONFIRMED** — only a two-process lock test; no 1/8/64-client measurement |
| 2 | Backup/restore timing | **CONFIRMED** — nothing in `bench/` times an export/import cycle |
| 3 | **HNSW backend soak** | **CONFIRMED and worse than stated** — `soak-results-v2.txt:8` and `growth-results.txt:7` both read **`backend: brute`**. `soak.py:120` defaults to `--backend brute`. **The shipped hnsw backend has never been soaked or grown.** |
| 4 | Cold start on a populated home | **CONFIRMED** — the 26 ms figure is an empty home |
| 5 | Recall quality through `recall()` | **CONFIRMED** — v3 substituted `BruteForceIndex` |
| 6 | Graph latency / cost | **CONFIRMED** |
| 7 | Per-tenant disk consumption | **CONFIRMED** |
| 8 | Datasets above 57,638 docs | **CONFIRMED** |
| 9 | Stores above 4,000 live memories | **CONFIRMED** — `growth-results.txt:10` `growth_cap: 4000`; the 4,000→8,000 band has **2 samples** |
| 10 | Restore peak RAM | **CONFIRMED** |
| 11 | Worst-case hostile queries | **CONFIRMED** — the 5 s / 5M-step / 32 MiB budgets are unexercised |
| 12 | Multilingual retrieval quality | **CONFIRMED** |

### 2.2 Extensions — gaps `DATASTORE_SPIKE.md` does not list (9 more)

| # | Gap | Why it matters here |
|---|---|---|
| 13 | **Flush cost as a function of corpus size** | The single largest hole. Every write-amplification figure in PERF-A2-01 is arithmetic; **not one flush has been timed above 4,000 rows.** The entire P0 rests on a format argument, not a timing. |
| 14 | **SQLite VM instructions/second on this hardware** | Determines whether the 5M-step budget or the 5 s wall clock binds (PERF-A2-09). Order-of-magnitude reasoning says the wall clock by 2–3 orders; unmeasured. |
| 15 | **`import_jsonl` / `rebuild_index` peak RSS** | Both are unbounded `.fetchall()`/single-encode paths. Gap #10 says "restore peak RAM"; this is the *code* side of the same hole. |
| 16 | **Multi-agent recall quality (`agent_overfetch`)** | A knob whose effect has never been measured (PERF-A2-05). No test, no benchmark. |
| 17 | **hnsw vs brute latency *through `recall()`*** | **PARTIALLY CLOSED.** `bench/verify_qdrant_rootcause.py` now measures the decomposition on synthetic data (5k/25k/60k, 768-d). **Still open on the real BEIR corpora** — and the synthetic run says the answer is "it barely matters," so the gap matters less than believed while the *lexical* leg matters much more. |
| 18 | **Recall p95 beyond 8,000 live rows** | `growth-results.txt:38` has **2 samples** in the top band. `recall_p95_ms` — the ADR-7 gate metric — is unknown above 4,000. |
| 19 | **SQ8 p95, build time, and rerank fan-out** | The only path to ≥0.95 recall@10 at 100k is measured at p50 only, with no tail. |
| 20 | **MCP/anyio transport overhead** | Every latency number in this repo is an in-process `recall()`. `server.py:15-16`'s `to_thread.run_sync` hop — the real client path — is in **no** measurement. |
| 21 | **`_MAX_RECALL_K` / `max_search_rows` interaction under k > 50** | `recall.py:58-66` clamps k to 1,000 and over-fetches to `_MAX_RECALL_K`, while `config.py:89` caps search rows at 500. The crossover behaviour is untested and undocumented. |
| **22** | **⭐ FTS5 lexical-leg scaling — now the #1 retrieval cost, and completely unbenchmarked** | Measured at **209.8 ms p50 / 255.6 ms p95 at 60k** (PERF-A2-07-R), growing **53×** from 5k while the hnsw dense leg grows 7.3×. **8.4× over the 25 ms p50 gate.** `bench/results.md` sweeps dim × n × ef for HNSW and **never once measures the lexical leg** — the product's dominant cost has no benchmark, no gate, and no baseline. |

**Total: 21 gaps.** Gaps 13, 17 and 20 block other findings from being resolved with evidence rather than argument.

---

## §3 · What I did not find, and the corrections this pass makes

1. **SUPERSEDED → RESOLVED. See PERF-A2-07-R.** The BruteForceIndex substitution is
   **CONFIRMED** by source inspection; the "HNSW would close the gap" implication is
   **REFUTED by measurement** (`bench/verify_qdrant_rootcause.py`: at 25k the dense leg is
   5.35× faster on HNSW and end-to-end `recall()` is **32% slower**). **The real root cause is
   the FTS5 lexical leg — 209.8 ms p50 at 60k, 75% of the total, growing 53× from 5k to 60k.**
   The "not fixable" verdict survives, but for an entirely different reason than first filed.
2. **Retracted: the `.npz` load makes a *third* copy.** `np.ascontiguousarray` on an already-contiguous f32 array returns the same object. **It is two copies** (PERF-A2-03).
3. **Refined: the adaptive persist control is not merely "overridden," it is *dead code* at every measured corpus** — it cannot engage unless a flush exceeds 62.5 ms, and measured flushes are 4.8–9.8 ms (PERF-A2-01).
4. **Refined: the `forget_many`/`purge` item is one limit, not two**, and it is a *correctness* finding (the loss is the original), not a performance one. **Not re-asserted here.**
5. **Refuted: `source_identity_sha256` is why the harness is slow** (PERF-A2-12).
6. **New: SQ8 is strictly dominated in 4 of 6 cells** — any adoption must be corpus-size-gated, not a global flag (PERF-A2-02).

---

## §4 · Contradictions and unverifiable citations

| # | Contradiction | Detail |
|---|---|---|
| 1 | **`DATASTORE_SPIKE.md:28` does not describe `soak-results-v2.txt`** | Spike says *"788 flushes = 5.84 s of 60 s · recall p50 6.70 / p95 12.99 ms · RSS 39.6→69.4 MB · p95 creep ratio 4.39"*. The file it cites says **786 flushes · 3,761.719 ms · p50 5.255 / p95 17.625 · RSS 38.777→68.363 · p95_ratio 5.337**. **Six of six numbers differ.** GROUND_TRUTH §5 agrees with the file. Either the spike cites a different run or it is fabricated. |
| 2 | **CE inference numbers not in the cited `.md`** | 274 s / 10,995 s are attributed by GROUND_TRUTH §5 to `research-v3-results.json`; a direct fetch of `research-v3-results.md` confirms **both are absent from it**. Second-hand; not independently verified. |
| 3 | **GROUND_TRUTH's "1.6–2.0×" understates the measured range** | The 5-process lists give **2.42× at scifact**. The defensible range is **1.2–2.4×**. |
| 4 | **fiqa qdrant has 3 values under a "5 processes" heading** | `research-v3-results.md:212-214` lists `97.1318, 110.0549, 390.1627` — three, not five, spanning 4×. `DATASTORE_SPIKE.md:66-69` independently notes qdrant local "swung 97→390 ms across processes." **The fiqa row is not a 5-process median and should not be quoted as one.** |
| 5 | **`perf-report.md` headline numbers are pre-remediation and partly unreproducible** | The file self-labels (`:3-9`) as "historical pre-remediation" and says v1 is "not comparable to v2," yet 4 of its 6 headline numbers (26 ms cold start, 3.8 k/s, 29 k/s, 0.37 ms p50) appear in **no** current artifact. **§0.1 and §0.2 label them accordingly.** |
| 6 | **`results.md` (measured) vs `perf-report.md` (asserted) disagree on the same sweep** | 384-d/100k/ef128: `results.md` **0.519 / 0.536 ms**, `perf-report.md:34` **0.383 / 0.632 ms**. Same benchmark, different numbers. GROUND_TRUTH §5 sides with `results.md`. **A second instance of the pattern `7_CROSS_VERIFICATION.md` describes.** |
| 7 | **`.npz` "614 MB peak" reconciles to 616.0 MB** | `2 × 307.2 + 2 × 0.8 = 616.0`. The 614 figure is the same arithmetic rounded. **Not a contradiction** — recorded so the lead does not treat them as two claims. |
| 8 | **`DATASTORE_SPIKE.md:41` locates `_RECONCILE_BATCH_CAP` nowhere** | Spike cites `store.py:32-33,822`; the constant is at `store.py:37`, and `:822` is inside the search result loop. Minor, but the citation does not resolve. |