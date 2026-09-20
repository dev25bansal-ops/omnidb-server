# OmniDB Performance Report

> **Status (2026-09-19):** these are historical pre-remediation measurements.
> Post-remediation evidence lives in `bench/soak-results-v2.txt` (corrected
> bounded-live stability run, gate passing on the final runtime) and
> `bench/growth-results.txt` (monotonic scaling: recall p95 ~1.1 ms at 110→500
> live rows rising to ~4.7 ms at 2000→4000, brute backend, 512-d hashing).
> The v1 soak workload was not steady-state (AUDIT-I12) and its numbers are
> not comparable to v2.

Measured locally on Python 3.14 / Windows (numpy 2.3.5, hnswlib M=16 efC=200).
Gates are interactive latency, cold start, and RAM — not QPS — so these numbers
target a single agent session, not a server fleet.

## Headline numbers

| Metric | Value | Notes |
|---|---|---|
| Cold start, fresh home | **26 ms** | `MemoryStore` + BruteForceIndex build to first `remember` |
| Single `remember` throughput | **3.8 k writes/s** | hashing embedder, 500 sequential inserts |
| `remember_many(1000)` throughput | **~29 k writes/s** | single transactional batch, ~34 ms total |
| Recall p50, small scale | **0.37 ms** | k=10, hashing embedder, sub-1k live rows |
| Recall p95, soak 60 s | **~12 ms** | 20 k ops, mixed remember/recall/forget |

## Vector baseline sweep (recall@10)

Full table at [`bench/results.md`](../bench/results.md). Headline rows:

| dim | n | ef | recall@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 384 | 1 000 | 96 | 0.997 | 0.047 | 0.054 |
| 384 | 1 000 | 128 | 1.000 | 0.056 | 0.063 |
| 384 | 10 000 | 128 | 0.873 | 0.147 | 0.258 |
| 384 | 100 000 | 128 | 0.383 | 0.632 | 0.911 |
| 768 | 1 000 | 96 | 0.993 | 0.093 | 0.105 |
| 768 | 1 000 | 128 | 0.997 | 0.108 | 0.129 |
| 768 | 10 000 | 128 | 0.823 | 0.243 | 0.438 |
| 768 | 100 000 | 128 | 0.359 | 0.920 | 1.191 |

Build cost: 384-d stays under 0.12 s through 10 k rows and ~5.5 s at 100 k.
768-d costs ~12–14 s at 100 k. Above ~10 k rows the brute-force path needs the
hnswlib index or the SQ8 shortlist+f32 rerank path to keep recall@10 >= 0.95.

## Stability soak (60 s)

Full output at [`bench/soak-results.txt`](../bench/soak-results.txt).

- duration 60 s, ops 21 280, cycles 2 660
- recall mean 5.50 ms, p95 11.84 ms
- p95 first third 3.40 ms, last third 12.95 ms
- RSS 37.1 -> 68.8 MB (well under 300 MB gate)
- WAL peak 1.84 MB (well under 80 MB gate)

The soak passes all three stability gates (p95 ratio < 4x, WAL < 80 MB,
RSS growth < 300 MB). Last-third p95 creep is from the BruteForceIndex's
growing candidate set; under hnswlib it stays flat.

## Reproduction

```sh
python bench/baseline.py     # writes bench/results.md, exits 0 if recall@10 gate holds
python bench/soak.py --minutes 1   # writes bench/soak-results.txt
```

Source: [`bench/baseline.py`](../bench/baseline.py), [`bench/soak.py`](../bench/soak.py).