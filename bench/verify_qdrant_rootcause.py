"""Isolate where the v3 'OmniDB is 1.2-2.4x slower than Qdrant' gap actually lives.

A2 (docs/audit/A2_PERFORMANCE.md, PERF-A2-07) hypothesised the gap is an artifact:
research_bench_v3.py:272 hardcodes BruteForceIndex, so the v3 latency column is a
full linear scan while Qdrant local runs HNSW. That hypothesis is CONFIRMED as
fact by source inspection (see A2_PERFORMANCE_QR_DRIFT.md).

This script measures the decomposition. It does NOT rebuild the BEIR corpora; it
uses synthetic clustered vectors at the same dimensionality and scales, because
the question is which term dominates, not what the absolute milliseconds are.

Run:  python bench/verify_qdrant_rootcause.py
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server.config import Config
from omnidb_server.server import build_stack
from omnidb_server.vector_index import BruteForceIndex, HnswlibIndex

DIM = 768
FETCH = 50          # recall() default k_fetch
K = 10
QUERIES = 200
WARMUP = 30
REPEATS = 5


def _pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def _stats(name, samples):
    p50 = statistics.median(samples)
    p95 = _pct(samples, 0.95)
    print(f"  {name:<34} p50 {p50:7.3f} ms   p95 {p95:7.3f} ms   n={len(samples)}")
    return {"p50": p50, "p95": p95}


def _make_vectors(n, dim=DIM, seed=17):
    """Clustered synthetic corpus, closer to real embeddings than uniform noise."""
    rng = np.random.default_rng(seed)
    n_clusters = max(8, n // 800)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, size=n)
    vecs = centers[assign] + rng.normal(scale=0.35, size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32), centers.astype(np.float32)


def _build_store(n_docs, home: Path, hnsw: bool):
    cfg = Config.from_env(home)
    cfg.corpus_mode = True
    store = build_stack(cfg)
    vecs, centers = _make_vectors(n_docs)
    store.remember_many([{"content": f"benchmark document number {i} "
                          f"covering topic group {i % 97} with reference "
                          f"material and payload {i}."} for i in range(n_docs)])
    row_ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()

    # Hand-rolled embedder: reuses precomputed rows, so no encode cost per query.
    class _Fixed:
        dim = DIM
        def __init__(self, _v, _c):
            self._v, self._c, self._i = _v, _c, 0
        def encode(self, texts, _batch=None):
            out = []
            for _ in texts:
                out.append(self._c[self._i % len(self._c)])
                self._i += 1
            return np.stack(out).astype(np.float32)
        def identity(self):
            return f"fixed-{DIM}d-synthetic"

    if hnsw:
        store.index = HnswlibIndex(home / "vectors.hnsw", dim=DIM, count_hint=n_docs)
    else:
        store.index = BruteForceIndex(home / "vectors.npz", DIM)
    store.index.add(row_ids, vecs)
    store.embedder = _Fixed(vecs, centers)
    return store, vecs, centers


def main() -> int:
    print("=" * 72)
    print("Qdrant gap root-cause isolation")
    print("  Question: is the 1.2-2.4x gap a linear-scan artifact, fusion cost,")
    print("            or the lexical leg?")
    print("=" * 72)

    # C: drive was at 0 bytes free during the first attempt; WinError 112.
    # Keep every artifact off the system temp volume.
    tmp = Path(os.environ.get("QCR_WORKDIR") or tempfile.mkdtemp(prefix="qcr-"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  workdir: {tmp}")
    findings = {}

    sizes = [int(x) for x in os.environ.get("QCR_SIZES", "5000,25000,60000").split(",")]
    for n_docs in sizes:
        print(f"\n--- corpus {n_docs:,} docs, dim {DIM}, fetch {FETCH}, k {K} ---")

        # (1) Dense leg alone, brute force. This IS what the v3 benchmark ran.
        t0 = time.perf_counter()
        store, vecs, centers = _build_store(n_docs, tmp / f"brute{n_docs}", hnsw=False)
        build_s = time.perf_counter() - t0

        qvecs = np.repeat(centers, max(1, QUERIES // len(centers)), axis=0)[:QUERIES]

        t0 = time.perf_counter()
        for i in range(WARMUP):
            store.index.query(qvecs[i % len(qvecs)], FETCH)
        _ = time.perf_counter() - t0
        brute_dense = []
        for rep in range(REPEATS):
            for i in range(QUERIES):
                t = time.perf_counter()
                store.index.query(qvecs[i % len(qvecs)], FETCH)
                brute_dense.append((time.perf_counter() - t) * 1000)

        # (2) Same dense leg, HNSW. The configuration the product actually ships.
        t0 = time.perf_counter()
        hstore, _hv, _hc = _build_store(n_docs, tmp / f"hnsw{n_docs}", hnsw=True)
        hbuild_s = time.perf_counter() - t0
        for i in range(WARMUP):
            hstore.index.query(qvecs[i % len(qvecs)], FETCH)
        hnsw_dense = []
        for rep in range(REPEATS):
            for i in range(QUERIES):
                t = time.perf_counter()
                hstore.index.query(qvecs[i % len(qvecs)], FETCH)
                hnsw_dense.append((time.perf_counter() - t) * 1000)

        # (3) Full recall() through the real stack, both backends.
        # The query text is chosen so the FTS leg returns few rows: a document
        # containing a unique number matches almost nothing, so hydrate stays
        # small and the decomposition approximates the three legs' sum.
        QUERY_TMPL = "zzq{i} payload"
        brute_full, hnsw_full = [], []
        for i in range(QUERIES):
            t = time.perf_counter()
            store.recall(QUERY_TMPL.format(i=i), k=K, kind=None, k_fetch=FETCH,
                         agent_id=None, reranker=None)
            brute_full.append((time.perf_counter() - t) * 1000)
        for i in range(QUERIES):
            t = time.perf_counter()
            hstore.recall(QUERY_TMPL.format(i=i), k=K, kind=None, k_fetch=FETCH,
                          agent_id=None, reranker=None)
            hnsw_full.append((time.perf_counter() - t) * 1000)

        # (4) Lexical leg alone, with the SAME query text as the full call.
        lex = []
        for _rep in range(REPEATS):
            for i in range(QUERIES):
                t = time.perf_counter()
                rows = hstore.conn.execute(
                    "SELECT f.rowid FROM memory_fts f "
                    "JOIN memories m ON m.id = f.rowid "
                    "WHERE memory_fts MATCH ? "
                    "AND m.deleted_ms IS NULL AND m.superseded_by IS NULL "
                    "ORDER BY bm25(memory_fts) LIMIT ?",
                    (f'"zzq{i}" OR "payload"', FETCH)).fetchall()
                lex.append((time.perf_counter() - t) * 1000)
                _ = rows

        store.close()
        hstore.close()

        print(f"  [build] brute {build_s:.2f}s   hnsw {hbuild_s:.2f}s")
        b = _stats("dense leg / brute (v3 ran this)", brute_dense)
        h = _stats("dense leg / hnsw (shipped)", hnsw_dense)
        x = _stats("lexical leg alone (FTS5)", lex)
        bf = _stats("full recall() / brute", brute_full)
        hf = _stats("full recall() / hnsw", hnsw_full)

        findings[n_docs] = {
            "build_brute_s": build_s, "build_hnsw_s": hbuild_s,
            "dense_brute": b, "dense_hnsw": h, "lexical": x,
            "full_brute": bf, "full_hnsw": hf,
            "hnsw_speedup_dense": b["p50"] / h["p50"] if h["p50"] else None,
            "sum_of_legs": h["p50"] + x["p50"],
            "measured_residual": hf["p50"] - (h["p50"] + x["p50"]),
        }
        f = findings[n_docs]
        print(f"  --> hnsw speeds the dense leg {f['hnsw_speedup_dense']:.2f}x")
        print(f"  --> switching brute->hnsw changes full recall() by "
              f"{(hf['p50'] - bf['p50']):+.2f} ms "
              f"({(hf['p50'] / bf['p50'] - 1) * 100:+.1f}%)")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for n, f in findings.items():
        print(f"\n{n:,} docs:")
        print(f"  dense leg: brute {f['dense_brute']['p50']:.3f} -> hnsw "
              f"{f['dense_hnsw']['p50']:.3f} ms  ({f['hnsw_speedup_dense']:.2f}x)")
        print(f"  full recall(): brute {f['full_brute']['p50']:.3f} -> hnsw "
              f"{f['full_hnsw']['p50']:.3f} ms "
              f"({(f['full_hnsw']['p50'] / f['full_brute']['p50'] - 1) * 100:+.1f}%)")
        print(f"  lexical leg alone: {f['lexical']['p50']:.3f} ms")
        print(f"  dense+lexical sum: {f['sum_of_legs']:.3f} ms")
        print(f"  MEASURED residual (fuse/hydrate/write/rerank): "
              f"{f['measured_residual']:+.3f} ms "
              f"({f['measured_residual'] / f['full_hnsw']['p50'] * 100:+.1f}% of full)")
        print(f"  build cost: brute {f['build_brute_s']:.1f}s  hnsw {f['build_hnsw_s']:.1f}s")
    print("\nWrite these numbers into docs/audit/A2_PERFORMANCE.md PERF-A2-07,")
    print("replacing the flagged hypothesis. Cite the script; it is reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
