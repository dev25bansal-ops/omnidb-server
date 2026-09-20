"""Vector re-baseline harness (roadmap wk2-3).

The design doc's numbers were SIFT-1M 128-dim figures; this measures the real
thing instead: clustered synthetic embeddings at 384/768 dims, hnswlib
(M=16, efC=200) recall@10 vs exact brute-force ground truth across an ef sweep,
plus the SQ8-shortlist -> f32-rerank experiment. Writes bench/results.md and
exits nonzero if no ef>=96 configuration reaches recall@10 >= 0.95.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def make_dataset(n: int, dim: int, clusters: int = 32, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (clusters, dim)).astype(np.float32)
    assign = rng.integers(0, clusters, n)
    vectors = centers[assign] + rng.normal(0, 0.35, (n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors)


def brute_topk(matrix: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    sims = matrix @ query
    idx = np.argpartition(-sims, k)[:k]
    return set(idx[np.argsort(-sims[idx])].tolist())


def sq8_quantize(matrix: np.ndarray):
    scale = np.maximum(np.abs(matrix.max(axis=0)), np.abs(matrix.min(axis=0))) / 127.0
    scale[scale == 0] = 1e-12
    quantized = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)
    return quantized, scale


def main() -> int:
    try:
        import hnswlib
    except ImportError:
        hnswlib = None

    rows = []
    for dim in (384, 768):
        for n in (1_000, 10_000, 100_000):
            data = make_dataset(n, dim)
            queries = make_dataset(100, dim, seed=99)

            row: dict = {"dim": dim, "n": n}

            if hnswlib is not None:
                t0 = time.perf_counter()
                index = hnswlib.Index(space="ip", dim=dim)
                index.init_index(max_elements=n * 2, ef_construction=200, M=16)
                index.add_items(data, list(range(n)), num_threads=-1)
                row["build_s"] = round(time.perf_counter() - t0, 2)

                for ef in (32, 64, 96, 128):
                    index.set_ef(ef)
                    recalls, latencies = [], []
                    for qi in range(len(queries)):
                        t = time.perf_counter()
                        labels, _ = index.knn_query(queries[qi : qi + 1], k=10)
                        latencies.append(time.perf_counter() - t)
                        truth = brute_topk(data, queries[qi], 10)
                        recalls.append(len(truth & {int(x) for x in labels[0]}) / 10)
                    row[f"ef{ef}"] = {
                        "recall@10": round(float(np.mean(recalls)), 4),
                        "p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 3),
                        "p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 3),
                    }

            quantized, scale = sq8_quantize(data)
            q_int = np.clip(np.round(queries / scale), -127, 127).astype(np.int32)
            recalls, latencies = [], []
            for qi in range(len(queries)):
                t = time.perf_counter()
                scores = quantized.astype(np.int32) @ q_int[qi]
                candidates = np.argpartition(-scores, 50)[:50]
                reranked = candidates[np.argsort(-(data[candidates] @ queries[qi]))[:10]]
                latencies.append(time.perf_counter() - t)
                truth = brute_topk(data, queries[qi], 10)
                recalls.append(len(truth & set(reranked.tolist())) / 10)
            row["sq8_rerank"] = {
                "recall@10": round(float(np.mean(recalls)), 4),
                "p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 3),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    lines = [
        "# Vector baseline — synthetic clustered cosine, measured locally",
        "",
        f"- host: {platform.processor() or platform.machine()}, "
        f"python {sys.version.split()[0]}, numpy {np.__version__}",
        f"- hnswlib: {'available' if hnswlib is not None else 'NOT INSTALLED (ef rows omitted)'}",
        "- SIFT-1M/128-dim folklore deliberately not used; dims match real embedders",
        "",
        "| dim | n | build s | ef | recall@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for ef in (32, 64, 96, 128):
            d = r.get(f"ef{ef}")
            if d is None:
                lines.append(f"| {r['dim']} | {r['n']} | {r.get('build_s', '-')} | {ef} | n/a | - | - |")
                continue
            lines.append(
                f"| {r['dim']} | {r['n']} | {r.get('build_s', '-')} | {ef} "
                f"| {d['recall@10']} | {d['p50_ms']} | {d['p95_ms']} |"
            )
    lines += [
        "",
        "## SQ8 shortlist(50) + f32 rerank",
        "",
        "| dim | n | recall@10 | p50 ms |",
        "|---|---|---|---|",
    ]
    for r in rows:
        d = r["sq8_rerank"]
        lines.append(f"| {r['dim']} | {r['n']} | {d['recall@10']} | {d['p50_ms']} |")

    out = Path(__file__).with_name("results.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    hnsw_scores = [
        r[f"ef{ef}"]["recall@10"] for r in rows for ef in (96, 128) if f"ef{ef}" in r
    ]
    sq8_scores = [r["sq8_rerank"]["recall@10"] for r in rows]
    best = max(hnsw_scores + [min(sq8_scores)]) if hnsw_scores else min(sq8_scores)
    gate = best >= 0.95 if hnsw_scores else min(sq8_scores) >= 0.90
    print(f"GATE best recall@10: {best} -> pass={gate}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
