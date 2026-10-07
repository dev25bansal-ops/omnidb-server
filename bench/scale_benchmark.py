"""Scaling benchmark: measure latency vs graph size for graph leg expansion.

This addresses critical review finding C3 by providing actual measurements of how
wall-clock latency scales with graph size, distinguishing algorithmic bounds from
SQL overhead.

Usage:
    python -m bench.scale_benchmark --json out.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3 as db
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server.config import Config


def create_synthetic_home(path: Path, n_nodes: int) -> tuple[list[int], int]:
    """Create synthetic home with given number of connected memories via raw SQL."""
    path.mkdir(parents=True, exist_ok=True)

    conn = db.connect(str(path / "omnidb.db"))
    now_ms = int(time.time() * 1000)

    # Create schema (minimal version, no vectors needed for this benchmark)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            content TEXT, kind TEXT, metadata TEXT, created_ms INTEGER, access_count INTEGER, tenant_id TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            src INTEGER, dst INTEGER, label TEXT, created_ms INTEGER,
            PRIMARY KEY(src, dst), FOREIGN KEY(src) REFERENCES memories(id), FOREIGN KEY(dst) REFERENCES memories(id)
        )
    """)
    conn.commit()

    ids = []
    for i in range(n_nodes):
        cur = conn.execute(
            "INSERT INTO memories (content, kind, created_ms, access_count, tenant_id) VALUES (?, ?, ?, ?, ?)",
            ("synthetic memory node", "semantic", now_ms, 0, "default"),
        )
        ids.append(cur.lastrowid)

    # Create random edges
    rng = np.random.default_rng(seed=42)
    edges_added = 0
    max_edges = min(n_nodes * 5, len(ids) * (len(ids) - 1))
    attempts = 0

    while edges_added < max_edges and attempts < max_edges * 3:
        u = rng.integers(0, len(ids))
        v = rng.integers(0, len(ids))
        if u != v:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (src, dst, label, created_ms) VALUES (?, ?, ?, ?)",
                    (ids[u], ids[v], "related", now_ms),
                )
                edges_added += 1
            except Exception:
                pass
        attempts += 1

    conn.close()
    return ids, edges_added


def measure_expansion_latency(db_path: Path, seed_id: int, budget: int, n_trials: int = 200) -> dict:
    """Time graph_ranking calls using same SQL pattern as store.graph_ranking."""
    conn = db.connect(str(db_path / "omnidb.db"))

    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()

        # Execute the same SQL pattern used by store.graph_ranking:
        # level-by-level bounded expansion with both-direction edges
        cur = conn.execute("""
            WITH RECURSIVE
                frontier(level, src, dst) AS (
                    SELECT 1, src, dst FROM edges WHERE src = ?
                    UNION ALL
                    SELECT frontier.level + 1, frontier.dst, edges.dst
                    FROM frontier JOIN edges ON frontier.dst = edges.src
                    WHERE frontier.level < 2
                ),
                backfrontier(level, dst, src) AS (
                    SELECT 1, dst, src FROM edges WHERE dst = ?
                    UNION ALL
                    SELECT backfrontier.level + 1, backfrontier.dst, edges.src
                    FROM backfrontier JOIN edges ON backfrontier.dst = edges.dst
                    WHERE backfrontier.level < 2
                ),
                all_reachable AS (
                    SELECT dst AS neighbor FROM frontier WHERE level > 0
                    UNION
                    SELECT src AS neighbor FROM backfrontier WHERE level > 0
                )
            SELECT DISTINCT neighbor FROM all_reachable LIMIT ?
        """, (seed_id, seed_id, budget)).fetchall()

        latency = (time.perf_counter() - t0) * 1000  # Convert to ms
        latencies.append(latency)

    conn.close()

    ordered = sorted(latencies)
    n = len(ordered)
    return {
        "mean_ms": sum(latencies) / n,
        "median_ms": ordered[n // 2],
        "p95_ms": ordered[min(n - 1, int(n * 0.95))],
        "p99_ms": ordered[min(n - 1, int(n * 0.99))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "n_trials": n,
    }


def run_scaling_benchmark(tmp_dir: Path, sizes: list[int]) -> list[dict]:
    """Run benchmark across multiple graph sizes."""
    results: list[dict] = []

    for n_nodes in sizes:
        print(f"Creating synthetic graph with {n_nodes:,} nodes...")
        home_path = tmp_dir / f"home_{n_nodes}_nodes"

        # Clean previous run
        import shutil
        if home_path.exists():
            shutil.rmtree(home_path)

        ids, edges_added = create_synthetic_home(home_path, n_nodes)
        seed_id = ids[0] if ids else 1

        print(f"Edges added: {edges_added:,}")

        stats = measure_expansion_latency(home_path, seed_id=seed_id, budget=256, n_trials=200)
        stats["n_nodes"] = n_nodes
        stats["n_edges"] = edges_added

        print(f"Latency: mean={stats['mean_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms")
        results.append(stats)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="scale-bench-"))

    print("Running scaling benchmark...")
    sizes = [10_000, 50_000, 100_000, 200_000]
    results = run_scaling_benchmark(tmp_dir, sizes)

    output = {
        "cells": results,
        "metadata": {
            "budget": 256,
            "trials": 200,
            "sizes_tested": sizes,
            "note": "Measures SQL adjacency lookup cost for bounded best-first expansion"
        },
    }

    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
