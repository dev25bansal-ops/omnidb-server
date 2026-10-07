"""Re-measure the budgeted graph leg on a 200k-node / 1M-edge store.

The addendum's cost table (tab:graphcost) reported 0.0018 s and "419 finalized"
for the budgeted arm, with no artifact in the repo backing either number; the
PPR prototype it is compared against is likewise not in the released tree.
This script produces the number we CAN reproduce today: the budgeted
best-first expansion on the same graph shape, over 200 trials, reporting the
pop count and finalized-node count explicitly so the distinction the paper
draws is visible in the data.

Usage:
    python -m bench.graph_cost --json bench/graph-cost-results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server.graph_leg import graph_expand, graph_seed_masses  # noqa: E402

NODES = 200_000
OUT_DEGREE = 5          # stored adjacency rows per node, as in the scaling bench
BUDGET = 256
RRF_K = 60
DECAY = 0.5
TRIALS = 200


def build_adjacency(n_nodes: int, out_degree: int, seed: int = 42):
    """A synthetic adjacency of the same shape the sweeping benches use."""
    import numpy as np
    rng = np.random.default_rng(seed)
    adj: dict[int, set[int]] = {}
    for i in range(n_nodes):
        targets = rng.integers(0, n_nodes, size=out_degree)
        adj[i] = {int(t) for t in targets if int(t) != i}
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    print(f"building {NODES:,}-node adjacency (out-degree {OUT_DEGREE}) ...")
    t0 = time.perf_counter()
    adj = build_adjacency(NODES, OUT_DEGREE)
    build_s = time.perf_counter() - t0
    n_edges = sum(len(v) for v in adj.values())
    print(f"  built in {build_s:.1f}s, {n_edges:,} directed adjacency rows")

    seeds = [0, 1, 2, 3, 4]
    masses = graph_seed_masses({s: 1.0 for s in seeds}, seeds, rrf_k=RRF_K)

    lat, pops = [], []
    finalized = 0
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        out = graph_expand(masses, adj, decay=DECAY, budget=BUDGET)
        lat.append((time.perf_counter() - t0) * 1000.0)
        finalized = len(out)
    lat.sort()
    n = len(lat)
    result = {
        "nodes": NODES,
        "directed_adjacency_rows": n_edges,
        "out_degree": OUT_DEGREE,
        "budget": BUDGET,
        "seeds": len(seeds),
        "decay": DECAY,
        "trials": n,
        "median_ms": round(lat[n // 2], 4),
        "p95_ms": round(lat[min(n - 1, int(n * 0.95))], 4),
        "p99_ms": round(lat[min(n - 1, int(n * 0.99))], 4),
        "min_ms": round(lat[0], 4),
        "max_ms": round(lat[-1], 4),
        "finalized_nodes": finalized,
        "note": (
            "'finalized_nodes' is the leg's output size at this budget "
            "(nodes whose final score was set and that are not seeds). "
            "It is not a pop count: the loop executes at most `budget` pops "
            "across all seeds, and seeds are excluded from the output."
        ),
    }
    print(f"  median {result['median_ms']:.3f} ms, p95 {result['p95_ms']:.3f}, "
          f"p99 {result['p99_ms']:.3f} over {n} trials")
    print(f"  finalized nodes emitted: {finalized}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())