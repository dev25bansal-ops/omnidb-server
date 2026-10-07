"""Verify the two pinned arithmetic claims in the addendum that had no artifact.

1. The cap crossover: with w_d=0.7, w_g=0.10, k_RRF=60, a graph-only rank-0 row
   ties a dense-only row at dense rank 366 and strictly beats it from 367.

2. The PPR restart arithmetic: on the 3-node path 0-1-2 the transition matrix's
   stationary distribution is 0.25/0.50/0.25, and the personalized (restart)
   iteration converges to a different, seed-influenced distribution --- which is
   why omitting the restart term flipped the design recommendation.

Both are computed exactly here rather than left as prose.

Usage:
    python -m bench.arithmetic_pins --json bench/arithmetic-pins.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RRF_K = 60
W_DENSE, W_SPARSE, W_GRAPH = 0.7, 0.3, 0.10
DAMPING = 0.85

# The path graph 0 - 1 - 2 as P[from][to]. 0 -> 1, 2 -> 1, and 1 splits evenly.
P = [
    [0.0, 1.0, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 1.0, 0.0],
]


def rrf_term(weight: float, position: int) -> float:
    return weight / (RRF_K + position + 1)


def crossover_rank() -> dict:
    graph0 = rrf_term(W_GRAPH, 0)
    tie = (RRF_K + 1) * ((W_DENSE / W_GRAPH) - 1)
    first_beaten = next(
        (p for p in range(0, 5000) if rrf_term(W_DENSE, p) < graph0), None)
    expected = round(tie) + 1
    return {
        "graph_only_rank0_term": round(graph0, 8),
        "tie_rank_formula": round(tie, 6),
        "tie_rank_rounded": round(tie),
        "first_dense_rank_strictly_beaten": first_beaten,
        "consistent": first_beaten == expected,
    }


def stationary_distribution() -> list[float]:
    """Solve pi = P^T pi exactly for the 3-node path (no restart, no damping)."""
    # balance: pi_1 = pi_0 + pi_2 (all mass enters 1), pi_0 = 0.5 pi_1,
    # pi_2 = 0.5 pi_1; with sum = 1 -> pi_1 = 0.5, pi_0 = pi_2 = 0.25.
    pi1 = 0.5
    pi0 = 0.5 * pi1
    pi2 = 0.5 * pi1
    total = pi0 + pi1 + pi2
    return [round(pi0 / total, 6), round(pi1 / total, 6), round(pi2 / total, 6)]


def ppr(damping: float, restart: bool, iters: int = 2000) -> list[float]:
    """Personalized PageRank from seed 0, or the plain damped walk without it.

    Uses the standard damped form r <- d * P r + (1-d) * p, which is what the
    shipped prototype was missing. The restart is what keeps the seed present;
    without it the iteration collapses onto a distribution the seed cannot
    influence (and, on this period-2 chain, one that oscillates between
    [0.5,0,0.5] and [0,1,0] rather than settling).
    """
    r = [1.0, 0.0, 0.0]
    seed = [1.0, 0.0, 0.0]
    for _ in range(iters):
        nxt = [0.0, 0.0, 0.0]
        for j in range(3):
            for i in range(3):
                nxt[i] += P[j][i] * r[j]
        if restart:
            nxt = [damping * x for x in nxt]
            nxt = [nxt[i] + (1 - damping) * seed[i] for i in range(3)]
        r = nxt
    return [round(x, 6) for x in r]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    out = {
        "rrf_constant": RRF_K,
        "weights": {"dense": W_DENSE, "sparse": W_SPARSE, "graph": W_GRAPH},
        "crossover": crossover_rank(),
        "ppr_3node_path": {
            "damping": DAMPING,
            "stationary_distribution": stationary_distribution(),
            "with_restart_2000_iters": ppr(DAMPING, restart=True),
            "without_restart_2000_iters": ppr(DAMPING, restart=False),
            "addendum_claims": {
                "stationary": [0.25, 0.50, 0.25],
                "with_restart": [0.345, 0.459, 0.195],
            },
        },
    }

    c = out["crossover"]
    print(f"crossover: graph rank-0 term {c['graph_only_rank0_term']:.8f}")
    print(f"  tie at rank {c['tie_rank_rounded']} (formula {c['tie_rank_formula']}), "
          f"first strictly beaten {c['first_dense_rank_strictly_beaten']} "
          f"-> consistent: {c['consistent']}")
    p = out["ppr_3node_path"]
    print(f"stationary distribution:   {p['stationary_distribution']}  "
          f"(claimed {p['addendum_claims']['stationary']})")
    print(f"with restart (2000 iters): {p['with_restart_2000_iters']}  "
          f"(claimed {p['addendum_claims']['with_restart']})")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())