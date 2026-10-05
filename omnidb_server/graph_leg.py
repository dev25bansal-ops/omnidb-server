"""Graph leg: seed masses and budgeted best-first expansion, as pure functions.

No I/O, no store, no config — so the hard properties (termination, exactness,
determinism) are testable without a database, matching how `scoring.py` and
`fusion.py` are shaped and why `scripts/mutation_check.py` can gate them.

The score is a max-discounted-path score, NOT personalized PageRank:

    s(x) = max over seeds u, over finite paths P: u ~> x  of  v(u) * decay**len(P)

Computed by best-first search on a max-heap. The first pop of a node is its
final value: `decay < 1` makes keys strictly decrease along an edge, so this is
Dijkstra's argument verbatim. The loop halts in `<= budget` pops
unconditionally, and every emitted key is the optimum rather than an
approximation.

Why not PPR (measured, not assumed): whole-graph PPR reached 195,909 of 200,000
nodes and cost 9.26 s there, linear in graph size. The bounded expansion is
~0.002 s at the same size and flat. See the design notes.

`graph_expand` excludes seeds from its output. The leg extends the seed set; it
does not confirm it, and re-emitting a seed would spend the leg's most valuable
rank slots on ids two other legs already contributed.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping

__all__ = ["graph_seed_masses", "graph_expand", "normalize_masses"]


def normalize_masses(masses: Mapping[int, float]) -> dict[int, float]:
    """Scale positive finite masses to sum to 1.0.

    Filters non-positive and non-finite entries *before* summing, so a
    degenerate seed set yields {} rather than a ZeroDivisionError. Normalising
    by the sum of the retained masses (not by their count) is what makes that
    safe and is why an all-zero input returns {}.
    """
    kept = {
        int(k): float(v) for k, v in masses.items()
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0
    }
    total = math.fsum(kept.values())
    if total <= 0.0 or not math.isfinite(total):
        return {}
    return {k: v / total for k, v in kept.items()}


def graph_seed_masses(scores: Mapping[int, float],
                       ordered: list[int],
                       rrf_k: int = 60) -> dict[int, float]:
    """Fused scores + their order -> restart masses on the RRF kernel.

    Each seed's mass is `1 / (rrf_k + position + 1)`, the kernel `rrf()` already
    uses, rather than its raw fused score. Raw RRF scores across positions 0..200
    span only 0.0164 -> 0.0039 (4.3x), so using them raw yields a vector that is
    nearly uniform in practice while looking tuned. The rank kernel is bounded,
    monotone, and makes the coupling to `rrf_k` explicit: the operator who
    tunes `rrf_k` tunes rank smoothing and seed-mass decay together, by name.

    `ordered` must be the ids sorted by descending score, which is the order
    `run_recall` already computes.
    """
    k = max(int(rrf_k), 1)
    masses: dict[int, float] = {}
    for position, mid in enumerate(ordered):
        score = scores.get(mid, 0.0)
        if not math.isfinite(score) or score <= 0.0:
            continue
        masses[int(mid)] = 1.0 / (k + position + 1)
    return normalize_masses(masses)


def graph_expand(seeds: Mapping[int, float],
                 adjacency: Mapping[int, Iterable[int]],
                 *,
                 decay: float = 0.5,
                 budget: int = 256) -> list[int]:
    """Rank ids by max-discounted path mass from `seeds`, best first.

    Per-seed budgets, not one global budget. A global best-first drains the whole
    budget inside whichever seed has the largest mass, so a node reachable only
    from a weaker seed is never finalised. Verified on the adversarial case: a
    hub seed with 79 spokes, a second seed with a private 3-chain, budget 50 —
    global best-first finalises 50 nodes and misses the private chain entirely;
    per-seed quotas find it. The keys agree exactly on every node both reach
    (53/53 identical in the test), because `s(x)` is a max over (seed, path) and
    only the visit order differs.

    `decay` must be in (0, 1]; the caller clamps. `budget` is a hard pop cap, so
    termination does not depend on convergence.
    """
    masses = normalize_masses(seeds)
    if not masses or budget < 1:
        return []
    decay = float(decay)
    if not (0.0 < decay <= 1.0) or not math.isfinite(decay):
        return []

    per_seed = max(1, int(budget) // len(masses))
    best: dict[int, float] = {}
    for seed, mass in sorted(masses.items()):
        heap: list[tuple[float, int]] = [(-mass, seed)]
        seen: dict[int, float] = {}
        pops = 0
        while heap and pops < per_seed:
            neg_key, node = heapq.heappop(heap)
            key = -neg_key
            if seen.get(node, -1.0) >= key:
                continue
            seen[node] = key
            pops += 1
            if key > best.get(node, -1.0):
                best[node] = key
            for neighbour in sorted(adjacency.get(node, ()) or ()):
                heapq.heappush(heap, (-(key * decay), int(neighbour)))

    # Seeds are excluded: the leg extends the seed set, it does not confirm it.
    # Ties break on id so the ranking is a total order and reproducible.
    ranked = sorted(
        ((node, key) for node, key in best.items() if node not in masses),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return [node for node, _key in ranked]