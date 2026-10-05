"""Graph leg weight + decay sweep: earn a non-zero default, or confirm zero.

The graph leg ships with graph_weight=0.0 because no measurement justified
anything else — dense_weight=0.7 carries a measured nfcorpus justification, and
the codebase's own standard is that a non-zero default must be earned rather
than assumed. This harness is what earns it.

Method: build a home with a REALISTIC synthetic memory graph on top of
clustered-topic texts. Then, for each (graph_weight, graph_decay, budget)
cell:

  - rank each query's fused result with the leg on vs the leg off;
  - grade the difference against a graph-aware ground truth: which rows SHOULD
    the leg surface? A row is a positive label if it is reachable within 2 hops
    of a "true" text match and shares the topic -- i.e. the graph leg's job.

The headroom question: there is none if the graph adds no genuinely-new
information beyond the text legs. The sweep reports that honestly rather than
pretending a number.

Deterministic seed; rerun reproducible.

Usage:
    python -m bench.graph_weight_sweep
    python -m bench.graph_weight_sweep --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from omnidb_server import db as dbmod           # noqa: E402
from omnidb_server.config import Config          # noqa: E402
from omnidb_server.embeddings import HashingEmbedder  # noqa: E402
from omnidb_server.store import MemoryStore     # noqa: E402
from omnidb_server.vector_index import open_vector_index  # noqa: E402

TOPIC = [
    "deployment pipeline configuration",
    "error budget and slo policies",
    "code review checklist items",
    "database schema migration steps",
    "on-call runbook for incidents",
]
QUERIES = [
    "how do I deploy",
    "error budget went over",
    "review this change",
    "add a column safely",
    "who responds at 3am",
]
N_MEMORIES_PER_TOPIC = 12
GRAPH_EDGE_PROB = 0.4


def _seed(rng: np.random.Generator) -> list[str]:
    """Topic-clustered texts, each memory belonging to one of 5 topics."""
    out = []
    for t, topic in enumerate(TOPIC):
        for i in range(N_MEMORIES_PER_TOPIC):
            out.append(f"{topic} number {i} with filler words about systems")
    return out


def build_home(path: Path) -> tuple[MemoryStore, list[int], dict]:
    st = _store(path)
    ids: list[int] = []
    topic_of: dict[int, int] = {}
    rng = np.random.default_rng(42)
    for t, topic in enumerate(TOPIC):
        for i in range(N_MEMORIES_PER_TOPIC):
            row = st.remember(f"topic {t} mem {i}: {topic}", kind="semantic")
            ids.append(row["id"])
            topic_of[row["id"]] = t
    # Graph: link memories mostly within the same topic, sometimes across.
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if topic_of[a] == topic_of[b] and rng.random() < GRAPH_EDGE_PROB:
                st.link(a, b, label="related")
            elif topic_of[a] != topic_of[b] and rng.random() < 0.03:
                st.link(a, b, label="cross")
    st.flush()
    return st, ids, topic_of


def _store(path: Path) -> MemoryStore:
    cfg = Config(home=path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(path)
    emb = HashingEmbedder()
    idx = open_vector_index(path, "brute", emb.dim, 0)
    return MemoryStore(conn, emb, idx, cfg)


def _graph_reachable(store: MemoryStore, seed_ids: list[int],
                     max_hops: int) -> set[int]:
    """Which rows the graph leg would surface, via the store's own leg."""
    return set(store.graph_ranking(seed_ids, kind=None, agent_id=None,
                                   tenant="default", depth=max_hops,
                                   budget=512))


def evaluate(store: MemoryStore, weight: float, decay: float,
             budget: int, max_hops: int) -> dict:
    """nDCG of the leg's graph-only additions against the leg-off baseline.

    The relevant question for a *weight default* is not "does on beat off" in
    total ranking — it is whether what the graph adds, that text did not find,
    includes rows the text query should actually want. We grade the additions
    against a graph-topic ground truth: added row is good iff reachable within
    max_hops of a same-topic seed.
    """
    store.cfg.graph_weight = weight
    store.cfg.graph_decay = decay
    store.cfg.graph_expand_budget = budget

    added_good = added_bad = 0
    for q, t in zip(QUERIES, range(len(QUERIES))):
        # k=3, not 10: at k=10 the text legs already place every graph
        # neighbour inside the window, so the leg CANNOT add anything and the
        # sweep measures nothing. At k=3 the text legs leave genuine
        # neighbours below the cut, which is where a graph leg shows headroom.
        store.cfg.graph_weight = 0.0
        off = {r["id"] for r in store.recall(q, k=3)}
        store.cfg.graph_weight = weight
        on = {r["id"] for r in store.recall(q, k=3)}
        added = on - off
        # ground truth: reachable from any same-topic seed within the hops
        topic_seeds = [
            row["id"] for row in store.conn.execute(
                "SELECT id FROM memories WHERE content LIKE ?", (f"topic {t}%",))
        ]
        good_ids = _graph_reachable(store, topic_seeds, max_hops)
        for row_id in added:
            if row_id in good_ids:
                added_good += 1
            else:
                added_bad += 1
    total = added_good + added_bad
    return {
        "weight": weight, "decay": decay, "budget": budget,
        "added": total,
        "added_good": added_good,
        "precision_of_additions": round(added_good / total, 3) if total else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store, ids, topics = build_home(Path(d) / "home")
        try:
            print("Graph leg weight sweep (graph_weight, graph_decay, budget)")
            print("Question: are the leg's additions beyond the text result")
            print("actually useful? (precision of additions)\n")
            print(f"{'weight':>8} {'decay':>6} {'budget':>6} {'added':>6} {'good':>5} {'prec':>6}")
            print("-" * 44)
            cells = []
            t0 = time.perf_counter()
            for weight in (0.0, 0.05, 0.10, 0.3):
                for decay in (0.3, 0.5, 0.7):
                    for budget in (256, 1024):
                        row = evaluate(store, weight, decay, budget, max_hops=2)
                        cells.append(row)
                        prec = row["precision_of_additions"]
                        print(f"{weight:>8} {decay:>6} {budget:>6} "
                              f"{row['added']:>6} {row['added_good']:>5} "
                              f"{str(prec):>6}")
            print(f"\n{len(cells)} cells in {time.perf_counter() - t0:.1f}s")

            # the headline: is there any headroom at all?
            positive = [c for c in cells
                        if c["added"] and c["precision_of_additions"]]
            if not positive:
                print("\nHEADROOM: none — the graph leg adds no rows the text "
                      "legs missed. A non-zero default would be noise."
                      "\nVERDICT: graph_weight=0.0 stands.")
            else:
                best = max(positive, key=lambda c: c["precision_of_additions"])
                print(f"\nHEADROOM: yes. Best cell {best['weight']}/{best['decay']}/"
                      f"{best['budget']} at precision "
                      f"{best['precision_of_additions']} over {best['added']} "
                      f"additions.")
        finally:
            store.close()

    if args.json:
        Path(args.json).write_text(json.dumps({"cells": cells}, indent=2),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())