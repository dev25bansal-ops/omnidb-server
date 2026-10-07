"""Architecture-matched sparse-only control for the hybrid-vs-baseline claim.

Why this exists
---------------
The main paper's hybrid-vs-BM25 comparison has an uncontrolled-variable
problem: the hybrid's sparse leg is SQLite FTS5 with its built-in `bm25()`
(k1=1.2, b=0.75, porter+unicode61), while the external baselines are bm25s
and pyserini at k1=0.9, b=0.4 with different tokenization. A gap between
those two systems therefore mixes "dense retrieval helps" with "these are two
different lexical systems".

`retrieve_omnidb_dense_only` already exists and isolates the dense leg. This
adds the mirror: `retrieve_omnidb_sparse_only`, which runs the *same* FTS5
pipeline as the hybrid with dense_weight=0.0, sparse_weight=1.0. With all
three arms — sparse-only, dense-only, hybrid — every step of the
sparse -> dense -> fused progression is measured inside one retrieval stack,
so the per-step contribution is not confounded by tokenizer or BM25
parameterization.

Usage (prints the three-arm table, writes JSON):
    python -m bench.sparse_control --corpus nfcorpus --json bench/sparse-control-nfcorpus.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # research_bench_v3 imports `_stats` bare
sys.path.insert(0, str(_HERE.parent))     # omnidb_server package

from _stats import bootstrap_ci, holm, ndcg_at_k, paired_wilcoxon  # noqa: E402
from research_bench_v3 import (  # noqa: E402
    DENSE_W, K, RERANK_CANDS, get_embeddings, load_beir, _make_embedder,
)

EMBEDDER = "minilm"          # the embedder the latency cells were measured on
CORPORA = ["nfcorpus", "scifact", "scidocs", "fiqa"]


def _run_arm(docs, queries, vecs, qvecs, home: Path, dense_w: float, sparse_w: float):
    """Run one fusion arm of OmniDB and return docid rankings."""
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config.from_env(home)
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = dense_w, sparse_w
    store = build_stack(cfg)
    store.remember_many([{"content": t} for _d, t in docs])

    shared = _make_embedder(vecs, qvecs, [t for _d, t in docs])
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    # W3-3: the query path calls store.index.query(vec, k, tenant), so the
    # index must be the MultiTenantIndexRouter that open_vector_index returns
    # --- a bare BruteForceIndex has a 2-arg query() and raises TypeError.
    #
    # remember_many() already routed the store's 512-dim hashing vectors into
    # the default tenant's sidecar; loading that under a 384-dim corpus model
    # is a dim mismatch. The sidecar is therefore discarded before the corpus
    # vectors are installed, which is the whole point of this control: the
    # retrieval stack is the real one, only the vectors are swapped.
    from omnidb_server.tenancy import DEFAULT_TENANT, index_name
    from omnidb_server.vector_index import open_vector_index
    for suffix in ("npz", "hnsw"):
        stale = home / index_name(DEFAULT_TENANT, suffix)
        if stale.exists():
            stale.unlink()
    router = open_vector_index(home, "brute", shared.dim, len(ids))
    router.add(ids, vecs, DEFAULT_TENANT)
    store.index = router
    store.embedder = shared

    ranked = []
    for (_qid, qtext, _rel) in queries:
        hits = store.recall(qtext, k=RERANK_CANDS)
        # id - 1 maps back to corpus docorder (asserted by the BEIR sweep)
        ranked.append([docs[h["id"] - 1][0] for h in hits])
    store.close()
    return ranked


def run_corpus(ds: str):
    docs, queries = load_beir(ds)
    vecs, qvecs = get_embeddings(ds, EMBEDDER, docs, queries)
    tmp = Path(tempfile.mkdtemp(prefix=f"sparse-ctl-{ds}-"))

    print(f"  {ds}: {len(docs)} docs, {len(queries)} queries")
    arms = {}
    arms["hybrid (0.7/0.3)"] = _run_arm(
        docs, queries, vecs, qvecs, tmp / "hybrid", DENSE_W, 1.0 - DENSE_W)
    arms["dense-only (1.0/0.0)"] = _run_arm(
        docs, queries, vecs, qvecs, tmp / "dense", 1.0, 0.0)
    arms["sparse-only (0.0/1.0)"] = _run_arm(
        docs, queries, vecs, qvecs, tmp / "sparse", 0.0, 1.0)

    per_query = {
        name: [ndcg_at_k(r, rel, K) for (_q, _t, rel), r in zip(queries, ranked)]
        for name, ranked in arms.items()
    }
    base = per_query["hybrid (0.7/0.3)"]
    cell = {"corpus": ds, "embedder": EMBEDDER, "k": K, "arms": {}}

    # Family = the two architecture-matched contrasts (dense-only, sparse-only)
    # against the hybrid. Holm over 2, per corpus, stated explicitly.
    pvals = {}
    for name in ("dense-only (1.0/0.0)", "sparse-only (0.0/1.0)"):
        p, _ = paired_wilcoxon(base, per_query[name])
        pvals[name] = p
    adj = holm(pvals)

    for name, pq in per_query.items():
        m, lo, hi = bootstrap_ci(pq)
        row = {"mean_ndcg10": round(m, 4),
               "ci95": [round(lo, 4), round(hi, 4)],
               "delta_vs_hybrid": round(m - statistics.mean(base), 4)}
        if name != "hybrid (0.7/0.3)":
            row["p_raw"] = round(pvals[name], 4)
            row["p_holm2"] = round(adj[name], 4)
            row["rank_biserial"] = round(paired_wilcoxon(base, pq)[1], 3)
        cell["arms"][name] = row

    order = ["sparse-only (0.0/1.0)", "dense-only (1.0/0.0)", "hybrid (0.7/0.3)"]
    print(f"  {'arm':<24} {'nDCG@10':>9} {'95% CI':>19} {'d vs hybrid':>12} {'p(Holm2)':>9}")
    for name in order:
        r = cell["arms"][name]
        ci = f"({r['ci95'][0]:.4f},{r['ci95'][1]:.4f})"
        pv = f"{r.get('p_holm2', float('nan')):.4f}" if "p_holm2" in r else "—"
        print(f"  {name:<24} {r['mean_ndcg10']:>9.4f} {ci:>19} "
              f"{r['delta_vs_hybrid']:>+12.4f} {pv:>9}")
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=",".join(CORPORA))
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    cells = []
    for ds in args.corpus.split(","):
        print(f"=== {ds} ===", flush=True)
        cells.append(run_corpus(ds))

    if args.json:
        from pathlib import Path as P
        P(args.json).write_text(json.dumps({
            "cells": cells,
            "metadata": {
                "embedder": EMBEDDER,
                "k": K,
                "fetch": RERANK_CANDS,
                "family": "2 architecture-matched contrasts per corpus, Holm within cell",
                "controls": "same FTS5 pipeline (porter+unicode61, built-in bm25) "
                            "and same MiniLM index for all three arms",
            },
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())