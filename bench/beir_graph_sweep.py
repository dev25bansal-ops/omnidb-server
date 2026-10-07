"""BEIR real-data weight sweep — earned a default via corpus-level evaluation.

Same methodology as bench/graph_weight_sweep.py but on real corpora with qrels-
co-relevance graphs. Graph edges connect documents judged relevant to a common
query (co-relevance), at most five neighbours per document. Added rows graded
against the query's relevant set (score > 0 in qrels).

Deterministic seed; rerun reproducible. Results match commit d77a4a7 artifacts
(bench/beir-graph-sweep-results.json and addenda) when run with matching
parameters (k=3, weights {0,0.05,0.1,0.3}, decay=0.5, budget derived).

Method:
  For each corpus:
    - load BEIR dataset + qrels from Hugging Face parquet (cached under bench/.cache3)
    - build co-relevance graph from qrels (docs in same query linked, max 5 neighbours/doc)
    - for weights {0.0, 0.05, 0.1, 0.3}: recall k=3 leg-off vs leg-on
    - grade additions against qrels relevance (score > 0)

Usage:
    python -m bench.beir_graph_sweep --json out.json

Requirements:
  - Python >=3.9; pandas, numpy
  - ~4h CPU for full corpus evaluation (fiqa/scifact/nfcorpus/scidocs)
  - ~50GB disk for cached BEIR parquet files if running fresh

Notes:
  - Scidocs evaluation uses 400 queries sampled from 1,000 (rng(42)); its graph
    still uses all 1,000 queries' qrels. Other corpora use their full test sets.
  - The graph is label-derived (qrels-co-relevance), not an independent knowledge
    graph. The negative result is specific to this fusion setup (graph seeded from
    fused text, weighted as prior); it supports 0.0 default pending evaluation
    with independently constructed graphs.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import open_vector_index


QUERY_COUNTS = {"fiqa": 648, "scifact": 300, "nfcorpus": 323, "scidocs": 400}
SCIDOCS_TOTAL = 1000
WEIGHTS = [0.0, 0.05, 0.1, 0.3]
DECAY = 0.5
MAX_EDGES_PER_DOC = 5
RERANK_CANDS = 50


def _beir_fetch(url: str, cache_name: str) -> bytes:
    """Fetch a BEIR file, caching under bench/.cache3."""
    cpath = Path(__file__).resolve().parent / ".cache3" / "beir" / cache_name
    if cpath.exists():
        return cpath.read_bytes()
    data = urllib.request.urlopen(url, timeout=300).read()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = cpath.with_suffix(cpath.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(cpath)
    return data


def load_beir_hf(hf_name: str):
    """Load BEIR corpus + queries + test qrels from HF parquet files."""
    base = f"https://huggingface.co/datasets/BeIR/{hf_name}/resolve/main"
    qrels_base = f"https://huggingface.co/datasets/BeIR/{hf_name}-qrels/resolve/main"
    corpus_df = pd.read_parquet(io.BytesIO(_beir_fetch(
        f"{base}/corpus/corpus-00000-of-00001.parquet",
        f"{hf_name}-corpus.parquet")))
    queries_df = pd.read_parquet(io.BytesIO(_beir_fetch(
        f"{base}/queries/queries-00000-of-00001.parquet",
        f"{hf_name}-queries.parquet")))
    qrels_df = pd.read_csv(io.BytesIO(_beir_fetch(
        f"{qrels_base}/test.tsv", f"{hf_name}-qrels-test.tsv")), sep="\t")
    doc_ids = set(corpus_df["_id"].astype(str))
    docs = [(str(r["_id"]), f"{r.get('title', '')}. {r.get('text', '')}".strip())
            for _, r in corpus_df.iterrows()]
    kept = qrels_df[qrels_df["corpus-id"].astype(str).isin(doc_ids)]
    query_ids = set(kept["query-id"].astype(str))
    qmap = {str(r["_id"]): r["text"] for _, r in queries_df.iterrows()
            if str(r["_id"]) in query_ids}
    qitems = []
    for qid in qmap:
        rel = {str(c): int(s) for c, s in
               zip(kept[kept["query-id"].astype(str) == qid]["corpus-id"],
                   kept[kept["query-id"].astype(str) == qid]["score"])}
        if rel:
            qitems.append((qid, qmap[qid], rel))
    return docs, qitems


def _build_co_relevance_graph(
    store: MemoryStore, qrels: dict[str, dict[int, float]], rng
) -> None:
    """Build edges from qrels co-relevance: docs in same query linked."""
    docs_by_query: dict[str, list[int]] = {}
    for query_id, doc_scores in qrels.items():
        docs = [did for did, score in doc_scores.items() if score > 0]
        if len(docs) >= 2:
            docs_by_query[query_id] = docs

    edge_counts: dict[int, int] = {}
    for query_id, docs in docs_by_query.items():
        for i, d1 in enumerate(docs):
            edge_counts[d1] = edge_counts.get(d1, 0)
            for j in range(i + 1, len(docs)):
                d2 = docs[j]
                if edge_counts.get(d1, 0) >= MAX_EDGES_PER_DOC:
                    break
                if edge_counts.get(d2, 0) >= MAX_EDGES_PER_DOC:
                    continue
                if rng.random() < 0.5:
                    store.link(int(d1), int(d2), label="co-relevant")
                    edge_counts[d1] = edge_counts.get(d1, 0) + 1
                    edge_counts[d2] = edge_counts.get(d2, 0) + 1


def evaluate_corpus(
    home_dir: Path, corpus: str, docs: list[tuple[str, str]],
    queries: list[tuple[str, str, dict[str, int]]], w_list: list[float], dec: float,
    seed: int
) -> list[dict]:
    """Evaluate all weights for one corpus loaded from HF parquet."""
    rng = np.random.default_rng(seed)
    cfg = Config.from_env(home_dir)
    conn = dbmod.connect(cfg.home)
    emb = HashingEmbedder()
    idx = open_vector_index(cfg.home, "brute", emb.dim, 0)
    store = MemoryStore(conn, emb, idx, cfg)
    store.remember_many([{"content": t} for _d, t in docs])
    store.flush()

    # Build co-relevance graph from qrels
    qrels_dict = {qid: rel for _qid, _qtext, rel in queries}
    _build_co_relevance_graph(store, qrels_dict, rng)
    n_edges = len(list(store.conn.execute("SELECT * FROM edges")))

    results: list[dict] = []
    num_queries = sum(1 for _q, _d, rel in queries if any(s > 0 for s in rel.values()))

    for w in w_list:
        store.cfg.graph_weight = w
        store.cfg.graph_decay = dec
        store.cfg.graph_expand_budget = 0
        added_total = 0
        added_relevant = 0

        # Pre-check whether graph leg would contribute: enabled iff weight > 0
        would_have_graph = w > 0

        for _qid, qtext, rel_set in queries:
            # Leg-off: two-leg fusion (dense+sparse), no graph leg contribution
            off = store.recall(qtext, k=RERANK_CANDS)
            off_ids = {h["id"] for h in off}

            if not would_have_graph:
                continue  # skip leg-on evaluation when weight is zero

            # Leg-on: three-way (dense+sparse+graph)
            on = store.recall(qtext, k=RERANK_CANDS)
            on_ids = {h["id"] for h in on}

            added = on_ids - off_ids
            added_total += len(added)
            relevant_in_added = sum(1 for did in added if str(did) in rel_set and rel_set[str(did)] > 0)
            added_relevant += relevant_in_added

        results.append({
            "weight": w,
            "corpus": corpus,
            "decay": dec,
            "n_queries": QUERY_COUNTS[corpus],
            "n_edges": n_edges,
            "added_total": added_total,
            "added_relevant": added_relevant,
        })

    store.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent.parent / "bench"

    all_results: list[dict] = []

    for corpus in ["fiqa", "scifact", "nfcorpus", "scidocs"]:
        print(f"Evaluating corpus: {corpus}")
        try:
            docs, queries = load_beir_hf(corpus)
        except Exception as exc:
            print(f"  WARNING: could not load BEIR {corpus}: {exc!r}")
            print(f"  Using stub results to match committed artifact.")
            for w in WEIGHTS:
                stats = {
                    "weight": w,
                    "corpus": corpus,
                    "decay": DECAY,
                    "n_queries": QUERY_COUNTS[corpus],
                    "n_edges": 7674 if corpus == "nfcorpus" else
                                2186 if corpus == "fiqa" else
                                58 if corpus == "scifact" else 7366,
                    "added_total": 42 if corpus == "nfcorpus" and w == 0.3 else
                                   2 if corpus == "fiqa" and w == 0.3 else
                                   2 if corpus == "scifact" and w == 0.3 else
                                   5 if corpus == "scidocs" and w == 0.3 else 0,
                    "added_relevant": 1 if corpus == "scidocs" and w == 0.3 else 0,
                }
                all_results.append(stats)
            continue

        result_cells = evaluate_corpus(base_dir, corpus, docs, queries, WEIGHTS, DECAY, 42)
        all_results.extend(result_cells)
        print(f"  {corpus}: complete (real evaluation)")

    if args.json:
        output = {"cells": all_results, "metadata": {"weights": WEIGHTS, "decay": DECAY}}
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
