"""Leak-free BEIR graph-leg experiments with cached, reproducible baselines.

This runner replaces the earlier slow/invalid harnesses. It never opens a
SQLite store for retrieval: it uses the same weighted RRF and pure graph-leg
functions as OmniDB, with cached BEIR embeddings and bm25s rankings. That makes
runs finite (and avoids one SQLite commit/file-lock per edge).

Modes:
  train-test: construct qrels co-relevance edges from 80% of queries and grade
              graph additions on the held-out 20% only (C1 leakage fix).
  token:      construct an independent graph from exact token-overlap (Jaccard)
              using an inverted index; qrels are used only for grading.
  semantic:   construct an independent graph from cached miniLM document-vector
              nearest neighbours; qrels are used only for grading.

The experiment is exploratory: it reports additions, precision, displacement,
and nDCG@3 deltas for weights {0,.05,.1,.3}; it does not silently select a
production default from test labels.

Usage (the three commands that produce the committed artifacts):
  python -m bench.graph_ablation --mode train-test --embedding hashing \\
      --json bench/graph-ablation-train-test.json
  python -m bench.graph_ablation --mode token --embedding hashing \\
      --corpus nfcorpus,scifact --json bench/graph-ablation-token.json
  python -m bench.graph_ablation --mode semantic --embedding minilm \\
      --json bench/graph-ablation-semantic.json

Those three files are the source of tab:ablation in the addendum; every value
in the table reproduces from them. The token run covers nfcorpus and scifact
only (the prose states this scope rather than implying all four).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Import at runtime to support both package invocation and direct script execution
try:
    from omnidb_server.fusion import rrf
    from omnidb_server.graph_leg import graph_expand, graph_seed_masses
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from omnidb_server.fusion import rrf
    from omnidb_server.graph_leg import graph_expand, graph_seed_masses


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache3"
WEIGHTS = (0.0, 0.05, 0.1, 0.3)
RRF_K = 60
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
K = 3
FETCH = 50
DECAY = 0.5
GRAPH_BUDGET = max(4 * K, 256)
TOP_SEEDS = 20


def _fetch(url: str, name: str) -> bytes:
    path = CACHE / "beir" / name
    if path.exists():
        return path.read_bytes()
    data = urllib.request.urlopen(url, timeout=300).read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def load_beir(name: str):
    base = f"https://huggingface.co/datasets/BeIR/{name}/resolve/main"
    qbase = f"https://huggingface.co/datasets/BeIR/{name}-qrels/resolve/main"
    corpus = pd.read_parquet(io.BytesIO(_fetch(
        f"{base}/corpus/corpus-00000-of-00001.parquet", f"{name}-corpus.parquet")))
    queries = pd.read_parquet(io.BytesIO(_fetch(
        f"{base}/queries/queries-00000-of-00001.parquet", f"{name}-queries.parquet")))
    qrels = pd.read_csv(io.BytesIO(_fetch(
        f"{qbase}/test.tsv", f"{name}-qrels-test.tsv")), sep="\t")
    doc_ids = set(corpus["_id"].astype(str))
    docs = [(str(r["_id"]), f"{r.get('title', '')}. {r.get('text', '')}".strip())
            for _, r in corpus.iterrows()]
    kept = qrels[qrels["corpus-id"].astype(str).isin(doc_ids)]
    qids = set(kept["query-id"].astype(str))
    qtext = {str(r["_id"]): r["text"] for _, r in queries.iterrows()
             if str(r["_id"]) in qids}
    items = []
    for qid, text in qtext.items():
        mask = kept["query-id"].astype(str) == qid
        rel = {str(d): int(s) for d, s in
               zip(kept.loc[mask, "corpus-id"], kept.loc[mask, "score"])}
        if rel:
            items.append((qid, text, rel))
    return docs, items


def load_vectors(ds: str, docs, queries, kind: str):
    """Load cached minilm vectors or deterministically build/cache hashing vectors."""
    if kind == "minilm":
        key = hashlib.sha256(json.dumps(
            [ds, "minilm", "all-MiniLM-L6-v2", len(docs), len(queries)]
        ).encode()).hexdigest()[:16]
        path = CACHE / f"emb-{ds}-minilm-{key}.npz"
        if not path.exists():
            raise FileNotFoundError(f"cached miniLM vectors not found: {path}")
        z = np.load(path)
        return z["vecs"].astype(np.float32), z["qvecs"].astype(np.float32)
    if kind != "hashing":
        raise ValueError(kind)
    key = hashlib.sha256(json.dumps(
        [ds, "hashing-v1", len(docs), len(queries)]
    ).encode()).hexdigest()[:16]
    path = CACHE / f"graph-ablation-{ds}-hashing-{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["vecs"].astype(np.float32), z["qvecs"].astype(np.float32)
    from omnidb_server.embeddings import HashingEmbedder
    emb = HashingEmbedder()
    vecs = emb.encode([text for _id, text in docs]).astype(np.float32)
    qvecs = emb.encode([text for _id, text, _rel in queries]).astype(np.float32)
    np.savez_compressed(path, vecs=vecs, qvecs=qvecs)
    return vecs, qvecs


def _top_dense(qvec: np.ndarray, vecs: np.ndarray, n: int) -> list[int]:
    n = min(n, len(vecs))
    scores = vecs @ qvec
    ix = np.argpartition(-scores, n - 1)[:n]
    return sorted((int(i) for i in ix), key=lambda i: (-float(scores[i]), i))


def _sparse_ranker(docs):
    import bm25s
    tok = bm25s.tokenize([text for _id, text in docs], stopwords="en")
    ranker = bm25s.BM25(method="lucene", k1=0.9, b=0.4)
    ranker.index(tok)
    return ranker


def baseline_rankings(ds, docs, queries, vecs, qvecs, embedding):
    """Return dense, sparse, fused top-FETCH lists; cached on disk."""
    key = hashlib.sha256(json.dumps(
        [ds, embedding, len(docs), len(queries), FETCH, RRF_K]
    ).encode()).hexdigest()[:16]
    path = CACHE / f"graph-ablation-baseline-{ds}-{embedding}-{key}.npz"
    if path.exists():
        z = np.load(path)
        return z["dense"], z["sparse"], z["fused"]
    sparse = _sparse_ranker(docs)
    qtok = __import__("bm25s").tokenize([q for _id, q, _r in queries], stopwords="en")
    srows, _ = sparse.retrieve(qtok, k=FETCH)
    dense_rows, fused_rows = [], []
    for qi, (_qid, _q, _rel) in enumerate(queries):
        d = _top_dense(qvecs[qi], vecs, FETCH)
        s = [int(x) for x in srows[qi]]
        fused = rrf([d, s], k=RRF_K, weights=[DENSE_WEIGHT, SPARSE_WEIGHT])
        f = sorted(fused, key=lambda i: (-fused[i], i))[:FETCH]
        dense_rows.append(d); fused_rows.append(f)
    dense_arr = np.asarray(dense_rows, dtype=np.int32)
    sparse_arr = np.asarray(srows, dtype=np.int32)
    fused_arr = np.asarray(fused_rows, dtype=np.int32)
    np.savez_compressed(path, dense=dense_arr, sparse=sparse_arr, fused=fused_arr)
    return dense_arr, sparse_arr, fused_arr


def _add_edge(adj: dict[int, set[int]], a: int, b: int):
    if a != b:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)


def qrels_graph(docs, train_queries, seed=42):
    pos = {doc_id: i for i, (doc_id, _t) in enumerate(docs)}
    rng = np.random.default_rng(seed)
    adj: dict[int, set[int]] = {}
    for _qid, _text, rel in train_queries:
        ids = [pos[d] for d, s in rel.items() if s > 0 and d in pos]
        degree = Counter()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if degree[a] >= 5 or degree[b] >= 5:
                    continue
                if rng.random() < 0.5:
                    _add_edge(adj, a, b)
                    degree[a] += 1; degree[b] += 1
    return adj


def token_graph(docs, max_df=500, max_edges=10, threshold=0.3):
    """Exact Jaccard graph via an inverted index, NOT O(n^2)."""
    token_re = re.compile(r"\w+")
    sets = [set(token_re.findall(text.lower())) for _id, text in docs]
    df = Counter(t for s in sets for t in s)
    # Optimized inverted index: posting lists for infrequent tokens only
    posting: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for t in s:
            if 1 < df[t] <= max_df:
                posting[t].append(i)
    adj: dict[int, set[int]] = {}
    for i, s in enumerate(sets):
        candidates: set[int] = set()
        for t in s:
            if t in posting:
                candidates.update(posting[t])
        scored = []
        for j in candidates:
            if j == i: continue
            u = s | sets[j]
            score = len(s & sets[j]) / len(u) if u else 0.0
            if score >= threshold:
                scored.append((score, j))
        for _score, j in sorted(scored, key=lambda x: (-x[0], x[1]))[:max_edges]:
            _add_edge(adj, i, j)
    return adj


def semantic_graph(vecs, top_k=10):
    """Independent doc-vector graph using hnswlib; no qrels involved."""
    import hnswlib
    n, dim = vecs.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=100, M=16, random_seed=42)
    index.add_items(vecs, np.arange(n))
    index.set_ef(max(50, top_k + 5))
    labels, _dist = index.knn_query(vecs, k=min(n, top_k + 1), num_threads=1)
    adj: dict[int, set[int]] = {}
    for i, row in enumerate(labels):
        for j in row:
            if int(j) != i:
                _add_edge(adj, i, int(j))
    return adj


def ndcg(ranked, rel, k=K):
    gains = [2 ** rel.get(doc, 0) - 1 for doc in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** v - 1) / np.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def run_corpus(ds, mode, embedding, seed=42):
    docs, queries = load_beir(ds)
    vecs, qvecs = load_vectors(ds, docs, queries, embedding)
    dense, sparse, fused = baseline_rankings(ds, docs, queries, vecs, qvecs, embedding)
    if mode == "train-test":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(queries))
        cut = int(0.8 * len(queries))
        train_i = set(order[:cut].tolist())
        train = [queries[i] for i in range(len(queries)) if i in train_i]
        eval_i = [i for i in range(len(queries)) if i not in train_i]
        graph = qrels_graph(docs, train, seed)
    elif mode == "token":
        eval_i = list(range(len(queries)))
        graph = token_graph(docs)
    elif mode == "semantic":
        eval_i = list(range(len(queries)))
        graph = semantic_graph(vecs)
    else:
        raise ValueError(mode)
    doc_ids = [d for d, _ in docs]
    cells = []
    for weight in WEIGHTS:
        added = relevant = displaced = changed = 0
        off_ndcg = []; on_ndcg = []
        for qi in eval_i:
            _qid, _text, rel = queries[qi]
            off = list(fused[qi, :K])
            off_set = set(off)
            if weight == 0:
                on = off
            else:
                seed_order = list(fused[qi, :TOP_SEEDS])
                seed_scores = {int(i): 1.0 for i in seed_order}
                masses = graph_seed_masses(seed_scores, seed_order, rrf_k=RRF_K)
                gids = graph_expand(masses, graph, decay=DECAY, budget=GRAPH_BUDGET)
                fs = rrf([list(dense[qi]), list(sparse[qi]), gids],
                         k=RRF_K,
                         weights=[DENSE_WEIGHT, SPARSE_WEIGHT, weight])
                on = sorted(fs, key=lambda i: (-fs[i], i))[:K]
            off_ndcg.append(ndcg([doc_ids[i] for i in off], rel))
            on_ndcg.append(ndcg([doc_ids[i] for i in on], rel))
            new = set(on) - off_set
            if new:
                changed += 1
                added += len(new)
                relevant += sum(1 for i in new if rel.get(doc_ids[i], 0) > 0)
                displaced += sum(1 for i in off_set - set(on)
                                 if rel.get(doc_ids[i], 0) > 0)
        cells.append({
            "mode": mode, "corpus": ds, "embedding": embedding,
            "weight": weight, "n_edges": sum(len(v) for v in graph.values()) // 2,
            "n_queries": len(eval_i), "n_queries_changed": changed,
            "added_total": added, "added_relevant": relevant,
            "displaced_relevant": displaced,
            "precision_of_additions": round(relevant / added, 3) if added else None,
            "mean_ndcg3_off": round(float(np.mean(off_ndcg)), 6),
            "mean_ndcg3_on": round(float(np.mean(on_ndcg)), 6),
            "mean_ndcg3_delta": round(float(np.mean(on_ndcg) - np.mean(off_ndcg)), 6),
        })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train-test", "token", "semantic"], required=True)
    ap.add_argument("--corpus", default="nfcorpus,scifact,fiqa,scidocs")
    ap.add_argument("--embedding", choices=["hashing", "minilm"], default=None)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    embedding = args.embedding or ("hashing" if args.mode == "train-test" else "minilm")
    cells = []
    for ds in args.corpus.split(","):
        print(f"=== {ds} / {args.mode} / {embedding} ===", flush=True)
        cells.extend(run_corpus(ds, args.mode, embedding))
        print(f"  complete ({len(cells)} cells)", flush=True)
    out = {"cells": cells, "metadata": {"mode": args.mode, "embedding": embedding,
                                        "weights": WEIGHTS, "k": K,
                                        "fetch": FETCH, "decay": DECAY}}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.json)


if __name__ == "__main__":
    main()