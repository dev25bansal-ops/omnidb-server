"""Research benchmark v2: datasets x embedders x engines with real statistics.

Upgrades over v1 (bench/research_bench.py), per review:
  1. DATASETS: BEIR nfcorpus, scidocs, fiqa, scifact, webis-touche2020
     (real IR benchmarks, human qrels, HF parquet layout).
  2. EMBEDDERS: all-MiniLM-L6-v2 (384d), BAAI/bge-small-en-v1.5 (384d),
     intfloat/e5-small-v2 (384d), BAAI/bge-base-en-v1.5 (768d) — with the
     bge/e5 query/passage prefix convention applied correctly.
  3. STATISTICS: engines are deterministic, so cross-run CIs are degenerate.
     Instead we compute:
       - 95% bootstrap CIs over the QUERY POPULATION (10k resamples) for
         every engine/dataset/embedder cell;
       - paired Wilcoxon signed-rank tests (per-query nDCG deltas) of
         omnidb-corpus vs each vector-only engine, with Holm-Bonferroni
         correction across the engine family;
       - rank-biserial effect size per comparison.
  4. PRODUCT PATH: OmniDB runs in corpus_mode (config flag shipped with this
     benchmark: memory rerank off, weighted-RRF fusion dense_weight=0.7).

Engines: OmniDB (hybrid), sqlite-vec, LanceDB, Qdrant (all vector-only).

Output: bench/research-v2-results.md + .json. Run time ~1-2 h on a laptop
(embedding dominates); every intermediate is cached in JSON so re-renders
of the report do not re-run engines.
"""

from __future__ import annotations

import io
import json
import math
import statistics
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

K = 10
N_DOCS_CAP = 2000
BOOTSTRAP_N = 10_000
RNG_SEED = 1234
DENSE_W = 0.7

DATASETS = {  # canonical BEIR name -> HF repo name
    "nfcorpus": "nfcorpus",
    "scidocs": "scidocs",
    "fiqa": "fiqa",
    "scifact": "scifact",
    "touche-2020": "webis-touche2020",
}

EMBEDDERS = {
    "minilm": {"model": "all-MiniLM-L6-v2", "q_prefix": "", "d_prefix": ""},
    "bge-small": {"model": "BAAI/bge-small-en-v1.5",
                  "q_prefix": "Represent this sentence for searching relevant passages: ",
                  "d_prefix": ""},
    "e5-small": {"model": "intfloat/e5-small-v2",
                 "q_prefix": "query: ", "d_prefix": "passage: "},
    "bge-base": {"model": "BAAI/bge-base-en-v1.5",
                 "q_prefix": "Represent this sentence for searching relevant passages: ",
                 "d_prefix": ""},
}

OUT_MD = Path(__file__).with_name("research-v2-results.md")
OUT_JSON = Path(__file__).with_name("research-v2-results.json")


# ------------------------------------------------------------------ data ---

def load_beir(hf_name: str):
    import pandas as pd

    base = f"https://huggingface.co/datasets/BeIR/{hf_name}/resolve/main"
    qrels_base = f"https://huggingface.co/datasets/BeIR/{hf_name}-qrels/resolve/main"
    corpus_df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(
        f"{base}/corpus/corpus-00000-of-00001.parquet").read())).head(N_DOCS_CAP)
    queries_df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(
        f"{base}/queries/queries-00000-of-00001.parquet").read()))
    qrels_df = pd.read_csv(io.BytesIO(urllib.request.urlopen(
        f"{qrels_base}/test.tsv").read()), sep="\t")

    doc_ids = set(corpus_df["_id"].astype(str))
    docs = [(str(r["_id"]),
             f"{r.get('title', '')}. {r.get('text', '')}".strip())
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


def embed(texts: list[str], spec: dict, is_query: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    prefix = spec["q_prefix"] if is_query else spec["d_prefix"]
    model = SentenceTransformer(spec["model"])
    inputs = [prefix + t for t in texts] if prefix else list(texts)
    # 128 OOMs the 8GB GPU on long-document corpora (touche) where every
    # sequence hits the 512-token cap; 32 fits and short corpora are unaffected
    vecs = model.encode(inputs, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------- metrics --

from _stats import (bootstrap_ci, holm, ndcg_at_k,
                   paired_wilcoxon, recall_at_k)
# ---------------------------------------------------------------- engines --

class _FixedVecEmbedder:
    def __init__(self, vecs, qvecs, corpus_texts):
        self._by_text = {t: vecs[i] for i, t in enumerate(corpus_texts)}
        self._qvecs = qvecs
        self._i = 0
        self.name = "fixed"
        self.dim = int(vecs.shape[1])

    def encode(self, texts: list[str]):
        out = []
        for t in texts:
            if t in self._by_text:
                out.append(self._by_text[t])
            else:
                out.append(self._qvecs[self._i % len(self._qvecs)])
                self._i += 1
        return np.stack(out)


def bench_omnidb(docs, queries, vecs, qvecs, home: Path) -> dict:
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config.from_env(home)
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = DENSE_W, round(1.0 - DENSE_W, 2)
    store = build_stack(cfg)
    store.remember_many([{"content": t} for _d, t in docs])
    shared = _FixedVecEmbedder(vecs, qvecs, [t for _d, t in docs])
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    store.index = BruteForceIndex(home / "vectors.npz", shared.dim)
    store.index.add(ids, vecs)
    store.embedder = shared

    per_query = []
    per_query_recall = []
    for (_qid, qtext, rel) in queries:
        hits = store.recall(qtext, k=K)
        ranked = [docs[h["id"] - 1][0] for h in hits]
        per_query.append(ndcg_at_k(ranked, rel, K))
        per_query_recall.append(recall_at_k(ranked, {d for d, s in rel.items() if s > 0}, K))
    store.close()
    return {"per_query": per_query, "per_query_recall": per_query_recall}


def _vec_engine_per_query(engine_fn, docs, queries, vecs, qvecs, home: Path) -> dict:
    per_query, per_query_recall = engine_fn(docs, queries, vecs, qvecs, home)
    return {"per_query": per_query, "per_query_recall": per_query_recall}


def bench_sqlite_vec(docs, queries, vecs, qvecs, home: Path):
    import sqlite3

    import sqlite_vec

    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "vec.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    dim = vecs.shape[1]
    conn.execute("CREATE VIRTUAL TABLE v USING vec0(id INTEGER PRIMARY KEY, "
                 f"embedding float[{dim}])")
    for i, v in enumerate(vecs):
        conn.execute("INSERT INTO v(id, embedding) VALUES (?, ?)",
                     (i, np.asarray(v, dtype=np.float32).tobytes()))
    conn.commit()
    per_q, per_qr = [], []
    for (_qid, _qtext, rel), qv in zip(queries, qvecs):
        rows = conn.execute(
            "SELECT id, distance FROM v WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (np.asarray(qv, dtype=np.float32).tobytes(), K),
        ).fetchall()
        ranked = [docs[int(r[0])][0] for r in rows]
        per_q.append(ndcg_at_k(ranked, rel, K))
        per_qr.append(recall_at_k(ranked, {d for d, s in rel.items() if s > 0}, K))
    conn.close()
    return per_q, per_qr


def bench_lancedb(docs, queries, vecs, qvecs, home: Path):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(home / "lance"))
    tbl = db.create_table("mem", data=pa.table({
        "id": pa.array(list(range(len(docs))), type=pa.int64()),
        "vector": pa.array([v.tolist() for v in vecs],
                           type=pa.list_(pa.float32())),
        "doc_id": pa.array([d[0] for d in docs], type=pa.string()),
    }), mode="overwrite")
    per_q, per_qr = [], []
    for (_qid, _qtext, rel), qv in zip(queries, qvecs):
        res = tbl.search(np.asarray(qv, dtype=np.float32)).limit(K).to_list()
        ranked = [r["doc_id"] for r in res]
        per_q.append(ndcg_at_k(ranked, rel, K))
        per_qr.append(recall_at_k(ranked, {d for d, s in rel.items() if s > 0}, K))
    return per_q, per_qr


def bench_qdrant(docs, queries, vecs, qvecs, home: Path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(home / "qd"))
    client.create_collection("mem", vectors_config=VectorParams(
        size=vecs.shape[1], distance=Distance.COSINE))
    client.upsert("mem", points=[PointStruct(
        id=i, vector=[float(x) for x in vecs[i]],
        payload={"doc_id": docs[i][0]}) for i in range(len(docs))])
    per_q, per_qr = [], []
    for (_qid, _qtext, rel), qv in zip(queries, qvecs):
        res = client.query_points("mem", query=[float(x) for x in qv], limit=K)
        ranked = [p.payload["doc_id"] for p in res.points]
        per_q.append(ndcg_at_k(ranked, rel, K))
        per_qr.append(recall_at_k(ranked, {d for d, s in rel.items() if s > 0}, K))
    client.close()
    return per_q, per_qr


# ------------------------------------------------------------------ main --

def main() -> None:
    import os as _os

    # reuse the model cache built by the v3 run instead of a fresh tempdir:
    # a fresh dir would force a from-scratch re-download of all four models
    _cache_root = Path(__file__).resolve().parent / ".cache3" / "hf"
    _cache_root.mkdir(parents=True, exist_ok=True)
    _os.environ["HF_HUB_DISABLE_XET"] = "1"
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
        _os.environ[var] = str(_cache_root)

    results: dict = {}
    for ds_name, hf_name in DATASETS.items():
        print(f"=== {ds_name}: load ===", flush=True)
        docs, queries = load_beir(hf_name)
        texts = [t for _d, t in docs]
        results[ds_name] = {}
        for emb_name, spec in EMBEDDERS.items():
            print(f"  embed {emb_name} ({spec['model']}) ...", flush=True)
            vecs = embed(texts, spec, is_query=False)
            qvecs = embed([q for _q, q, _r in queries], spec, is_query=True)
            cell: dict = {}
            tmp = Path(tempfile.mkdtemp(prefix=f"rb2-{ds_name}-{emb_name}-"))
            cell["omnidb"] = bench_omnidb(docs, queries, vecs, qvecs,
                                          tmp / "omnidb")
            cell["sqlite-vec"] = _vec_engine_per_query(
                bench_sqlite_vec, docs, queries, vecs, qvecs, tmp / "sv")
            cell["lancedb"] = _vec_engine_per_query(
                bench_lancedb, docs, queries, vecs, qvecs, tmp / "lance")
            cell["qdrant"] = _vec_engine_per_query(
                bench_qdrant, docs, queries, vecs, qvecs, tmp / "qd")
            results[ds_name][emb_name] = cell
            print(f"    omnidb={statistics.mean(cell['omnidb']['per_query']):.4f} "
                  f"sv={statistics.mean(cell['sqlite-vec']['per_query']):.4f} "
                  f"lance={statistics.mean(cell['lancedb']['per_query']):.4f} "
                  f"qd={statistics.mean(cell['qdrant']['per_query']):.4f}",
                  flush=True)
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    render(results)


def render(results: dict) -> None:
    lines = [
        "# Research benchmark v2: hybrid vs vector-only across datasets x embedders",
        "",
        f"- datasets: BEIR {', '.join(DATASETS)} (human qrels, cap {N_DOCS_CAP} docs, k={K})",
        "- embedders: all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, intfloat/e5-small-v2,",
        "  BAAI/bge-base-en-v1.5 (prefix conventions applied); identical vectors",
        "  served to every engine (corpus AND queries)",
        f"- statistics: 95% bootstrap CIs over the query population "
        f"({BOOTSTRAP_N} resamples); paired Wilcoxon signed-rank of omnidb vs each",
        "  vector-only engine per cell, Holm-Bonferroni corrected across the",
        "  3-engine family; rank-biserial effect size",
        "- omnidb runs corpus_mode (memory rerank off) + weighted-RRF (0.7/0.3)",
        "",
    ]
    for ds_name, emb_cells in results.items():
        lines.append(f"## {ds_name}")
        for emb_name, cell in emb_cells.items():
            lines += [f"### embedder: {emb_name}", "",
                      "| engine | nDCG@10 mean (95% boot CI) | recall@10 mean | Δ vs omnidb | p (Holm) | effect (rb) |",
                      "|---|---|---|---|---|---|"]
            base = cell["omnidb"]["per_query"]
            base_m, base_lo, base_hi = bootstrap_ci(base)
            engines = ["omnidb", "sqlite-vec", "lancedb", "qdrant"]
            pvals = {}
            for eng in engines[1:]:
                p, _rb = paired_wilcoxon(base, cell[eng]["per_query"])
                pvals[eng] = p
            adj = holm(pvals)
            rbs = {eng: paired_wilcoxon(base, cell[eng]["per_query"])[1]
                   for eng in engines[1:]}
            rq = cell["omnidb"].get("per_query_recall", [])
            rec = statistics.mean(rq) if rq else float("nan")
            lines.append(f"| omnidb | {base_m:.4f} ({base_lo:.4f}–{base_hi:.4f}) "
                         f"| {rec:.4f} | — | — | — |")
            for eng in engines[1:]:
                m, lo, hi = bootstrap_ci(cell[eng]["per_query"])
                rq = cell[eng].get("per_query_recall", [])
                rec = statistics.mean(rq) if rq else float("nan")
                delta = m - base_m
                lines.append(
                    f"| {eng} | {m:.4f} ({lo:.4f}–{hi:.4f}) | {rec:.4f} "
                    f"| {delta:+.4f} | {adj[eng]:.4f} | {rbs[eng]:+.3f} |")
            lines.append("")
    lines += [
        "## Conclusion template (fill from table)",
        "",
        "- If omnidb rows show statistically significant positive Δ with CIs",
        "  disjoint from baselines on a majority of dataset x embedder cells,",
        "  the hybrid-when-configured-for-corpus claim holds broadly.",
        "- Cells where omnidb loses identify the regime boundary (embedder",
        "  strength vs lexical-leg value).",
        "",
        "## Honesty notes",
        "",
        "- Engines are deterministic: cross-run variance is zero. All",
        "  uncertainty here is over the QUERY POPULATION (bootstrap), which is",
        "  the correct variance source for IR metrics on a fixed corpus.",
        "- v1 harness bug (LanceDB row doubling) fixed; invariant checks",
        "  (nDCG <= 1) are asserted on every cell in v2 via the metrics themselves.",
        "- No user study; single machine; doc cap 2000 per dataset.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
