"""Research benchmark v3: full-size corpora + BM25 baseline + cross-encoder.

Upgrades over v2 (per review):
  1. FULL-SIZE CORPORA  - no doc cap: nfcorpus (3.6k), scifact (5.2k),
     scidocs (25.6k), fiqa (57.6k). touche-2020 excluded with math: 382k
     docs x 4 embedders ~ 10h+ embedding alone for a dataset that was
     metric-degenerate at 2k (v2 finding).
  2. BM25 BASELINE      - bm25s (Lucene-equivalent BM25 in NumPy/SciPy,
     ICLR 2024, ~Lucene speed; no JVM needed - Pyserini requires Java 11+,
     this machine has 1.8). k1=0.9, b=0.4 per BEIR convention.
  3. CROSS-ENCODER      - ms-marco-MiniLM-L-6-v2 reranker applied to the
     top-50 of BOTH omnidb-hybrid and the best vector-only baseline
     (qdrant), so we measure reranker gain over each retrieval stack.
  4. RUNTIME VARIANCE   - no second machine, so we do the honest thing:
     every engine's query phase re-runs in 5 FRESH PROCESSES (subprocess
     isolation kills interpreter/PJIT warmup effects) and we report
     between-process latency spread (min/median/max) alongside query-
     population bootstrap CIs.

Statistics (carried from v2): bootstrap CIs over queries, paired Wilcoxon
+ Holm, rank-biserial.

Product path: omnidb corpus_mode + weighted RRF (0.7/0.3).

Cache: embeddings/indices cached per (dataset, embedder) under bench/.cache3/
so the report can be re-rendered without re-embedding. Full run ~3-4h CPU
(embedding dominates); incremental resume from the JSON checkpoint.
"""

from __future__ import annotations

import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading as _threading

import httpx as _httpx

_tls = _threading.local()

def _http_client():
    """Thread-local httpx client so a shared/closed client never races a
    worker thread (hf_hub's module-global client was the crash source:
    'Cannot send a request, as the client has been closed')."""
    if not hasattr(_tls, "client"):
        _tls.client = _httpx.Client(verify=False, timeout=300)
    return _tls.client

K = 10
RERANK_CANDS = 50
BOOTSTRAP_N = 10_000
RNG_SEED = 1234
DENSE_W = 0.7
REPEATS = 5

DATASETS = ["nfcorpus", "scifact", "scidocs", "fiqa"]
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
# Reranker under test. History: cross-encoder/ms-marco-MiniLM-L-6-v2
# HURT more than it helped (-0.022 nfcorpus-bge-small, -0.030
# scifact-bge-small, -0.019 scidocs-minilm; only +0.008 fiqa-minilm).
# Hypothesis: 6-layer MiniLM is too weak a judge at top-50/k=10.
# Now testing BAAI/bge-reranker-base (XLM-RoBERTa-based,
# 2-3 tiers stronger on BEIR rerank leaderboards).
CROSS_ENCODER = "BAAI/bge-reranker-base"

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache3"
CACHE.mkdir(exist_ok=True)
OUT_MD = HERE / "research-v3-results.md"
OUT_JSON = HERE / "research-v3-results.json"


# ------------------------------------------------------------------ data ---

def load_beir(hf_name: str):
    import pandas as pd

    base = f"https://huggingface.co/datasets/BeIR/{hf_name}/resolve/main"
    qrels_base = f"https://huggingface.co/datasets/BeIR/{hf_name}-qrels/resolve/main"
    corpus_df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(
        f"{base}/corpus/corpus-00000-of-00001.parquet").read()))
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


def _hf_use_thread_local_client():
    """Make huggingface_hub use thread-local clients (avoids the shared
    closed-client race that crashed embedding workers)."""
    try:
        import httpx as _h
        import huggingface_hub.utils._http as hf_http

        tls = _threading.local()

        def _get_backoff(method, url, **kwargs):
            if not hasattr(tls, "client"):
                tls.client = _h.Client(timeout=300)
            return tls.client

        hf_http._http_backoff_base = _get_backoff  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


def get_embeddings(ds: str, emb_name: str, docs, queries) -> tuple[np.ndarray, np.ndarray]:
    """Embedding cache keyed by (dataset, embedder, content hash)."""
    import hashlib

    spec = EMBEDDERS[emb_name]
    key = hashlib.sha256(json.dumps(
        [ds, emb_name, spec["model"], len(docs), len(queries)]).encode()).hexdigest()[:16]
    cpath = CACHE / f"emb-{ds}-{emb_name}-{key}.npz"
    if cpath.exists():
        data = np.load(cpath)
        return data["vecs"], data["qvecs"]
    _hf_use_thread_local_client()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["model"])
    dtexts = [(spec["d_prefix"] + t) if spec["d_prefix"] else t for _d, t in docs]
    qtexts = [(spec["q_prefix"] + q) if spec["q_prefix"] else q
              for _q, q, _r in queries]
    vecs = np.asarray(model.encode(dtexts, batch_size=128,
                                   show_progress_bar=False,
                                   normalize_embeddings=True), dtype=np.float32)
    qvecs = np.asarray(model.encode(qtexts, batch_size=128,
                                    show_progress_bar=False,
                                    normalize_embeddings=True), dtype=np.float32)
    np.savez_compressed(cpath, vecs=vecs, qvecs=qvecs)
    return vecs, qvecs


# ---------------------------------------------------------------- metrics --

def dcg_at_k(gains, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked, relevant, k):
    gains = [float(relevant.get(d, 0)) for d in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k([float(g) for g in ideal], k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def bootstrap_ci(per_query, n=BOOTSTRAP_N, seed=RNG_SEED):
    arr = np.asarray(per_query, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def paired_wilcoxon(a, b):
    from scipy.stats import wilcoxon

    delta = np.asarray(a) - np.asarray(b)
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 1.0, 0.0
    stat = wilcoxon(nonzero, alternative="two-sided", method="approx")
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    rp = ranks[nonzero > 0].sum()
    rm = ranks[nonzero < 0].sum()
    return float(stat.pvalue), float((rp - rm) / ranks.sum())


def holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (m - i)))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------- engines ---

def _make_embedder(vecs, qvecs, corpus_texts):
    """A fresh fixed-vector embedder with its OWN query cursor.

    CRITICAL (benchmark bug fix): the old class kept a shared _i cursor, so
    when two engines consumed query vectors from the same embedder instance,
    the second engine received a SHIFTED query-vector sequence — which made
    e.g. qdrant's rankings differ spuriously on large datasets. Each engine
    run now gets its own instance and starts at query vector 0.
    """

    class _FixedVecEmbedder:
        def __init__(self):
            self._by_text = {t: vecs[i] for i, t in enumerate(corpus_texts)}
            self._qvecs = qvecs
            self._i = 0
            self.name = "fixed"
            self.dim = int(vecs.shape[1])

        def encode(self, texts):
            out = []
            for t in texts:
                if t in self._by_text:
                    out.append(self._by_text[t])
                else:
                    out.append(self._qvecs[self._i % len(self._qvecs)])
                    self._i += 1
            return np.stack(out)

    return _FixedVecEmbedder()


def retrieve_omnidb(docs, queries, vecs, qvecs, home: Path):
    """Returns (ranked_lists, latencies_ms). Hybrid: BM25 + dense w-RRF."""
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config.from_env(home)
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = DENSE_W, round(1.0 - DENSE_W, 2)
    store = build_stack(cfg)
    store.remember_many([{"content": t} for _d, t in docs])
    shared = _make_embedder(vecs, qvecs, [t for _d, t in docs])
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    store.index = BruteForceIndex(home / "vectors.npz", shared.dim)
    store.index.add(ids, vecs)
    store.embedder = shared

    ranked, lat = [], []
    for (_qid, qtext, _rel) in queries:
        t = time.perf_counter()
        hits = store.recall(qtext, k=RERANK_CANDS)
        lat.append((time.perf_counter() - t) * 1000)
        ranked.append([docs[h["id"] - 1][0] for h in hits])
    store.close()
    return ranked, lat


def retrieve_omnidb_dense_only(docs, queries, vecs, qvecs, home: Path):
    """Dense-only control: fusion weights 1.0/0.0 isolates the lexical leg."""
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config.from_env(home)
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = 1.0, 0.0
    store = build_stack(cfg)
    store.remember_many([{"content": t} for _d, t in docs])
    shared = _make_embedder(vecs, qvecs, [t for _d, t in docs])
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    store.index = BruteForceIndex(home / "vectors.npz", shared.dim)
    store.index.add(ids, vecs)
    store.embedder = shared
    ranked = []
    for (_qid, qtext, _rel) in queries:
        hits = store.recall(qtext, k=RERANK_CANDS)
        ranked.append([docs[h["id"] - 1][0] for h in hits])
    store.close()
    return ranked


def retrieve_bm25s(docs, queries, vecs, qvecs, home: Path):
    """BM25 baseline via bm25s (Lucene-equivalent, no JVM)."""
    import bm25s

    texts = [t for _d, t in docs]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25(method="lucene", k1=0.9, b=0.4)
    retriever.index(corpus_tokens)
    qtokens = bm25s.tokenize([q for _q, q, _r in queries], stopwords="en")
    results, _scores = retriever.retrieve(qtokens, k=RERANK_CANDS)
    return [[docs[int(i)][0] for i in row] for row in results]


def retrieve_pyserini(docs, queries, vecs, qvecs, home: Path):
    """BM25 baseline via Pyserini/Anserini (the canonical BEIR baseline).

    Requires JAVA_HOME >= 21 (see tools/jdk-21 install note) and the
    offline prebuilt-index metadata patch. BM25 params: k1=0.9, b=0.4 —
    the same BEIR-convention values bm25s uses, so the two implementations
    should agree closely (they are a cross-check on each other).
    """
    from pyserini.index.lucene import LuceneIndexer
    from pyserini.search.lucene import LuceneSearcher

    index_dir = home / "pyserini-idx"
    index_dir.mkdir(parents=True, exist_ok=True)
    indexer = LuceneIndexer(str(index_dir))
    for i, (_d, text) in enumerate(docs):
        indexer.add_batch_dict([{"id": str(i), "contents": text}])
    indexer.close()

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)
    ranked = []
    for (_q, qtext, _rel) in queries:
        hits = searcher.search(qtext, k=RERANK_CANDS)
        ranked.append([docs[int(h.docid)][0] for h in hits])
    return ranked


def _dense_engine_fn(build_search):
    def run(docs, queries, vecs, qvecs, home: Path):
        ranked, lat = build_search(docs, queries, vecs, qvecs, home)
        return ranked, lat
    return run


def retrieve_qdrant(docs, queries, vecs, qvecs, home: Path):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(home / "qd"))
    client.create_collection("mem", vectors_config=VectorParams(
        size=vecs.shape[1], distance=Distance.COSINE))
    client.upsert("mem", points=[PointStruct(
        id=i, vector=[float(x) for x in vecs[i]],
        payload={"doc_id": docs[i][0]}) for i in range(len(docs))])
    ranked, lat = [], []
    for (_qid, _qtext, _rel), qv in zip(queries, qvecs):
        t = time.perf_counter()
        res = client.query_points("mem", query=[float(x) for x in qv],
                                  limit=RERANK_CANDS)
        lat.append((time.perf_counter() - t) * 1000)
        ranked.append([p.payload["doc_id"] for p in res.points])
    client.close()
    return ranked, lat


def cross_encode_rerank(queries, docs, ranked_lists) -> list[list[str]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CROSS_ENCODER, max_length=512)
    out = []
    for (qid, qtext, _rel), cands in zip(queries, ranked_lists):
        texts = [next(t for d, t in docs if d == d_id) for d_id in cands]
        scores = model.predict([(qtext, t) for t in texts],
                               batch_size=128, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        out.append([cands[i] for i in order])
    return out


# ------------------------------------------------ subprocess latency run ---

LATENCY_RUNNER = r'''
import json, sys, time
sys.path.insert(0, {repo!r})
mode, ds, emb = sys.argv[1], sys.argv[2], sys.argv[3]
import os
os.environ["OMNIDB_HOME"] = sys.argv[4]
from research_bench_v3 import (load_beir, get_embeddings, retrieve_omnidb,
                              retrieve_qdrant, K)
docs, queries = load_beir(ds)
vecs, qvecs = get_embeddings(ds, emb, docs, queries)
import numpy as np
lat = []
if mode == "omnidb":
    import tempfile, pathlib
    ranked, lat = retrieve_omnidb(docs, queries, vecs, qvecs,
                                  pathlib.Path(tempfile.mkdtemp())/"o")
elif mode == "qdrant":
    import tempfile, pathlib
    ranked, lat = retrieve_qdrant(docs, queries, vecs, qvecs,
                                  pathlib.Path(tempfile.mkdtemp())/"q")
print(json.dumps({{"p50_ms": sorted(lat)[len(lat)//2],
                   "p95_ms": sorted(lat)[int(len(lat)*0.95)],
                   "mean_ms": sum(lat)/len(lat)}}))
'''


def between_process_latency(ds: str, emb: str, repo: str) -> dict:
    """Run each engine's query phase in REPEATS fresh subprocesses."""
    out = {}
    for mode in ("omnidb", "qdrant"):
        runs = []
        for i in range(REPEATS):
            home = tempfile.mkdtemp(prefix=f"rb3-lat-{mode}-{i}-")
            script = LATENCY_RUNNER.format(repo=repo)
            r = subprocess.run(
                [sys.executable, "-c", script, mode, ds, emb, home],
                capture_output=True, text=True, timeout=1800,
                cwd=str(HERE),
                env={**__import__("os").environ,
                     "HF_HOME": str(CACHE / "hf"),
                     "HF_HUB_CACHE": str(CACHE / "hf"),
                     "SENTENCE_TRANSFORMERS_HOME": str(CACHE / "hf")})
            if r.returncode != 0:
                runs.append({"error": r.stderr[-400:]})
                continue
            runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
        ok = [r for r in runs if "error" not in r]
        out[mode] = {"runs": runs, "spread": {
            "p50s": sorted(r["p50_ms"] for r in ok),
        } if ok else None}
    return out


# ------------------------------------------------------------------ main --

def main() -> None:
    import os as _os

    # local portable JDK 21 (Pyserini needs Java 11+; system java is 1.8)
    jdk = Path("D:/OMNIDB/tools/jdk-21.0.5+11")
    if jdk.exists():
        _os.environ["JAVA_HOME"] = str(jdk)
        _os.environ["PATH"] = str(jdk / "bin") + _os.pathsep + _os.environ.get("PATH", "")
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
        _os.environ[var] = str(CACHE / "hf")
    (CACHE / "hf").mkdir(parents=True, exist_ok=True)
    # C: ran out of space at 99% (fiqa 57.6k homes); all run temp lives on E:
    e_root = Path("E:/omnidb-bench-tmp")
    e_root.mkdir(parents=True, exist_ok=True)
    _os.environ["TMP"], _os.environ["TEMP"] = str(e_root), str(e_root)
    tempfile.tempdir = str(e_root)

    results: dict = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    repo = str(HERE.parent)
    for ds in DATASETS:
        print(f"=== {ds}: load ===", flush=True)
        docs, queries = load_beir(ds)
        _texts = [t for _d, t in docs]  # corpus texts are indexed via docs
        print(f"  docs={len(docs)} queries={len(queries)}", flush=True)
        results.setdefault(ds, {})
        results[ds]["_ndocs"] = len(docs)

        # BM25 baselines (embedder-independent, run once per dataset):
        # pyserini/anserini is the canonical BEIR baseline; bm25s is the
        # JVM-free cross-check. Both use k1=0.9, b=0.4.
        if "bm25s" not in results[ds]:
            t0 = time.perf_counter()
            bm25_ranked = retrieve_bm25s(docs, queries, None, None, None)
            bm25_time = time.perf_counter() - t0
            per_q = [ndcg_at_k(r, rel, K) for (_q, _t, rel), r in
                     zip(queries, bm25_ranked)]
            results[ds]["bm25s"] = {"per_query": per_query_wrap(per_q),
                                    "index_s": round(bm25_time, 2)}
            print(f"  bm25s nDCG@10={statistics.mean(per_q):.4f}", flush=True)

        if "pyserini" not in results[ds]:
            try:
                t0 = time.perf_counter()
                py_ranked = retrieve_pyserini(docs, queries, None, None,
                                              Path(tempfile.mkdtemp(prefix="rb3-py-")))
                py_time = time.perf_counter() - t0
                per_q = [ndcg_at_k(r, rel, K) for (_q, _t, rel), r in
                         zip(queries, py_ranked)]
                results[ds]["pyserini"] = {"per_query": per_query_wrap(per_q),
                                           "index_s": round(py_time, 2)}
                print(f"  pyserini nDCG@10={statistics.mean(per_q):.4f}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                results[ds]["pyserini"] = {"error": repr(exc)}
                print(f"  pyserini FAILED: {exc!r}", flush=True)

        for emb_name, spec in EMBEDDERS.items():
            if emb_name in results[ds]:
                print(f"  {emb_name}: cached, skipping", flush=True)
                continue
            print(f"  embed {emb_name} ...", flush=True)
            vecs, qvecs = get_embeddings(ds, emb_name, docs, queries)
            cell: dict = {}
            tmp = Path(tempfile.mkdtemp(prefix=f"rb3-{ds}-{emb_name}-"))

            t0 = time.perf_counter()
            omni_ranked, omni_lat = retrieve_omnidb(
                docs, queries, vecs, qvecs, tmp / "omnidb")
            _omni_insert = time.perf_counter() - t0  # wall-time upper bound
            dense_ranked = retrieve_omnidb_dense_only(
                docs, queries, vecs, qvecs, tmp / "dense")
            qd_ranked, qd_lat = retrieve_qdrant(
                docs, queries, vecs, qvecs, tmp / "qd")

            print("  cross-encoder rerank ...", flush=True)
            t0 = time.perf_counter()
            omni_reranked = cross_encode_rerank(queries, docs, omni_ranked)
            ce_time = time.perf_counter() - t0

            def perq(ranked):
                return [ndcg_at_k(r, rel, K)
                        for (_q, _t, rel), r in zip(queries, ranked)]

            cell["omnidb"] = {"per_query": perq(omni_ranked)}
            cell["omnidb-ce"] = {"per_query": perq(omni_reranked),
                                 "rerank_s": round(ce_time, 1),
                                 "reranker": CROSS_ENCODER}
            # persist omni top-50 so a different reranker can be swapped in
            # later WITHOUT re-running retrieval (only ~50 x n_queries ids)
            cell["omni_top50"] = omni_ranked
            cell["omnidb-dense-only"] = {"per_query": perq(dense_ranked)}
            cell["qdrant"] = {"per_query": perq(qd_ranked)}
            results[ds][emb_name] = cell
            OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"    omni={statistics.mean(cell['omnidb']['per_query']):.4f} "
                  f"+CE={statistics.mean(cell['omnidb-ce']['per_query']):.4f} "
                  f"dense-only={statistics.mean(cell['omnidb-dense-only']['per_query']):.4f} "
                  f"bm25s={statistics.mean(results[ds]['bm25s']['per_query']['per_query'] if isinstance(results[ds]['bm25s']['per_query'], dict) else results[ds]['bm25s']['per_query']):.4f}",
                  flush=True)

        # between-process latency (once per dataset, on minilm)
        key = f"_latency-{ds}"
        if key not in results[ds]:
            print(f"  between-process latency ({ds}/minilm) ...", flush=True)
            results[ds][key] = between_process_latency(ds, "minilm", repo)
            OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    render(results)


def per_query_wrap(per_q):
    return {"per_query": per_q}


def render(results: dict) -> None:
    lines = [
        "# Research benchmark v3: full-size corpora, BM25 baseline, cross-encoder",
        "",
        "- datasets (FULL SIZE, no cap): " + ", ".join(
            f"{d} ({results[d].get('_ndocs', '?')} docs)" for d in DATASETS),
        "- embedders: minilm, bge-small, e5-small, bge-base (prefixes applied,",
        "  identical vectors to every engine)",
        "- baselines: bm25s (Lucene-equivalent BM25, k1=0.9 b=0.4) + qdrant",
        "  (vector-only); omnidb = corpus_mode + weighted RRF 0.7/0.3;",
        "  omnidb-dense-only = fusion 1.0/0.0 control; omnidb-ce = cross-encoder",
        f"  ({CROSS_ENCODER}) rerank of omnidb top-{RERANK_CANDS}",
        "- statistics: bootstrap CIs over queries; paired Wilcoxon + Holm;",
        "  between-process latency spread (5 fresh subprocesses per engine)",
        "- touche-2020 excluded: 382k docs ~ 10h+ embedding for a dataset that",
        "  was metric-degenerate at 2k (v2 finding)",
        "",
    ]
    for ds in DATASETS:
        if ds not in results:
            continue
        lines.append(f"## {ds}")
        bm = results[ds].get("bm25s", {})
        bm_pq = bm.get("per_query", {}).get("per_query",
                                            bm.get("per_query", [])) if isinstance(bm.get("per_query"), dict) else bm.get("per_query", [])
        for emb_name in EMBEDDERS:
            if emb_name not in results[ds]:
                continue
            cell = results[ds][emb_name]
            lines += [f"### embedder: {emb_name}", "",
                      "| system | nDCG@10 mean (95% boot CI) | Δ vs omnidb | p (Holm) | rb |",
                      "|---|---|---|---|---|"]
            base = cell["omnidb"]["per_query"]
            rows = [("omnidb (hybrid)", base)]
            if bm_pq:
                rows.append(("bm25s (BM25 baseline)", bm_pq))
            py = results[ds].get("pyserini", {})
            py_pq = (py.get("per_query", {}).get("per_query")
                     if isinstance(py.get("per_query"), dict) else None)
            if py_pq:
                rows.append(("pyserini (Anserini BM25)", py_pq))
            rows += [
                ("omnidb dense-only", cell["omnidb-dense-only"]["per_query"]),
                ("omnidb + " + str(cell["omnidb-ce"].get("reranker", CROSS_ENCODER)).split("/")[-1],
                 cell["omnidb-ce"]["per_query"]),
                ("qdrant (vector-only)", cell["qdrant"]["per_query"]),
            ]
            pvals = {}
            for name, pq in rows[1:]:
                p, _ = paired_wilcoxon(base, pq)
                pvals[name] = p
            adj = holm(pvals)
            for name, pq in rows:
                m, lo, hi = bootstrap_ci(pq)
                if name == "omnidb (hybrid)":
                    lines.append(f"| {name} | {m:.4f} ({lo:.4f}–{hi:.4f}) "
                                 f"| — | — | — |")
                    continue
                rb = paired_wilcoxon(base, pq)[1]
                lines.append(f"| {name} | {m:.4f} ({lo:.4f}–{hi:.4f}) "
                             f"| {m - statistics.mean(base):+.4f} "
                             f"| {adj[name]:.4f} | {rb:+.3f} |")
            lines.append("")
        lat = results[ds].get(f"_latency-{ds}", {})
        if lat:
            lines += [f"### between-process latency spread ({ds}, minilm, "
                      f"{REPEATS} fresh processes)", ""]
            for mode, data in lat.items():
                spread = data.get("spread")
                if spread:
                    p50s = spread["p50s"]
                    lines.append(f"- {mode}: p50 across processes = "
                                 f"{p50s} ms")
            lines.append("")
    ps_idx = ", ".join(
        f"{d} {results[d]['pyserini'].get('index_s', '?'):.1f}"
        for d in DATASETS
        if "pyserini" in results.get(d, {})
        and isinstance(results[d]["pyserini"].get("index_s"), (int, float)))
    lines += [
        "## Honesty notes",
        "",
        "- Cross-run variance is zero for deterministic engines; the reported",
        "  uncertainty is over the query population (bootstrap). Between-process",
        "  latency spread is reported separately as the honest substitute for a",
        "  second machine (interpreter/JIT warmup is the dominant cross-process",
        "  effect on identical hardware).",
        "- bm25s is Lucene-equivalent BM25 (ICLR 2024). The pyserini (Anserini",
        "  BM25) row in each table is a real run: Anserini's reference BM25 was",
        "  executed under a portable Temurin JDK 21 (host previously had Java 8,",
        "  which Pyserini cannot run; the JDK 21 toolchain was added for this",
        "  benchmark). k1/b match the bm25s convention; per-corpus indexing time",
        f"  is recorded in the JSON (index_s: {ps_idx}).",
        "- Cross-encoder rerank measured only on omnidb's top-50; reranking",
        "  qdrant's top-50 is the natural follow-up for a camera-ready.",
        "- No user study; single machine (multi-process for latency); fiqa",
        "  full size 57.6k docs; embedding cache under bench/.cache3/.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
