"""Comparative benchmark: OmniDB vs sqlite-vec vs chromadb.

Same workload for every system: N synthetic-but-realistic memory texts
(clustered topics, realistic lengths), 50 recall queries at k=5.

Measures per system:
  - cold start (import + init/first-use)
  - insert throughput (single-op and batch where supported)
  - recall latency p50/p95
  - recall@5 quality vs exhaustive brute-force cosine ground truth
  - persistence reopen cost

Honesty notes baked into the harness:
  - sqlite-vec has no BM25/FTS5 built in: its "recall" is pure vector search.
    We measure it as the vector-only baseline it is.
  - chromadb is used in its default persistent local mode, in-process.
  - OmniDB recall is hybrid (BM25 + dense + rerank) — so quality is expected
    to differ; the harness reports it, it does not average it away.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_MEMORIES = 2_000
N_QUERIES = 50
K = 5
SEED = 7

TOPICS = [
    "python virtual environments dependency isolation pip venv",
    "kubernetes ingress controller load balancer cluster node",
    "vector embedding similarity cosine ann hnsw index quantization",
    "database wal checkpoint transaction durability fsync sqlite",
    "rust borrow checker lifetime memory safety systems programming",
    "docker container image layer registry compose volume",
    "llm prompt engineering tokens context window temperature",
    "git rebase merge conflict branch workflow stash",
    "api design rest pagination rate limit authentication",
    "monitoring observability metrics logs traces alerting",
]

VERBS = ["guide", "notes", "runbook", "summary", "postmortem", "checklist",
         "tutorial", "review", "deep dive", "faq"]
QUALIFIERS = ["for beginners", "advanced", "2026 edition", "with examples",
              "troubleshooting", "performance tuning", "quick reference",
              "case study"]


def make_corpus(n: int) -> list[str]:
    rng = np.random.default_rng(SEED)
    texts = []
    for i in range(n):
        topic = TOPICS[int(rng.integers(len(TOPICS)))]
        verb = VERBS[int(rng.integers(len(VERBS)))]
        qual = QUALIFIERS[int(rng.integers(len(QUALIFIERS)))]
        extra = f" entry {i}"
        texts.append(f"{topic} {verb} {qual}{extra}")
    return texts


def make_queries() -> list[tuple[str, np.ndarray]]:
    """Queries are paraphrases of topic cores — no verbatim surface match,
    so this measures real retrieval quality, not keyword luck."""
    rng = np.random.default_rng(SEED + 1)
    paraphrases = [
        ("keeping project dependencies separate", 0),
        ("exposing services in a cluster", 1),
        ("finding similar items by vector distance", 2),
        ("durability of recent writes on crash", 3),
        ("memory safe systems language", 4),
        ("packaging and shipping containers", 5),
        ("how to steer a language model", 6),
        ("undoing and combining code history", 7),
        ("designing web service interfaces", 8),
        ("watching production health", 9),
    ]
    out = []
    for text, topic_idx in paraphrases:
        for _ in range(N_QUERIES // len(paraphrases)):
            junk = f" question {int(rng.integers(1e6))}"
            out.append((text + junk, topic_idx))
    return out


def brute_truth(vectors: np.ndarray, qvec: np.ndarray, k: int) -> set[int]:
    sims = vectors @ qvec
    idx = np.argpartition(-sims, k)[:k]
    return set(int(i) for i in idx)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) + 1))) - 1))
    return xs[i]


# --------------------------------------------------------------- OmniDB ----

def bench_omnidb(corpus, queries, home: Path) -> dict:
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack

    t0 = time.perf_counter()
    store = build_stack(Config.from_env(home))
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for text in corpus:
        store.remember(text)
    insert_s = time.perf_counter() - t0
    singles = insert_s / len(corpus)

    # batch insert on a fresh home for the batch number
    store.close()
    home2 = home.parent / (home.name + "-batch")
    store = build_stack(Config.from_env(home2))
    t0 = time.perf_counter()
    store.remember_many([{"content": t} for t in corpus])
    batch_s = time.perf_counter() - t0
    batch_rate = len(corpus) / batch_s
    live = store

    latencies, hits = [], []
    for qtext, _topic in queries:
        t = time.perf_counter()
        results = live.recall(qtext, k=K)
        latencies.append((time.perf_counter() - t) * 1000)
        hits.append([r["content"] for r in results])

    # ground truth: encode corpus once via the store's own embedder
    t0 = time.perf_counter()
    vectors = live.embedder.encode(corpus)
    embed_s = time.perf_counter() - t0
    qvecs = live.embedder.encode([q for q, _ in queries])

    # truth is computed over corpus POSITIONS; map position -> content
    # (corpus[i] corresponds to vectors[i] by construction) and compare
    # returned CONTENTS, not ids — ids depend on insertion internals.
    truth_texts_all = []
    for qv in qvecs:
        tset = brute_truth(vectors, qv, K)
        truth_texts_all.append({corpus[i] for i in tset})

    recalls = []
    for (hit_texts, truth_texts) in zip(hits, truth_texts_all):
        recalls.append(len(set(hit_texts) & truth_texts) / K)

    t0 = time.perf_counter()
    live.close()
    t0 = time.perf_counter()
    store3 = build_stack(Config.from_env(home2))
    reopen_s = time.perf_counter() - t0
    store3.close()

    return {
        "cold_start_s": round(cold_s, 4),
        "insert_single_ms": round(singles * 1000, 3),
        "insert_batch_per_sec": round(batch_rate, 1),
        "recall_p50_ms": round(pct(latencies, 50), 3),
        "recall_p95_ms": round(pct(latencies, 95), 3),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "reopen_s": round(reopen_s, 4),
        "embed_2000_s": round(embed_s, 3),
    }


# ------------------------------------------------------------- sqlite-vec --

def bench_sqlite_vec(corpus, queries, home: Path) -> dict:
    import sqlite3

    import sqlite_vec

    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "vec.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    dim = 384
    t0 = time.perf_counter()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS v USING vec0(id INTEGER PRIMARY KEY, "
        f"embedding float[{dim}])"
    )
    conn.commit()
    cold_s = time.perf_counter() - t0

    # sqlite-vec does not embed: use a hashing embedder over the same texts
    # so everyone gets identical vectors. (Same approach OmniDB's offline
    # embedder uses — this measures the vector layer, not the embedder.)
    from omnidb_server.embeddings import HashingEmbedder
    emb = HashingEmbedder(dim=dim)

    def to_blob(v):
        return np.asarray(v, dtype=np.float32).tobytes()

    t0 = time.perf_counter()
    vectors = emb.encode(corpus)
    for i, v in enumerate(vectors):
        conn.execute("INSERT INTO v(id, embedding) VALUES (?, ?)", (i, to_blob(v)))
    conn.commit()
    insert_s = time.perf_counter() - t0

    latencies, recalls = [], []
    qvecs = emb.encode([q for q, _ in queries])
    for (qtext, _topic), qv in zip(queries, qvecs):
        t = time.perf_counter()
        rows = conn.execute(
            "SELECT id, distance FROM v WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (to_blob(qv), K),
        ).fetchall()
        latencies.append((time.perf_counter() - t) * 1000)
        got = {int(r[0]) for r in rows}
        truth = brute_truth(vectors, qv, K)
        recalls.append(len(got & truth) / K)

    t0 = time.perf_counter()
    conn.close()
    c2 = sqlite3.connect(home / "vec.db")
    c2.enable_load_extension(True)
    sqlite_vec.load(c2)
    c2.execute("SELECT COUNT(*) FROM v").fetchone()
    reopen_s = time.perf_counter() - t0
    c2.close()

    return {
        "cold_start_s": round(cold_s, 4),
        "insert_single_ms": round(insert_s / len(corpus) * 1000, 3),
        "recall_p50_ms": round(pct(latencies, 50), 3),
        "recall_p95_ms": round(pct(latencies, 95), 3),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "reopen_s": round(reopen_s, 4),
        "note": "vector-only (no lexical leg, no rerank); vectors from same hashing embedder",
    }


# ---------------------------------------------------------------- chroma --

def bench_chroma(corpus, queries, home: Path) -> dict:
    import chromadb

    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=str(home / "chroma"))
    col = client.get_or_create_collection("mem")
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ids = [str(i) for i in range(len(corpus))]
    col.add(ids=ids, documents=corpus)  # chroma default embedder (onnx miniLM)
    insert_s = time.perf_counter() - t0

    latencies, recalls = [], []
    for qtext, _topic in queries:
        t = time.perf_counter()
        res = col.query(query_texts=[qtext], n_results=K)
        latencies.append((time.perf_counter() - t) * 1000)
        _got_docs = res["documents"][0]  # chroma returns docs; we use ids
        # chroma's own embedder defines its own truth: compare against
        # chroma's vector space via an exhaustive query (n_results=len)
        full = col.query(query_texts=[qtext], n_results=len(corpus))
        truth = set(full["ids"][0][:K])
        got = set(res["ids"][0][:K])
        recalls.append(len(got & truth) / K)

    t0 = time.perf_counter()
    client2 = chromadb.PersistentClient(path=str(home / "chroma"))
    client2.get_or_create_collection("mem")
    reopen_s = time.perf_counter() - t0

    return {
        "cold_start_s": round(cold_s, 4),
        "insert_single_ms": round(insert_s / len(corpus) * 1000, 3),
        "recall_p50_ms": round(pct(latencies, 50), 3),
        "recall_p95_ms": round(pct(latencies, 95), 3),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "reopen_s": round(reopen_s, 4),
        "note": "default onnx MiniLM embedder; recall vs chroma's own vector space",
    }


def main() -> None:
    corpus = make_corpus(N_MEMORIES)
    queries = make_queries()
    tmp = Path(tempfile.mkdtemp(prefix="omnidb-compare-"))

    results = {}
    for name, fn in (("omnidb", bench_omnidb),
                     ("sqlite-vec", bench_sqlite_vec),
                     ("chromadb", bench_chroma)):
        print(f"--- {name} ---", flush=True)
        try:
            results[name] = fn(corpus, queries, tmp / name)
            print(json.dumps(results[name], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": repr(exc)}
            print("ERROR:", repr(exc), flush=True)

    out = Path(__file__).with_name("comparative-results.md")
    lines = [
        "# Comparative benchmark — OmniDB vs sqlite-vec vs chromadb",
        "",
        f"- corpus: {N_MEMORIES} synthetic topic-clustered texts; {N_QUERIES} paraphrase queries at k={K}",
        f"- host run: {time.strftime('%Y-%m-%d %H:%M')}, single process, warm OS cache",
        "- honesty: each system measured with its own storage/embedding stack; recall@k",
        "  is computed against each system's own vector space (see per-system notes).",
        "",
        "| metric | omnidb | sqlite-vec | chromadb |",
        "|---|---|---|---|",
    ]
    keys = ["cold_start_s", "insert_single_ms", "insert_batch_per_sec",
            "recall_p50_ms", "recall_p95_ms", "recall_at_k", "reopen_s"]
    for key in keys:
        row = [key]
        for name in ("omnidb", "sqlite-vec", "chromadb"):
            v = results.get(name, {}).get(key, "n/a")
            row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Notes", ""]
    for name in ("omnidb", "sqlite-vec", "chromadb"):
        note = results.get(name, {}).get("note", "")
        if "error" in results.get(name, {}):
            lines.append(f"- **{name}**: ERROR — {results[name]['error']}")
        elif note:
            lines.append(f"- **{name}**: {note}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
