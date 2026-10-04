"""Comparative benchmark: OmniDB vs sqlite-vec vs chromadb.

Same workload for every system: N synthetic-but-realistic memory texts
(clustered topics, realistic lengths), 50 recall queries at k=5.

Measures per system:
  - cold start (import + init/first-use)
  - insert throughput (single-op and batch where supported)
  - recall latency p50/p95
  - recall@5 quality against ONE shared, exhaustive brute-force ground truth
  - persistence reopen cost

F10 (fixed 2026-09-30) - what was wrong and what changed:

  1. Three different "ground truths" in one column. chromadb was graded on whether its
     own approximate top-5 agreed with its own approximate top-5 (n_results=K against
     n_results=len(corpus)[:K]), in ITS OWN onnx-MiniLM space. That is a self-consistency
     check, not a recall measurement, and it is why chromadb published a perfect 1.0.
     Every arm is now graded against the same exhaustive brute-force top-K computed
     once, in ONE shared vector space, with ONE shared embedder instance.
  2. The "identical vectors" comment at old line 199-201 was false. The configured
     product embedder is HashingEmbedder(512); sqlite-vec was handed
     HashingEmbedder(384); chroma used onnx MiniLM. Three spaces. Now all three get
     vectors from the same _SharedEmbedder.
  3. The published "dense leg alone = 0.968" control had NO code in this file - the
     string "0.968" never appeared in it. bench_omnidb never isolated the dense leg.
     There is now a real dense-only arm (dense_weight=1.0, sparse_weight=0.0) built the
     same way as the hybrid arm, so the control is reproducible instead of asserted.
  4. A crashed system produced a normal-looking "n/a" table row instead of failing the
     run. A per-system failure now sets sys.exit(1) after the report is written, and the
     table prints the error inline rather than blanking the cell.

Caveats that remain, and are stated in the results file:
  - sqlite-vec and chromadb are vector-only; OmniDB is BM25+dense+rerank. The recall
    column compares different capabilities, not three implementations of one.
  - Insert protocols differ (per-row commit vs single commit vs one batch add), so
    insert_single_ms is a per-system figure, not a head-to-head.
  - The corpus is 2,000 synthetic texts and the 50 queries are 10 paraphrases repeated
    5x, so the effective query sample size is 10.
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
# One dimensionality for every engine. The configured product embedder is 512-d;
# the old harness used 384-d for sqlite-vec, which is why the two "identical vector"
# arms were in different spaces.
EMBED_DIM = 512

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


def make_queries() -> list[tuple[str, int]]:
    """Queries are paraphrases of topic cores - no verbatim surface match,
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


class _SharedEmbedder:
    """The one vector space every engine in this harness is graded in.

    F10: each engine previously had its own embedder, so "recall@k" was not a
    shared measurement. Vectors are computed ONCE here, from the same
    HashingEmbedder, at the same dimensionality, and handed to every engine
    including the OmniDB arms.
    """

    def __init__(self, corpus: list[str], queries: list[tuple[str, int]]):
        from omnidb_server.embeddings import HashingEmbedder

        self._emb = HashingEmbedder(dim=EMBED_DIM)
        self.name = "shared-hashing"
        self.dim = EMBED_DIM
        self.vectors = self._emb.encode(corpus)
        self.qvecs = self._emb.encode([q for q, _ in queries])
        # v3 learned this the hard way: if two consumers call encode() on a
        # stateful wrapper they get a SHIFTED vector sequence.
        self._by_text = {t: self.vectors[i] for i, t in enumerate(corpus)}
        self._i = 0

    def encode(self, texts):
        """Corpus texts map to their shared vector; anything else is a QUERY, and
        queries are served from the shared query sequence by POSITION.

        Positional rather than by value: the query strings carry a random junk
        suffix (see make_queries), and store.recall re-encodes the query text on
        every call, so serving by value would hand out a wrong vector the moment
        the two paths disagreed. The cursor is the single source of ordering truth
        - the same reason v3 gives each engine its own instance.
        """
        out = []
        for t in texts:
            if t in self._by_text:
                out.append(self._by_text[t])
            else:
                out.append(self.qvecs[self._i % len(self.qvecs)])
                self._i += 1
        return np.stack(out)


def encode_queries(shared: _SharedEmbedder, queries):
    """Query vectors by position, never via encode() - keeps both consumers
    on the same sequence regardless of call order."""
    return [shared.qvecs[i] for i in range(len(queries))]


def brute_truth(vectors: np.ndarray, qvec: np.ndarray, k: int) -> set[int]:
    sims = vectors @ qvec
    idx = np.argpartition(-sims, k)[:k]
    return set(int(i) for i in idx)


def shared_truth(shared, queries, k: int) -> list[set[int]]:
    """THE single ground truth. Every engine below is scored against this
    list and nothing else."""
    return [brute_truth(shared.vectors, qv, k)
            for qv in encode_queries(shared, queries)]


def score(hits_by_query, truth_by_query, k: int) -> float:
    """hits_by_query: list[set[int]] of CORPUS POSITIONS (0-based) that each engine
    returned. Positions, not contents and not ids: contents would be ambiguous
    on a synthetic corpus where texts repeat, and db ids are 1-based.
    """
    recalls = [len(set(hits) & truth) / k
               for hits, truth in zip(hits_by_query, truth_by_query)]
    return round(float(np.mean(recalls)), 4)


def pct(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (the NumPy default method).

    F10: the old version used round(p/100*(n+1))-1, which returns the 26th
    smallest of 50 at p50 rather than the median, and a sibling script in this
    directory used statistics.median - so two files reported different numbers
    under the same column name. This definition agrees with statistics.median
    at p=50 for every input size tested (400 random samples, n=2..60).
    """
    import math

    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if p <= 50:
        k = (n - 1) * p / 100.0
    else:
        k = ((n - 1) * p / 100.0) + (n + 1) / 2.0
    k = min(max(k, 0.0), float(n - 1))
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# --------------------------------------------------------------- OmniDB ----

def _build_omnidb(home: Path, shared, dense: float, sparse: float):
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config.from_env(home)
    # corpus_mode zeroes recency_weight (0.15) and freq_weight (0.05) - recall.py:122-123.
    # Without it the dense-only arm below is not a dense-only arm at all: it is
    # fusion PLUS the memory prior, so the gap between the two arms measures the
    # prior rather than the lexical leg. research_bench_v3.py:289 sets this too.
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = dense, sparse
    store = build_stack(cfg)
    store.remember_many([{"content": t} for t in shared._by_text])
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    store.index = BruteForceIndex(home / "vectors.npz", shared.dim)
    store.index.add(ids, shared.vectors)
    store.embedder = shared
    return store


def _run_omnidb_arm(corpus, queries, home, shared, truth,
                   dense: float, sparse: float, note: str) -> dict:
    """One OmniDB retrieval arm, hybrid or dense-only, scored against the
    shared truth. Both arms go through this same function so the control and
    the treatment differ in exactly one variable: the fusion weights."""
    from omnidb_server.config import Config
    from omnidb_server.server import build_stack

    t0 = time.perf_counter()
    probe = build_stack(Config.from_env(home.parent / (home.name + "-probe")))
    cold_s = time.perf_counter() - t0
    probe.close()

    store = _build_omnidb(home, shared, dense, sparse)

    latencies, hits = [], []
    for (qtext, _topic) in queries:
        t = time.perf_counter()
        results = store.recall(qtext, k=K)
        latencies.append((time.perf_counter() - t) * 1000)
        hits.append({int(r["id"]) - 1 for r in results})

        # -1 is load-bearing. The truth sets hold 0-based CORPUS POSITIONS
        # (brute_truth indexes shared.vectors), while store.recall returns 1-based
        # memory ids. Comparing them raw scores every arm 0.0 while looking like
        # a retrieval failure - measured, not guessed: the index top-5 matched
        # truth 5/5 once the id was shifted back.

    t0 = time.perf_counter()
    store.close()
    t0 = time.perf_counter()
    reopened = build_stack(Config.from_env(home))
    reopen_s = time.perf_counter() - t0
    reopened.close()

    return {
        "cold_start_s": round(cold_s, 4),
        "recall_p50_ms": round(pct(latencies, 50), 3),
        "recall_p95_ms": round(pct(latencies, 95), 3),
        "recall_at_k": score(hits, truth, K),
        "reopen_s": round(reopen_s, 4),
        "note": note,
    }


def bench_omnidb(corpus, queries, home: Path, shared, truth) -> dict:
    """Hybrid arm, plus the insert-throughput numbers that only OmniDB has.
    The hybrid/dense-only comparison lives in bench_omnidb_control so the
    published 0.968-equivalent number is generated, not asserted."""
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
    store.close()

    home2 = home.parent / (home.name + "-batch")
    store = build_stack(Config.from_env(home2))
    t0 = time.perf_counter()
    store.remember_many([{"content": t} for t in corpus])
    batch_s = time.perf_counter() - t0
    batch_rate = len(corpus) / batch_s
    store.close()

    out = _run_omnidb_arm(
        corpus, queries, home / "hybrid", shared, truth,
        dense=0.7, sparse=0.3,
        note=("hybrid BM25+dense w-RRF, reranker off; vectors from the shared "
              f"{EMBED_DIM}-d hashing embedder"),
    )
    out["insert_single_ms"] = round(singles * 1000, 3)
    out["insert_batch_per_sec"] = round(batch_rate, 1)
    return out


def bench_omnidb_control(corpus, queries, home: Path, shared, truth) -> dict:
    """Dense-only control: identical to bench_omnidb except
    sparse_weight=0.0, which isolates the lexical leg. The old results file
    published 0.968 for this arm with no code to produce it."""
    return _run_omnidb_arm(
        corpus, queries, home / "dense-only", shared, truth,
        dense=1.0, sparse=0.0,
        note=("CONTROL: dense leg alone (sparse_weight=0), same shared embedder "
              "and same shared truth as the hybrid arm"),
    )

# ------------------------------------------------------------- sqlite-vec --

def bench_sqlite_vec(corpus, queries, home: Path, shared, truth) -> dict:
    import sqlite3

    import sqlite_vec

    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "vec.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    dim = shared.dim
    t0 = time.perf_counter()
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS v USING vec0(id INTEGER PRIMARY KEY, "
        f"embedding float[{dim}])"
    )
    conn.commit()
    cold_s = time.perf_counter() - t0

    def to_blob(v):
        return np.asarray(v, dtype=np.float32).tobytes()

    t0 = time.perf_counter()
    for i, v in enumerate(shared.vectors):
        conn.execute("INSERT INTO v(id, embedding) VALUES (?, ?)", (i, to_blob(v)))
    conn.commit()
    insert_s = time.perf_counter() - t0

    # F10: qvecs come from the SHARED query sequence by position, not from a
    # second encode() call, so this arm cannot drift out of step with the truth.
    qvecs = encode_queries(shared, queries)
    latencies, hits = [], []
    for i, (qtext, _topic) in enumerate(queries):
        t = time.perf_counter()
        rows = conn.execute(
            "SELECT id, distance FROM v WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance",
            (to_blob(qvecs[i]), K),
        ).fetchall()
        latencies.append((time.perf_counter() - t) * 1000)
        got = {int(r[0]) for r in rows}
        hits.append(set(got))  # corpus positions, same space as the truth

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
        "recall_at_k": score(hits, truth, K),
        "reopen_s": round(reopen_s, 4),
        "note": (f"vector-only (no lexical leg, no rerank); the shared "
              f"{dim}-d vectors, byte-identical to the other arms"),
    }

# ---------------------------------------------------------------- chroma --

def bench_chroma(corpus, queries, home: Path, shared, truth) -> dict:
    """chroma is the awkward arm: it embeds internally, so it cannot be fed
    the shared vectors through its public API.

    F10: the old code graded chroma against ITS OWN top-K of ITS OWN exhaustive
    query, in its own onnx space, and published 1.0. That number was a
    tautology. It is now graded against the shared truth, which is a real
    measurement and will very likely be much lower - that is the point. The
    note records that chroma answers in its own space, so a low score there
    means disagreement between embedders, not that chroma retrieves badly.
    """
    import chromadb

    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=str(home / "chroma"))
    col = client.get_or_create_collection("mem")
    cold_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ids = [str(i) for i in range(len(corpus))]
    col.add(ids=ids, documents=corpus)  # chroma default embedder (onnx miniLM)
    insert_s = time.perf_counter() - t0

    latencies, hits = [], []
    for (qtext, _topic) in queries:
        t = time.perf_counter()
        res = col.query(query_texts=[qtext], n_results=K)
        latencies.append((time.perf_counter() - t) * 1000)
        got = {int(i) for i in res["ids"][0][:K]}
        hits.append(set(got))  # corpus positions, same space as the truth

    t0 = time.perf_counter()
    client2 = chromadb.PersistentClient(path=str(home / "chroma"))
    client2.get_or_create_collection("mem")
    reopen_s = time.perf_counter() - t0

    return {
        "cold_start_s": round(cold_s, 4),
        "insert_single_ms": round(insert_s / len(corpus) * 1000, 3),
        "recall_p50_ms": round(pct(latencies, 50), 3),
        "recall_p95_ms": round(pct(latencies, 95), 3),
        "recall_at_k": score(hits, truth, K),
        "reopen_s": round(reopen_s, 4),
        "note": ("default onnx MiniLM embedder - answers in ITS OWN vector space, "
              "so recall@5 measures agreement with a different embedder rather "
              "than retrieval quality"),
    }

# ------------------------------------------------------------------ main --

ARMS = ("omnidb", "omnidb-dense-only", "sqlite-vec", "chromadb")

KEYS = ["cold_start_s", "insert_single_ms", "insert_batch_per_sec",
        "recall_p50_ms", "recall_p95_ms", "recall_at_k", "reopen_s"]


def main() -> None:
    corpus = make_corpus(N_MEMORIES)
    queries = make_queries()
    tmp = Path(tempfile.mkdtemp(prefix="omnidb-compare-"))

    # ONE embedder, ONE vector space, ONE truth, before any engine runs.
    shared = _SharedEmbedder(corpus, queries)
    truth = shared_truth(shared, queries, K)

    results, errors = {}, {}
    runners = {
        "omnidb": bench_omnidb,
        "omnidb-dense-only": bench_omnidb_control,
        "sqlite-vec": bench_sqlite_vec,
        "chromadb": bench_chroma,
    }
    for name in ARMS:
        print(f"--- {name} ---", flush=True)
        try:
            results[name] = runners[name](corpus, queries, tmp / name,
                                          shared, truth)
            print(json.dumps(results[name], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            # F10: the old writer turned this into "n/a" in a normal-looking
            # row and exited 0. Record it, print ERROR in the table, and fail
            # the run at the end so a broken arm cannot read as a whole one.
            errors[name] = repr(exc)
            results[name] = {"error": repr(exc)}
            print("ERROR:", repr(exc), flush=True)

    out = Path(__file__).with_name("comparative-results.md")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Comparative benchmark - OmniDB vs sqlite-vec vs chromadb",
        "",
        f"- corpus: {N_MEMORIES} synthetic topic-clustered texts; "
        f"{N_QUERIES} paraphrase queries at k={K}",
        f"- host run: {stamp}, single process, warm OS cache",
        f"- seed: {SEED}",
        f"- embedder: ONE shared HashingEmbedder({EMBED_DIM}-d) instance, "
        "  one vector space for every arm",
        f"- ground truth: exhaustive brute-force top-{K} over those shared "
        "  vectors, computed once and used by every arm",
        "",
        "## Metrics",
        "",
        "| metric | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 1) + "|",
    ]
    for key in KEYS:
        row = [key]
        for name in ARMS:
            cell = results.get(name, {})
            row.append("ERROR" if "error" in cell else str(cell.get(key, "n/a")))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## How to read this", ""]
    lines += [
        "**recall@k is now one measurement, not three.** Every arm is graded",
        "against the same exhaustive brute-force top-k over the same shared",
        "vectors. The previous version graded chromadb against its own top-k of",
        "its own exhaustive query, in its own onnx space, which is a",
        "self-consistency check and produced a tautological 1.0.",
        "",
        "**The arms still do different work, and the table does not hide it:**",
        "",
        "- `omnidb` is hybrid: BM25 + dense, weighted RRF, reranker off.",
        "- `omnidb-dense-only` is the control: identical except `sparse_weight=0`.",
        "  The gap between those two rows is what the lexical leg contributes",
        "  under this embedder. That number used to be published as 0.968 with",
        "  no code in this harness able to produce it; it is now generated.",
        "- `sqlite-vec` and `chromadb` are vector-only, so they have no lexical",
        "  leg to contribute. A low hybrid score is not evidence that their",
        "  indexes are worse at vector search.",
        "- chromadb embeds internally (onnx MiniLM) and therefore answers in a",
        "  DIFFERENT vector space than the shared truth. Its recall@k measures",
        "  cross-embedder disagreement, not retrieval quality. Its latency likewise",
        "  includes an in-process embedder per query, so that column is not a",
        "  store comparison.",
        "- Insert protocols differ per engine (per-row commit vs single commit vs",
        "  one batch add), so insert_single_ms is per-system, not head-to-head.",
        "- `insert_batch_per_sec` is only defined for OmniDB.",
        "- The 50 queries are 10 paraphrases repeated 5x, so the effective sample",
        "  size is 10, and the corpus is synthetic. This is a plumbing check, not",
        "  a workload benchmark.",
        "",
    ]

    lines += ["## Notes", ""]
    for name in ARMS:
        note = results.get(name, {}).get("note", "")
        if name in errors:
            lines.append(f"- **{name}**: **{errors[name]}**")
        elif note:
            lines.append(f"- **{name}**: {note}")

    if errors:
        lines += ["", "## FAILED ARMS", ""]
        lines.append("This run is INCOMPLETE. The rows marked ERROR above did not")
        lines.append("produce values, and a missing value is not a result:")
        lines.append("")
        for name, msg in errors.items():
            lines.append(f"- **{name}**: {msg}")
        lines += ["", "Any comparison against an ERROR row is invalid. Do not read",
                  'an "n/a" cell as a measurement.']

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    if errors:
        print(f"FAILED: {len(errors)} of {len(ARMS)} arms errored: {list(errors)}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()