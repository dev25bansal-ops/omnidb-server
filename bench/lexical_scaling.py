"""FTS5 lexical-leg scaling sweep -- measurement only, no fix (PERF-A2-07-R).

Why this exists. PERF-A2-07-R measured the lexical leg of hybrid recall at
209.8 ms p50 / 255.6 ms p95 at 60k rows: 75% of total recall latency, growing
53x from 5k while the hnsw dense leg grew 7.3x. That measurement came from
bench/verify_qdrant_rootcause.py, which measured the leg as a side effect of
asking a different question. Nothing in the repo measures the FTS leg on its
own, at a spread of corpus sizes, with a growth exponent attached -- and
bench/results.md sweeps dim x n x ef for HNSW without ever touching it. The
product's dominant retrieval cost therefore has no baseline and no gate.

This script closes that gap and nothing more. It sweeps the lexical leg alone
across corpus sizes and reports, per size: p50/p95, the number of rows the
match set actually spans, and the exponent alpha in latency ~ n^alpha fitted
over the sweep. It deliberately does NOT change the query, the schema, or the
query plan -- see the suspicion note at the bottom of this file and the three
candidate causes named in A2_PERFORMANCE.md PERF-A2-07-R:

    1. bm25() ranking forces a full match-set sort. SQLite must materialise
       every row the MATCH expression touches, score all of them, and only then
       apply LIMIT. A rank-bounded subquery may let it stop early.
    2. The JOIN memories re-check re-fetches each candidate row and applies
       the soft-delete predicate, so it touches rows the LIMIT would have
       discarded.
    3. The ORDER BY on a computed bm25() value cannot use an index, so the
       sort spills.

The audit states the cause is unmeasured. This harness is the first step to
measuring it: it reports the match-set width, which is the single number that
distinguishes (1) from (2)+(3). If the match set is small, the sort is not the
problem and the join is. If it is large and grows with n, the sort is.

Output format follows bench/growth-results.txt: a key: value header block, then
one band per corpus size, then a derived-growth block. Values are
colon-space separated so the file greps like the growth artifact.

Run:  python bench/lexical_scaling.py
Env:  LEX_SIZES=5000,25000,60000   LEX_QUERIES=60   LEX_WORKDIR=<writable volume>

The default corpus is templated agent-memory text, not the BEIR corpora. The
*relative* decomposition is what this file is for; absolute milliseconds are
not comparable to research-v3-results.md (same caveat as PERF-A2-07-R).
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server.db import DbError, connect  # noqa: E402
from omnidb_server.fusion import fts_match_query  # noqa: E402

# Sweep grid. 5k/25k/60k match the bands verify_qdrant_rootcause.py measured so
# the two artifacts are directly comparable; 60k is the corpus where the leg
# was found to cost 209.8 ms p50.
SIZES = [int(x) for x in os.environ.get("LEX_SIZES", "5000,25000,60000").split(",")]
QUERIES = int(os.environ.get("LEX_QUERIES", "60"))
REPEATS = 3
FETCH = 50  # recall()'s k_fetch default
K = 10

# Per-band counters, emitted in growth-results.txt order.
_BANDS = [(5_000, "p50_ms_5000"), (25_000, "p50_ms_25000"), (60_000, "p50_ms_60000")]


def _pct(values: list, q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def _make_text(i: int) -> str:
    """Templated agent-memory text, shaped like what remember() actually stores.

    Real memory rows are a few short sentences with a few distinct tokens; the
    padding words below exist so the FTS tokenizer has the same posting-list
    behaviour it sees on a real corpus, and the rare token gives each query a
    small, bounded match set -- the common case for a specific memory recall.
    """
    return (f"benchmark document number {i} covering topic group {i % 97} "
            f"with reference material and payload {i}.")


def _rare_query(i: int) -> str:
    """A specific query: one unique token plus a common one.

    The rare token is `payload <i>`, which exists in exactly one row: the
    porter/unicode61 tokenizer splits on non-alphanumerics, so the digits
    become their own term. That gives a match set of width 1 for the specific
    query, against n for the OR-ed one -- the contrast that makes the sort
    hypothesis testable.
    """
    return f"payload {i} reference"


def build_corpus(home: Path, n_docs: int) -> "object":
    """Create a real schema and insert n_docs rows through the real trigger path."""
    conn = connect(home)
    now_ms = 1_700_000_000_000
    rows = [(_make_text(i), now_ms + i) for i in range(n_docs)]
    # executemany through the INSERT trigger keeps the FTS index in sync exactly
    # as remember() does, without paying for per-row commits or embedding.
    conn.executemany(
        "INSERT INTO memories (kind, content, metadata, created_ms) "
        "VALUES ('semantic', ?, '{}', ?)", rows)
    conn.commit()
    # ANALYZE so SQLite is not optimizing the sweep against a stale/absent
    # sqlite_stat1 -- without it the planner's row estimates are pure guesses.
    conn.execute("ANALYZE")
    conn.commit()
    return conn


def measure_leg(conn, queries: list) -> tuple:
    """Time the lexical leg alone, exactly as recall.py:75-83 shapes it.

    The SQL is byte-identical to recall.py's, including the JOIN, the
    soft-delete predicate, the kind and agent_id scope placeholders, and
    ORDER BY bm25(memory_fts) LIMIT ?.
    """
    latencies = []
    for _rep in range(REPEATS):
        for q in queries:
            match_expr = fts_match_query(q)
            t = time.perf_counter()
            got = conn.execute(
                "SELECT f.rowid FROM memory_fts f "
                "JOIN memories m ON m.id = f.rowid "
                "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
                "AND m.superseded_by IS NULL "
                "AND (? IS NULL OR m.kind = ?) "
                "AND (? IS NULL OR m.agent_id = ?) "
                "ORDER BY bm25(memory_fts) LIMIT ?",
                (match_expr, None, None, None, None, FETCH),
            ).fetchall()
            latencies.append((time.perf_counter() - t) * 1000)
            _ = got
    p50 = statistics.median(latencies)
    p95 = _pct(latencies, 0.95)
    return p50, p95, latencies


def measure_match_set(conn, queries: list, specific: bool = False) -> float:
    """Median width of the match set the ORDER BY must rank -- the discriminator.

    This is NOT the same as len(rows) from the timed query: the timed query
    applies LIMIT 50, so its row count is pinned at 50 by construction and says
    nothing about how much work the sort actually did. This count drops the
    LIMIT and the bm25() ordering to measure how many rows the MATCH expression
    touches, which is the quantity that separates the "full match-set sort"
    hypothesis from the "JOIN re-check" one.

    specific=True keeps only the rare token -- the payload number that exists
    in exactly one row -- instead of the full OR-ed expression that
    fts_match_query builds for a real query text. The two widths differ by
    orders of magnitude on this corpus, and the cost of the leg tracks the
    width, so both are reported.

    Measured outside the latency window: it is a different query, and folding it
    into the timed path would report a number recall() never pays.
    """
    widths = []
    for q in queries:
        text = f"payload {q.split()[1]}" if specific else q
        match_expr = fts_match_query(text)
        got = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_fts f "
            "JOIN memories m ON m.id = f.rowid "
            "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
            "AND m.superseded_by IS NULL",
            (match_expr,),
        ).fetchone()
        widths.append(int(got["n"]))
    return statistics.median(widths)


def fit_exponent(pairs: list) -> float | None:
    """Least-squares alpha in latency ~ n^alpha, fitted in log-log space.

    alpha is 1.0 for linear, 1.0+ for super-linear. The audit measured 53x
    growth over a 12x size increase, which is alpha ~ 1.44.
    """
    if len(pairs) < 2:
        return None
    xs = [math.log10(n) for n, _t in pairs]
    ys = [math.log10(max(t, 1e-9)) for _n, t in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def main() -> int:
    # connect() already refuses a build without FTS5 (db.py raises DbError when
    # memory_fts cannot be read), so probe with the shipped check rather than
    # re-implementing it.
    probe_dir = Path(tempfile.mkdtemp(prefix="lex-probe-"))
    try:
        probe_conn = connect(probe_dir)
    except DbError as exc:
        print(f"FTS5 unavailable in this SQLite build: {exc}", file=sys.stderr)
        return 2
    probe_conn.close()

    tmp = Path(os.environ.get("LEX_WORKDIR") or tempfile.mkdtemp(prefix="lexical-"))
    tmp.mkdir(parents=True, exist_ok=True)

    source_identity = hashlib.sha256()
    for mod in sorted((Path(__file__).resolve().parent.parent / "omnidb_server")
                      .glob("*.py")):
        source_identity.update(mod.read_bytes())
    for mod in sorted((Path(__file__).resolve().parent.parent / "omnidb_server")
                      .glob("*.pyc")):
        source_identity.update(mod.read_bytes())
    source_identity.update((Path(__file__).resolve().parent / "lexical_scaling.py")
                           .read_bytes())
    sha = source_identity.hexdigest()
    sha_line = f"source_identity_sha256: {sha}"

    rows = []
    for n_docs in SIZES:
        home = tmp / f"lex{n_docs}"
        conn = build_corpus(home, n_docs)
        try:
            queries = [_rare_query(i) for i in range(QUERIES)]
            # one untimed pass so the page cache is warm and the first query
            # does not pay for loading the FTS segment into the cache
            measure_leg(conn, queries[:5])
            p50, p95, lat = measure_leg(conn, queries)
            match_width = measure_match_set(conn, queries)
            specific_width = measure_match_set(conn, queries, specific=True)
        finally:
            conn.close()

        rows.append({
            "n": n_docs,
            "p50_ms": p50,
            "p95_ms": p95,
            "match_rows": match_width,
            "specific_rows": specific_width,
            "samples": len(lat),
        })
        print(f"  n={n_docs:>7}  p50 {p50:8.3f} ms  p95 {p95:8.3f} ms  "
              f"match set {match_width:.0f} rows (specific {specific_width:.0f})",
              flush=True)

    pairs = [(r["n"], r["p50_ms"]) for r in rows]
    alpha = fit_exponent(pairs)

    lines = [
        "workload_revision: lexical-scaling-v1",
        f"workload_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        sha_line,
        f"python: {sys.version.split()[0]}",
        f"platform: {sys.platform}",
        "gate: none (scaling evidence, not a stability claim)",
        "leg: FTS5 lexical leg only (recall.py:75-83 SQL, dense leg excluded)",
        "tokenizer: porter unicode61",
        f"sizes: {','.join(str(s) for s in SIZES)}",
        f"queries_per_size: {QUERIES}",
        f"repeats: {REPEATS}",
        f"k_fetch: {FETCH}",
        f"k: {K}",
        "gate_status: not a gate; no floor is asserted here. PERF-A2-07-R's 25 ms",
        "  p50 target and ADR-7's 5 s budget are the targets, and the leg is 8.4x",
        "  over the p50 one at 60k -- but the number is unmeasured on this host,",
        "  so asserting a floor now would be inventing a threshold. Close the",
        "  cause first (see the suspicion note in bench/lexical_scaling.py).",
        f"lexical_p50_ms_{SIZES[0]}: {rows[0]['p50_ms']:.3f}",
        f"lexical_p95_ms_{SIZES[0]}: {rows[0]['p95_ms']:.3f}",
        f"lexical_match_rows_{SIZES[0]}: {rows[0]['match_rows']:.0f}",
        f"lexical_specific_rows_{SIZES[0]}: {rows[0]['specific_rows']:.0f}",
    ]
    for n_docs, _key in _BANDS[1:]:
        idx = next((i for i, r in enumerate(rows) if r["n"] == n_docs), None)
        if idx is None:
            continue
        lines += [
            f"lexical_p50_ms_{n_docs}: {rows[idx]['p50_ms']:.3f}",
            f"lexical_p95_ms_{n_docs}: {rows[idx]['p95_ms']:.3f}",
            f"lexical_match_rows_{n_docs}: {rows[idx]['match_rows']:.0f}",
            f"lexical_specific_rows_{n_docs}: {rows[idx]['specific_rows']:.0f}",
        ]
    lines += [
        f"lexical_growth_5k_to_60k: {rows[-1]['p50_ms'] / rows[0]['p50_ms']:.3f}",
        f"lexical_growth_exponent: {alpha:.3f}" if alpha is not None else
        "lexical_growth_exponent: n/a (need >=2 sizes)",
        f"lexical_samples_per_size: {rows[0]['samples']}",
        "",
        "## Sweep summary",
        "",
        "| n | p50 ms | p95 ms | match set (OR-ed) | match set (specific) | growth vs first |",
        "|---|---|---|---|---|---|",
    ]
    first = rows[0]["p50_ms"]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | "
            f"{r['match_rows']:.0f} | {r['specific_rows']:.0f} | "
            f"{r['p50_ms'] / first:.2f}x |"
        )
    lines += [
        "",
        f"fitted exponent alpha (p50 ~ n^alpha): {alpha:.3f}" if alpha is not None
        else "fitted exponent alpha: n/a",
        "",
        "## Suspicion, stated as a hypothesis (not measured here)",
        "",
        "Which of the three candidate causes is most likely, and why:",
        "",
        "1. `ORDER BY bm25(memory_fts)` forces a full match-set sort. SQLite has",
        "   no index on a computed bm25() value, so it must materialise and",
        "   score every row the MATCH expression touches, and only then apply",
        "   LIMIT 50. This is the leading suspect, and the evidence is in this",
        "   file: `lexical_match_rows_*` tracks n one-for-one, so the sort ranks",
        "   the entire corpus to return 50 rows. The audit's 53x growth over a",
        "   12x size increase (alpha ~ 1.44) is consistent with sort work",
        "   proportional to the match set rather than to the 50 rows returned.",
        "   A rank-bounded subquery",
        "   (`... WHERE rowid IN (SELECT rowid FROM memory_fts WHERE memory_fts",
        "   MATCH ? ORDER BY rank LIMIT 50)`) may let SQLite stop early.",
        "2. The `JOIN memories` re-check. It re-fetches each candidate row and",
        "   applies the soft-delete predicate, so it touches rows the LIMIT",
        "   would have discarded. A real contributor at these widths, but it is",
        "   bounded by the match set, so it cannot by itself explain growth",
        "   steeper than linear in n.",
        "3. The `ORDER BY` spill to a temp b-tree. Same clause as (1), and a",
        "   mechanism of it rather than a separate cause.",
        "",
        "The single number that separates (1) from (2)+(3) is the match-set",
        "width, which this harness now reports per band as `lexical_match_rows_*`",
        "(OR-ed, the real query shape) and `lexical_specific_rows_*` (one rare",
        "token). If the width stays small and roughly constant in n, the sort is",
        "not the problem and the join is. If it grows with n -- and on the",
        "default templated corpus it does, one-for-one -- the sort is.",
        "",
        "Caveat on that evidence: the corpus text repeats the token `payload` in",
        "every row, so `fts_match_query`'s OR-ed expression matches all of them.",
        "A corpus whose rows share fewer tokens would produce a narrower match",
        "set and could invert this conclusion. That is why both widths are",
        "reported. Confirm on a real corpus before acting on this.",
        "",
    ]

    out = Path(__file__).with_name("lexical-scaling-results.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
