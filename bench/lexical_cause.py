"""Cause isolation for the FTS5 lexical leg (closes the PERF-A2-07-R open thread).

Two harnesses disagreed about the growth exponent of the same leg: this one
measured alpha ~ 1.60 over 5k/25k/60k, bench/lexical_scaling.py reported
alpha ~ 0.949 over 800/2000/4000. Both cannot be right for one leg, and neither
says WHICH clause is responsible. This file answers the clause question by
timing SQL variants that differ in exactly one property each, on one corpus,
with one match expression per query. Every variant is a real production-legal
statement; the differences are the ranking clause and the join.

The variants, and what each one isolates:

  V0 prod_order_bm25   the shipped statement, byte-shaped like recall.py:85-94
  V1 no_order          same, ORDER BY removed entirely -> V0-V1 is the whole
                       cost of ranking the match set, holding MATCH constant
  V2 fts_only_rank     rank-bounded inside FTS, no join to memories
                       -> V0-V2 is the cost the join adds
  V3 join_only         the join and row fetch, no ranking at all
                       -> V0-V3 is ranking + join interaction
  V4 match_count       COUNT(*) over the match expression, no ranking, no rows
                       -> the floor: what MATCH alone costs
  V5 single_term       ORDER BY bm25 LIMIT 50 on a ONE-token expression that
                       matches one row -> the same ranking machinery with a
                       match set of width 1

V5 against V0 is the load-bearing comparison. Both rank 50 rows with bm25 and
both go through the join; the only difference is how many rows the MATCH
expression touches. If V5 stays flat while V0 grows with n, the cost is
proportional to the match set and the ranking clause is the mechanism. If V5
grows too, the cost is per-query constant overhead and the match set is
incidental.

Why the match set is the size it is: fusion.fts_match_query ORs up to 24
lowercased word tokens, and so does every row of the corpus carry several of
them, so the expression matches most of the corpus. That is a design property,
not a benchmark artifact, which is why this corpus is deliberately templated
the way agent memory actually looks rather than made artificially disjoint.

Run:  python bench/lexical_cause.py
Env:  CAUSE_SIZES=5000,25000,60000   CAUSE_QUERIES=20   CAUSE_REPEATS=2
      CAUSE_WORKDIR=<writable volume>
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server.db import DbError, connect  # noqa: E402
from omnidb_server.fusion import fts_match_query  # noqa: E402

SIZES = [int(x) for x in os.environ.get("CAUSE_SIZES", "5000,25000,60000").split(",")]
QUERIES = int(os.environ.get("CAUSE_QUERIES", "20"))
REPEATS = int(os.environ.get("CAUSE_REPEATS", "2"))
FETCH = 50  # recall.py's k_fetch default

_SELECTIVE = "deleted_ms IS NULL AND superseded_by IS NULL"

# The six statements. All six take (match_expr, fetch) and return rows.
_VARIANTS: list[tuple[str, str]] = [
    ("V0_prod_order_bm25",
     "SELECT f.rowid FROM memory_fts f "
     "JOIN memories m ON m.id = f.rowid "
     "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
     "AND m.superseded_by IS NULL "
     "ORDER BY bm25(memory_fts) LIMIT ?"),
    ("V1_no_order",
     "SELECT f.rowid FROM memory_fts f "
     "JOIN memories m ON m.id = f.rowid "
     "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
     "AND m.superseded_by IS NULL "
     "LIMIT ?"),
    ("V2_fts_only_rank",
     "SELECT rowid FROM memory_fts "
     "WHERE memory_fts MATCH ? "
     "ORDER BY rank LIMIT ?"),
    ("V3_join_only",
     "SELECT f.rowid FROM memory_fts f "
     "JOIN memories m ON m.id = f.rowid "
     "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
     "AND m.superseded_by IS NULL "
     "LIMIT ?"),
    ("V5_single_term",
     "SELECT f.rowid FROM memory_fts f "
     "JOIN memories m ON m.id = f.rowid "
     "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
     "AND m.superseded_by IS NULL "
     "ORDER BY bm25(memory_fts) LIMIT ?"),
]

# V6 is the candidate fix, measured with the identical statement as V0 and only
# the MATCH expression changed. It asks whether narrowing the match set -- which
# is the one quantity V0 - V5 showed to be the whole cost -- actually buys the
# time back. Same SQL text as V0, so any difference is attributable to the
# expression and to nothing else.
_V6_SQL = _VARIANTS[0][1]

_COUNT_SQL = (
    "SELECT COUNT(*) AS n FROM memory_fts f "
    "JOIN memories m ON m.id = f.rowid "
    "WHERE memory_fts MATCH ? AND m.deleted_ms IS NULL "
    "AND m.superseded_by IS NULL"
)


def _make_text(i: int) -> str:
    """Row text shaped like what remember() actually stores.

    Six distinct tokens per row, two of which recur in every row. Real agent
    memory shares ordinary words across rows -- "the", "config", "note" -- and
    that sharing is what makes an OR-ed match expression wide. A corpus of
    disjoint nonsense tokens would make every lookup fast and every conclusion
    drawn from it wrong.
    """
    return (f"note {i} about config note: the model provider note {i % 97} "
            f"was configured with reference ref{i}qz.")


def _unique_token(i: int) -> str:
    """A token that occurs in exactly one row.

    The obvious candidate -- the bare row number -- is not unique, and using it
    is what made the first run of this file report a single-term width of 52
    at n=5000 instead of 1: "7" also tokenises out of every row whose
    `i % 97` bucket is 7. Folding the id into one alphanumeric token removes
    the collision, and the harness asserts width == 1 rather than trusting it.
    """
    return f"ref{i}qz"


def _and_expr(i: int) -> str:
    """The same terms as the shipped expression, AND-ed instead of OR-ed.

    This is the fix candidate, not a strawman: it keeps every token
    fts_match_query would have used and changes only the operator, so the
    comparison against V0 isolates the cost of match-set width with the term
    set held constant. An OR of ten terms each of which appears in most rows
    matches most rows; the AND of the same terms matches only rows containing
    all of them.
    """
    question = f"what did note {i} say about the model provider config"
    tokens = re.findall(r"\w+", question.lower())
    return " AND ".join('"%s"' % t for t in tokens)


def _query_for(i: int) -> tuple[str, str]:
    """Return (shipped_expression, single_term_expression) for query i.

    The shipped expression is what fts_match_query builds from an ordinary
    question, which is ORs over tokens that also occur in most other rows.
    The single-term expression is one token present in a single row, so its
    match set is width 1 at every corpus size.
    """
    question = f"what did note {i} say about the model provider config"
    return fts_match_query(question), '"%s"' % _unique_token(i)


def build_corpus(home: Path, n_docs: int):
    """Create a real schema and insert n_docs rows the way the store does.

    The explicit memory_fts insert is not optional decoration. db.py's
    _MIGRATION_5 dropped the three FTS triggers and replaced the
    external-content table with a standalone one, so the index is now synced
    by the store write path (store.py:571, 658) inside the same transaction.
    A harness that inserts into `memories` over raw SQL therefore builds a
    corpus the lexical leg cannot see: MATCH matches nothing, every variant
    returns 50 rows in microseconds, and the whole sweep reports a healthy
    flat line. That is not a hypothetical -- bench/lexical_scaling.py and
    bench/verify_qdrant_rootcause.py both build their corpora this way, and
    both produce numbers that cannot be reproduced on current code.
    """
    conn = connect(home)
    now_ms = 1_700_000_000_000
    conn.executemany(
        "INSERT INTO memories (kind, content, metadata, created_ms) "
        "VALUES ('semantic', ?, '{}', ?)",
        [(_make_text(i), now_ms + i) for i in range(n_docs)])
    # Sync the index from what was actually stored, not from a re-derived copy
    # of the text: reading the rows back keeps this correct even if a future
    # migration truncates or rewrites content on the way in.
    conn.executemany(
        "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
        [(int(r["id"]), r["content"]) for r in conn.execute(
            "SELECT id, content FROM memories")])
    conn.commit()
    conn.execute("ANALYZE")
    conn.commit()
    return conn


def _time(conn, sql: str, exprs: list[str], nparams: int = 2) -> tuple[float, float]:
    """Median and p95 of one statement over exprs.

    nparams is 2 for every LIMIT-bearing variant (match expr, limit) and 1 for
    the COUNT(*) probe, which takes no limit. Deriving it from the SQL instead
    of hardcoding it per call keeps a variant from silently binding the wrong
    arity and failing only at the largest corpus size.
    """
    lat = []
    for _rep in range(REPEATS):
        for expr in exprs:
            params = (expr, FETCH) if nparams == 2 else (expr,)
            t = time.perf_counter()
            conn.execute(sql, params).fetchall()
            lat.append((time.perf_counter() - t) * 1000)
    s = sorted(lat)
    return statistics.median(lat), s[min(len(s) - 1, int(len(s) * 0.95))]


def _fit_alpha(pairs: list[tuple[int, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [math.log10(n) for n, _ in pairs]
    ys = [math.log10(max(t, 1e-9)) for _n, t in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main() -> int:
    probe = Path(tempfile.mkdtemp(prefix="cause-probe-"))
    try:
        connect(probe).close()
    except DbError as exc:
        print(f"FTS5 unavailable in this SQLite build: {exc}", file=sys.stderr)
        return 2

    tmp = Path(os.environ.get("CAUSE_WORKDIR") or tempfile.mkdtemp(prefix="cause-"))
    tmp.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent.parent
    src = hashlib.sha256()
    for mod in sorted((root / "omnidb_server").glob("*.py")):
        src.update(mod.read_bytes())
    src.update(Path(__file__).read_bytes())

    rows: list[dict] = []
    for n_docs in SIZES:
        conn = build_corpus(tmp / f"cause{n_docs}", n_docs)
        try:
            pairs = [_query_for(i) for i in range(QUERIES)]
            shipped = [p for p, _ in pairs]
            single = [s for _, s in pairs]
            and_exprs = [_and_expr(i) for i in range(QUERIES)]
            widths = [int(conn.execute(_COUNT_SQL, (e,)).fetchone()["n"])
                      for e in shipped]
            widths_single = [int(conn.execute(_COUNT_SQL, (e,)).fetchone()["n"])
                             for e in single]
            widths_and = [int(conn.execute(_COUNT_SQL, (e,)).fetchone()["n"])
                          for e in and_exprs]
            # warm the cache so the first timed query does not pay for loading
            # the FTS segment; not folded into any reported number
            _time(conn, _VARIANTS[0][1], shipped[:3])
            band = {"n": n_docs,
                    "width": statistics.median(widths),
                    "width_single": statistics.median(widths_single)}
            for name, sql in _VARIANTS:
                exprs = single if name == "V5_single_term" else shipped
                p50, p95 = _time(conn, sql, exprs)
                band[name] = p50
                band[name + "_p95"] = p95
            band["V4_match_count"] = _time(conn, _COUNT_SQL, shipped, nparams=1)[0]
            p50_and, p95_and = _time(conn, _V6_SQL, and_exprs)
            band["V6_and_narrow"] = p50_and
            band["V6_and_narrow_p95"] = p95_and
            band["width_and"] = statistics.median(widths_and)
        finally:
            conn.close()
        # A zero-width match set means the index was never populated, and every
        # number below would then be the cost of returning nothing. Fail loudly
        # instead of publishing a flat, fast, meaningless sweep -- the failure
        # mode that produced the two irreproducible lexical artifacts already
        # in bench/.
        if band["width"] <= 0 or band["width_single"] <= 0:
            print(f"REFUSING TO REPORT: at n={n_docs} the shipped expression "
                  f"matched {band['width']:.0f} rows and the single-term "
                  f"expression matched {band['width_single']:.0f}. The FTS "
                  f"index is empty for this corpus, so every variant below is "
                  f"timing a no-op.", file=sys.stderr)
            return 3
        if band["width_single"] != 1:
            print(f"REFUSING TO REPORT: at n={n_docs} the single-term probe "
                  f"matched {band['width_single']:.0f} rows, expected 1. The "
                  f"width discriminator needs a token unique to one row.",
                  file=sys.stderr)
            return 3
        rows.append(band)
        v0 = band["V0_prod_order_bm25"]
        print(f"  n={n_docs:>7}  V0 {v0:8.3f}  V1 {band['V1_no_order']:8.3f}  "
              f"V2 {band['V2_fts_only_rank']:8.3f}  V4 {band['V4_match_count']:7.3f}"
              f"  V5 {band['V5_single_term']:7.3f}  width {band['width']:.0f}",
              flush=True)

    names = [n for n, _ in _VARIANTS] + ["V4_match_count", "V6_and_narrow"]
    alphas = {n: _fit_alpha([(r["n"], r[n]) for r in rows]) for n in names}

    lines = [
        "workload_revision: lexical-cause-v1",
        f"workload_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        f"source_identity_sha256: {src.hexdigest()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {sys.platform}",
        "gate: none (cause isolation, not a stability claim)",
        "leg: FTS5 lexical leg, variants differ in ranking clause and join only",
        f"sizes: {','.join(str(s) for s in SIZES)}",
        f"queries_per_size: {QUERIES}",
        f"repeats: {REPEATS}",
        f"k_fetch: {FETCH}",
        "",
        "## p50 by corpus size (ms)",
        "",
        "| variant | " + " | ".join(str(r["n"]) for r in rows) + " | alpha |",
        "|---|" + "---|" * (len(rows) + 1),
    ]
    for n in names:
        cells = " | ".join(f"{r[n]:.3f}" for r in rows)
        a = alphas[n]
        lines.append(f"| {n} | {cells} | "
                     f"{a:.3f} |" if a is not None else f"| {n} | {cells} | n/a |")
    lines.append("| match width (OR-ed) | "
                 + " | ".join(f"{r['width']:.0f}" for r in rows) + " | n/a |")
    lines.append("| match width (single term) | "
                 + " | ".join(f"{r['width_single']:.0f}" for r in rows)
                 + " | n/a |")
    lines.append("| match width (AND-ed) | "
                 + " | ".join(f"{r['width_and']:.0f}" for r in rows)
                 + " | n/a |")
    lines += [
        "",
        "## Deltas at the largest size (ms) -- what each clause costs",
        "",
    ]
    big = rows[-1]
    deltas = [
        ("V0 - V1  ranking the match set (ORDER BY bm25 removed)",
         big["V0_prod_order_bm25"] - big["V1_no_order"]),
        ("V0 - V3  ranking on top of the join's row set",
         big["V0_prod_order_bm25"] - big["V3_join_only"]),
        ("V1 - V4  fetching and joining rows, MATCH held constant",
         big["V1_no_order"] - big["V4_match_count"]),
        ("V0 - V5  cost attributable to match-set width",
         big["V0_prod_order_bm25"] - big["V5_single_term"]),
        ("V0 - V2  what the join adds over rank-bounded FTS alone",
         big["V0_prod_order_bm25"] - big["V2_fts_only_rank"]),
        ("V0 - V6  saving from AND-ing the same terms (fix candidate)",
         big["V0_prod_order_bm25"] - big["V6_and_narrow"]),
    ]
    for label, val in deltas:
        lines.append(f"{label:<58} {val:+9.3f}")

    out = Path(__file__).with_name("lexical-cause-results.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
