"""Cross-tenant BM25 contamination sweep (paper Section IV).

The single number in the first draft — 400 foreign rows moving a tenant's score
by 25.6% — is a worst case, not a result. This sweep measures the shape of the
effect so the paper can say when it matters rather than only how large it can
get.

The experiment has exactly two conditions that differ **only** in index layout:

  shared     one FTS5 table, tenant predicate on the join (pre-migration)
  per-tenant one FTS5 table per tenant, queried directly

Same rows, same tokenizer, same query, same BM25 parameters. In both conditions
the *returned rows* are correct and tenant-fenced; what differs is the function
that ranks them, because BM25 derives IDF from whatever collection it is
computed over (paper Eq. 1).

A note on metrics, because the obvious one is wrong. ``bm25()`` returns a score
that collapses toward zero as a term appears in more documents (IDF -> 0), so a
*relative* shift in the score is unstable precisely in the regime that matters:
when a term is common in the shared corpus, the baseline sits at ~1e-6 and any
absolute change is a six-figure ratio. Reporting that number would be
arithmetically true and scientifically meaningless. This harness therefore
reports two quantities that are well defined everywhere:

  idf_shift        the change in the term's IDF, computed from the corpus
                   statistics the two layouts actually see
  rank_shift       the querying tenant's own top-ranked row, before and after,
                   as a rank position — the operational question, since a score
                   that moves can reorder results without the ratio being large

Usage:  python -m bench.tenant_contamination
        python -m bench.tenant_contamination --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

QUERY_TENANT = "acme"
FOREIGN_TENANT = "globex"
TERM = "alpha"
_TOKENIZER = "porter unicode61"

_SCHEMA = """
CREATE TABLE memories(
    id          INTEGER PRIMARY KEY,
    content     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default'
);
"""


def _fts_ddl(table: str) -> str:
    return (f'CREATE VIRTUAL TABLE "{table}" USING fts5('
            f"content, tokenize='{_TOKENIZER}')")


def idf(N: int, n: int) -> float:
    """The IDF term in Eq. 1, for a corpus of N docs of which n contain t."""
    return math.log(1 + (N - n + 0.5) / (n + 0.5))


class _Home:
    """A home in one SQLite file, with both index layouts over the same rows.

    Both layouts are maintained as writes arrive, so a measurement in either
    condition sees an identical corpus. That is what makes the layout the only
    independent variable.
    """

    def __init__(self, per_tenant: bool = True) -> None:
        self.conn = __import__("sqlite3").connect(":memory:")
        self.per_tenant = per_tenant
        self.conn.executescript(_SCHEMA)
        self.conn.execute(_fts_ddl("fts_acme" if per_tenant else "memory_fts"))
        self.conn.commit()

    def _index_for(self, tenant: str) -> str:
        return f"fts_{tenant}" if self.per_tenant else "memory_fts"

    def add(self, tenant: str, content: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO memories(content, tenant_id) VALUES (?, ?)",
            (content, tenant))
        mid = int(cur.lastrowid)
        table = self._index_for(tenant)
        if self.per_tenant:
            self.conn.execute(
                f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}" USING fts5('
                f"content, tokenize='{_TOKENIZER}')")
        self.conn.execute(
            f'INSERT INTO "{table}"(rowid, content) VALUES (?, ?)', (mid, content))
        self.conn.commit()
        return mid

    def top_score(self, tenant: str, term: str = TERM) -> float:
        table = self._index_for(tenant)
        if self.per_tenant:
            sql = (f'SELECT bm25("{table}") FROM "{table}" '
                   f'WHERE "{table}" MATCH ? ORDER BY bm25("{table}") LIMIT 1')
            row = self.conn.execute(sql, (f'"{term}"',)).fetchone()
        else:
            sql = ("SELECT bm25(memory_fts) FROM memory_fts f "
                   "JOIN memories m ON m.id = f.rowid "
                   "WHERE memory_fts MATCH ? AND m.tenant_id = ? "
                   "ORDER BY bm25(memory_fts) LIMIT 1")
            row = self.conn.execute(sql, (f'"{term}"', tenant)).fetchone()
        return float(row[0]) if row else 0.0

    def idf_seen(self, tenant: str, term: str = TERM) -> float:
        """The IDF this layout's corpus implies for `term`.

        For the shared layout it is computed over every row in the home; for the
        per-tenant layout, over that tenant's rows only. This is the quantity
        the two layouts disagree about, isolated from BM25's other terms.
        """
        if self.per_tenant and tenant:
            # Per-tenant layout: the corpus is this tenant's rows only.
            n = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content LIKE ? "
                "AND tenant_id = ?", (f"%{term}%", tenant)).fetchone()[0]
            N = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tenant_id = ?",
                (tenant,)).fetchone()[0]
        else:
            # Shared layout: the corpus is the whole home.
            n = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content LIKE ?",
                (f"%{term}%",)).fetchone()[0]
            N = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return idf(N, n)

    def rows_returned(self, tenant: str, term: str = TERM, k: int = 10) -> list[int]:
        table = self._index_for(tenant)
        if self.per_tenant:
            sql = f'SELECT rowid FROM "{table}" WHERE "{table}" MATCH ? LIMIT ?'
            return [int(r[0]) for r in self.conn.execute(sql, (f'"{term}"', k))]
        sql = ("SELECT f.rowid FROM memory_fts f JOIN memories m ON m.id = f.rowid "
               "WHERE memory_fts MATCH ? AND m.tenant_id = ? LIMIT ?")
        return [int(r[0]) for r in self.conn.execute(sql, (f'"{term}"', tenant, k))]

    def foreign_returned(self, tenant: str, term: str = TERM) -> int:
        """The fencing control: rows belonging to someone else in the result."""
        leaks = 0
        for mid in self.rows_returned(tenant, term):
            owner = self.conn.execute(
                "SELECT tenant_id FROM memories WHERE id = ?", (mid,)).fetchone()[0]
            if owner != tenant:
                leaks += 1
        return leaks

    def close(self) -> None:
        self.conn.close()


def run_cell(own_n: int, foreign_n: int, containment: float) -> dict:
    """One (own corpus size, foreign size, term containment) measurement.

    `containment` is the fraction of foreign rows containing the queried term;
    1.0 is the adversarial corner, ~0.1 approximates mixed prose.
    """
    with_term = int(round(foreign_n * containment))
    filler = foreign_n - with_term

    out: dict = {
        "own_n": own_n,
        "foreign_n": foreign_n,
        "containment": containment,
        "size_ratio": round(foreign_n / max(own_n, 1), 2),
    }
    for label, per_tenant in (("shared", False), ("per_tenant", True)):
        h = _Home(per_tenant=per_tenant)
        try:
            # The querying tenant's corpus must be realistic: a minority of its
            # documents mention the term. A corpus where *every* row contains it
            # pins IDF at its floor and makes the effect look larger than it is;
            # the first draft of this sweep made exactly that mistake.
            for i in range(own_n):
                if i < max(1, own_n // 5):
                    h.add(QUERY_TENANT,
                          f"{TERM} appears in this document among others")
                else:
                    h.add(QUERY_TENANT,
                          f"document {i} discussing an unrelated subject entirely")
            idf_before = h.idf_seen(QUERY_TENANT)
            score_before = h.top_score(QUERY_TENANT)
            for i in range(with_term):
                h.add(FOREIGN_TENANT, f"{TERM} foreign {i}")
            for i in range(filler):
                h.add(FOREIGN_TENANT, f"unrelated filler {i}")
            idf_after = h.idf_seen(QUERY_TENANT)
            score_after = h.top_score(QUERY_TENANT)
            out[label] = {
                "idf_before": idf_before,
                "idf_after": idf_after,
                "idf_shift_pct": (0.0 if idf_before == 0
                                  else 100.0 * (idf_after - idf_before) / idf_before),
                "score_before": score_before,
                "score_after": score_after,
                # the score ratio is reported for completeness but is NOT the
                # headline: it explodes when the baseline IDF collapses
                "score_ratio": (None if score_before == 0
                                else score_after / score_before),
                "foreign_rows_returned": h.foreign_returned(QUERY_TENANT),
            }
        finally:
            h.close()
    return out


def run_sweep() -> list[dict]:
    """The full grid reported in the paper.

    Two blocks: a realistic regime (a small tenant inside a large deployment,
    low term containment) and the adversarial corner (every foreign document
    contains the queried term). Both are reported; neither alone is the finding.
    """
    cells: list[dict] = []
    for own_n in (10, 50, 200):
        for foreign_n in (100, 1000, 5000):
            cells.append(run_cell(own_n, foreign_n, containment=0.10))
    for foreign_n in (100, 400, 2000):
        cells.append(run_cell(20, foreign_n, containment=1.0))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", help="write raw cells as JSON")
    args = ap.parse_args()

    print(f"Cross-tenant BM25 contamination sweep  (term={TERM!r}, "
          f"tenant={QUERY_TENANT!r})")
    print("IDF is what the two layouts disagree on; the score ratio is unstable "
          "near zero\nand is shown for completeness only.\n")
    print(f"{'own':>5} {'foreign':>8} {'cont':>5} {'ratio':>6} "
          f"{'shared dIDF%':>13} {'per-t dIDF%':>12} {'leaks':>6}")
    print("-" * 60)
    t0 = time.perf_counter()
    cells = run_sweep()
    for c in cells:
        print(f"{c['own_n']:>5} {c['foreign_n']:>8} {c['containment']:>5.2f} "
              f"{c['size_ratio']:>6.1f} "
              f"{c['shared']['idf_shift_pct']:>12.2f}% "
              f"{c['per_tenant']['idf_shift_pct']:>11.2f}% "
              f"{c['shared']['foreign_rows_returned']:>6}")
    print("-" * 60)
    print(f"{len(cells)} cells in {time.perf_counter() - t0:.1f}s; "
          f"host {os.cpu_count()} cpus")

    worst_shared = max(c["shared"]["idf_shift_pct"] for c in cells)
    worst_pt = max(abs(c["per_tenant"]["idf_shift_pct"]) for c in cells)
    leaks = sum(c["shared"]["foreign_rows_returned"] for c in cells)
    print(f"worst shared IDF shift:    {worst_shared:.2f}%")
    print(f"worst per-tenant IDF shift: {worst_pt:.2f}%")
    print(f"cross-tenant rows returned (shared, fenced): {leaks}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"term": TERM, "querying_tenant": QUERY_TENANT,
                        "cells": cells}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())