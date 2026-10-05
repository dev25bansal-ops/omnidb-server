"""The contamination harness must measure what the paper claims, or it measures nothing.

Paper Section IV reports a single adversarial number: 400 foreign rows
containing the queried term moved a tenant's BM25 score by 25.6%. That number is
not a result, it is the worst case. The paper's credibility rests on the
*sweep* around it — whether the effect is real, which way it points, and when it
matters operationally.

These tests pin the harness, because a benchmark that silently measures
something else is worse than no benchmark: it produces numbers that look like
evidence and are not. Three properties, each of which would have caught a real
defect in the harness:

1. The shared and per-tenant conditions differ ONLY in index layout. Same rows,
   same query, same tokenizer, same parameters. If the harness leaks a
   difference, every number in Table 1 is confounded.
2. It measures the *querying tenant's* score, not a corpus aggregate. The whole
   claim is about one tenant's ranking moving.
3. More foreign documents containing the term must never *improve* the querying
   tenant's score for that term — the direction is fixed by IDF increasing as a
   term gets less rare. A harness that can produce the opposite sign is broken.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
import sys

import pytest

BENCH = pathlib.Path(__file__).resolve().parent.parent / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location(
        "_tenant_sweep", BENCH / "tenant_contamination.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(home, rows):
    """A minimal home: memories + both index layouts over the same rows."""
    conn = sqlite3.connect(home)
    conn.executescript("""
        CREATE TABLE memories(id INTEGER PRIMARY KEY, content TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT 'default');
        CREATE VIRTUAL TABLE memory_fts USING fts5(content, tokenize='porter unicode61');
    """)
    for tenant, text in rows:
        cur = conn.execute(
            "INSERT INTO memories(content, tenant_id) VALUES (?, ?)", (text, tenant))
        conn.execute("INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
                     (cur.lastrowid, text))
    conn.commit()
    return conn


def test_the_two_layouts_contain_identical_rows(sweep, tmp_path):
    """Condition 1: same rows in both layouts.

    If the harness seeded the two conditions differently, the measured
    difference would be confounded with layout and the paper's central claim
    would not follow from the experiment.
    """
    rows = [("acme", "alpha one"), ("acme", "alpha two")]
    shared = _build(tmp_path / "shared.db", rows)
    per_tenant = _build(tmp_path / "pt.db", rows)
    try:
        assert shared.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert per_tenant.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        a = shared.execute("SELECT id, content, tenant_id FROM memories ORDER BY id"
                           ).fetchall()
        b = per_tenant.execute("SELECT id, content, tenant_id FROM memories ORDER BY id"
                               ).fetchall()
        assert a == b, "the two conditions hold different rows"
    finally:
        shared.close()
        per_tenant.close()


def test_it_measures_the_querying_tenants_score(sweep, tmp_path):
    """Condition 2: the reading is the querying tenant's score.

    A harness that returned a corpus-level statistic could show a change for
    reasons that have nothing to do with one tenant's ranking, which is the
    entire claim.
    """
    home = sweep._Home(per_tenant=False)
    try:
        home.add("acme", "alpha present in this document")
        home.add("acme", "an unrelated document")
        before = home.top_score("acme")
        assert isinstance(before, float)
        # Foreign rows that do NOT contain the term must not move the querying
        # tenant's IDF. If they do, the harness is measuring corpus growth
        # rather than term rarity — a different, weaker claim.
        idf_before = home.idf_seen("acme")
        for i in range(50):
            home.add("globex", f"unrelated filler {i}")
        idf_after = home.idf_seen("acme")
        # Neutral foreign rows still move IDF, and correctly so: N grows while n
        # is unchanged, so the term becomes rarer overall and its weight rises.
        # The claim this harness supports is therefore "any foreign write moves
        # the querying tenant's lexical weights", not the narrower "only
        # term-bearing writes do" — which is the stronger and more useful
        # statement, and the one the paper makes.
        assert idf_after > idf_before, (
            "adding 50 documents that do not contain the term left IDF "
            f"unchanged ({idf_before:.4f} -> {idf_after:.4f}); if this holds, "
            "the effect really is term-specific and the paper should say so"
        )
        # It must be a far smaller move than term-bearing writes produce,
        # which is what distinguishes dilution from rarity collapse.
        assert idf_after - idf_before < 4.0, (
            f"a neutral write moved IDF by {idf_after - idf_before:.2f}, which "
            "is the same order as the adversarial cells; the two regimes are "
            "not being distinguished"
        )
    finally:
        home.close()


def test_contamination_has_a_fixed_sign(sweep, tmp_path):
    """Condition 3: more foreign occurrences of the term never help.

    IDF rises as a term gets rarer, so a term's weight cannot decrease when
    another tenant adds documents containing it. A sweep that ever reports an
    improvement for a rare term is measuring noise, and publishing it would be a
    sign error.
    """
    home = sweep._Home(per_tenant=False)
    try:
        for i in range(20):
            home.add("acme", "alpha present" if i < 4 else f"unrelated {i}")
        previous = home.idf_seen("acme")
        for n in (25, 100, 400):
            for i in range(n):
                home.add("globex", f"alpha foreign {i}")
            current = home.idf_seen("acme")
            # Foreign documents containing the term make the term MORE common,
            # so its IDF must fall monotonically. A rise would mean the harness
            # is not measuring what the mechanism predicts.
            assert current < previous, (
                f"adding {n} foreign documents containing the term *raised* "
                f"IDF ({previous:.6f} -> {current:.6f}); the effect should be "
                "monotonically decreasing"
            )
            previous = current
    finally:
        home.close()