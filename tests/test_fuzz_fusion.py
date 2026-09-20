"""Property-based tests for the fusion surface (fusion.rrf, fusion.fts_match_query).

The mutation report showed zero direct coverage here, letting 13 mutants
survive. These properties pin the contract run_recall depends on: exact id
coverage, positive scores, dominance and k monotonicity, and the FTS5
quoting guarantee that keeps untrusted input from becoming MATCH syntax.
"""

import re
import sqlite3

from hypothesis import given, settings
from hypothesis import strategies as st

from omnidb_server.fusion import fts_match_query, rrf

_ID = st.integers(min_value=1, max_value=50)
_RANKING = st.lists(_ID, max_size=8, unique=True)
_RANKINGS = st.lists(_RANKING, max_size=4)
_K = st.integers(min_value=1, max_value=120)
_TOKEN_EXPR = re.compile(r'^"([^"]|"")*"$')
_INJECTION_SHAPES = (
    '"; DROP TABLE memories; --',
    "a' OR '1'='1",
    'x" OR "1"="1',
    "NEAR( OR NOT )",
    '"unterminated quote',
    "\\..* && || ^",
    "'; DELETE FROM memories WHERE '1'='1",
)


@settings(max_examples=50)
@given(rankings=_RANKINGS)
def test_rrf_output_covers_exactly_the_ranked_ids(rankings):
    """Every id in any ranking appears; no id outside the rankings does."""
    scores = rrf(rankings)
    assert set(scores) == {mid for ranked in rankings for mid in ranked}


@settings(max_examples=50)
@given(rankings=_RANKINGS, k=_K)
def test_rrf_scores_are_positive_floats(rankings, k):
    scores = rrf(rankings, k=k)
    assert all(isinstance(v, float) and v > 0.0 for v in scores.values())


@settings(max_examples=50)
@given(rankings=_RANKINGS, k1=_K, extra=st.integers(min_value=1, max_value=60))
def test_rrf_smaller_k_scores_elementwise_at_least_as_high(rankings, k1, extra):
    """1/(k1+p) >= 1/(k2+p) termwise whenever k1 < k2, for every id."""
    small = rrf(rankings, k=k1)
    large = rrf(rankings, k=k1 + extra)
    assert set(small) == set(large)
    assert all(small[mid] >= large[mid] for mid in large)


@st.composite
def _dominated_pair(draw):
    """Rankings where id `a` sits strictly earlier than id `b` in every list."""
    a = draw(_ID)
    b = draw(_ID.filter(lambda x: x != a))
    n_lists = draw(st.integers(min_value=1, max_value=4))
    rankings = []
    for _ in range(n_lists):
        others = draw(
            st.lists(_ID.filter(lambda x: x not in (a, b)), max_size=6, unique=True)
        )
        pos_a = draw(st.integers(min_value=0, max_value=len(others)))
        pos_b = draw(st.integers(min_value=pos_a + 1, max_value=len(others) + 1))
        rankings.append(
            others[:pos_a] + [a] + others[pos_a:pos_b - 1] + [b] + others[pos_b - 1:]
        )
    return rankings, a, b


@settings(max_examples=50)
@given(pair=_dominated_pair())
def test_rrf_dominant_id_scores_at_least_the_dominated_one(pair):
    rankings, a, b = pair
    scores = rrf(rankings, k=60)
    assert scores[a] >= scores[b]


@settings(max_examples=50)
@given(text=st.text(max_size=200))
def test_fts_output_is_quoted_or_joined_tokens_with_balanced_quotes(text):
    """Arbitrary unicode/punct never raises and only yields quoted OR-ed tokens."""
    expr = fts_match_query(text)
    assert expr.count('"') % 2 == 0
    if not expr:
        return
    for part in expr.split(" OR "):
        assert _TOKEN_EXPR.match(part), (text, expr)


@settings(max_examples=50)
@given(text=st.text(min_size=1, max_size=600), cap=st.integers(min_value=0, max_value=40))
def test_fts_max_terms_cap_respected(text, cap):
    tokens = fts_match_query(text, max_terms=cap)
    n = len(tokens.split(" OR ")) if tokens else 0
    assert n <= cap
    assert len(fts_match_query(text).split(" OR ")) <= 24  # default cap


@settings(max_examples=50)
@given(shape=st.sampled_from(_INJECTION_SHAPES))
def test_fts_injection_shapes_produce_quoted_literals_only(shape):
    expr = fts_match_query(shape)
    if not expr:
        # empty output iff nothing was tokenizable; run_recall skips the FTS leg
        assert not re.search(r"\w", shape)
        return
    assert expr.count('"') % 2 == 0
    for part in expr.split(" OR "):
        assert _TOKEN_EXPR.match(part), (shape, expr)


@settings(max_examples=50, deadline=None)
@given(text=st.text(min_size=1, max_size=120))
def test_fts_expr_is_accepted_by_a_real_fts5_table(text):
    """End-to-end: the built expression is valid MATCH syntax for SQLite FTS5."""
    expr = fts_match_query(text)
    if not expr:
        return  # mirrors run_recall, which skips the FTS leg on an empty expr
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(c)")
        conn.execute("INSERT INTO t(c) VALUES ('kubernetes rollout undo steps')")
        conn.execute("INSERT INTO t(c) VALUES ('drop table tyranny attempt')")
        conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,)).fetchall()
    finally:
        conn.close()
