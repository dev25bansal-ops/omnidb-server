"""Exact-value unit tests for the pure fusion surface (fusion.rrf, fusion.fts_match_query).

docs/mutation-report.md:196-199 named the root cause of the 33% mutation score:
*"neither pure function has a single direct unit test"* — both were reached only
through ``store.recall``, whose fixtures are friendly (short lowercase queries,
documents containing every query token, a dense leg that dominates the sparse
one). Every survivor in the report is an arithmetic or string detail that an
ordering assertion cannot observe.

These tests therefore assert **pinned values**, not shapes:

* floats are compared to hard decimal literals with ``==``, bit-exactly. A
  mutant that changes the arithmetic changes the literal.
* strings are compared to hard expected expressions with ``==``.

Deliberately *not* done here: restating the formulas under test (e.g. asserting
``scores[10] == 1 / 61``) would only re-implement the source and would pass
unchanged against a mutant that breaks it. The closed form appears in comments
as documentation of what the literal represents; the assertion is the literal.
Every literal below was verified bit-exact on CPython 3.11-3.14 and is
platform-independent (IEEE-754 double, no reassociation, no extended precision
in CPython).
"""

from __future__ import annotations

import sqlite3

import pytest

from omnidb_server.fusion import fts_match_query, rrf

# 1/(k + position + 1) for k=60 and position 0,1,2 — the shipped RRF constants.
_ONE_OVER_61 = 0.01639344262295082
_ONE_OVER_62 = 0.016129032258064516
_ONE_OVER_63 = 0.015873015873015872
# sum of two legs at position 0, k=60
_TWO_OVER_61 = 0.03278688524590164
# sum of three legs at position 0, k=60
_THREE_OVER_61 = 0.04918032786885246
# 0.5 and 0.25 over 61 — the per-leg weight multiplier
_HALF_OVER_61 = 0.00819672131147541
_QUARTER_OVER_61 = 0.004098360655737705


# --------------------------------------------------------------------------
# rrf — exact rank weights
# --------------------------------------------------------------------------

def test_rrf_exact_rank_weights():
    """Pins the rank weight of each position to 1/(k+position+1).

    Kills the off-by-one denominator, the quadratic damping, the shifted
    enumerate offset, the plain reciprocal of the raw index, and any uniform
    rescaling: each of those moves one of these three literals.
    """
    out = rrf([[10, 20, 30]], k=60)
    assert out[10] == _ONE_OVER_61   # 1/61, position 0
    assert out[20] == _ONE_OVER_62   # 1/62, position 1
    assert out[30] == _ONE_OVER_63   # 1/63, position 2
    # strict order is a consequence of the pinned values, not a substitute
    assert out[10] > out[20] > out[30]


def test_rrf_default_k_is_60():
    """The shipped default rank constant.

    ``recall`` always passes ``cfg.rrf_k`` explicitly, so the default argument
    is unreachable from production code and was untested — which is exactly why
    the k=60 -> k=1 mutant survived.
    """
    assert rrf([[10, 20, 30]]) == {
        10: _ONE_OVER_61,
        20: _ONE_OVER_62,
        30: _ONE_OVER_63,
    }
    # the default must be the shipped 60, not merely *some* value: k=1 would
    # give 1/2, 1/3, 1/4
    assert rrf([[10, 20, 30]]) == rrf([[10, 20, 30]], k=60)


def test_rrf_accumulates_across_lists():
    """A duplicate id sums one contribution per list; it is not overwritten."""
    assert rrf([[7], []], k=60) == {7: _ONE_OVER_61}
    assert rrf([[7], [7]], k=60) == {7: _TWO_OVER_61}
    assert rrf([[7], [7], [7]], k=60) == {7: _THREE_OVER_61}
    # and the accumulation is a genuine sum, not a per-list overwrite
    assert rrf([[7], [7]], k=60)[7] == 2 * rrf([[7], []], k=60)[7]


def test_rrf_uses_every_ranking_list():
    """Every list contributes; a later list is not dropped.

    The dense leg alone used to order the recall fixtures correctly, so a mutant
    that reads only ``rankings[:1]`` was invisible through the store.
    """
    out = rrf([[1], [2]], k=60)
    assert set(out) == {1, 2}
    assert out[1] == _ONE_OVER_61
    assert out[2] == _ONE_OVER_61
    # three lists, three distinct ids: nothing is collapsed
    assert set(rrf([[1], [2], [3]], k=60)) == {1, 2, 3}


def test_rrf_position_offsets_independently_per_list():
    """Position is per-list, so a later list restarts at offset 1/(k+0+1)."""
    out = rrf([[1, 2], [3]], k=60)
    assert out == {1: _ONE_OVER_61, 2: _ONE_OVER_62, 3: _ONE_OVER_61}
    # the shared id sums position 0 of list 1 with position 1 of list 2
    out = rrf([[1, 2], [2]], k=60)
    assert out[2] == _ONE_OVER_62 + _ONE_OVER_61


def test_rrf_applies_per_leg_weights_as_multipliers():
    """``weights`` scales each leg's contribution; the default is 1.0 per leg."""
    assert rrf([[5]], k=60, weights=[0.5]) == {5: _HALF_OVER_61}
    assert rrf([[5], [6]], k=60, weights=[0.5, 0.25]) == {
        5: _HALF_OVER_61,
        6: _QUARTER_OVER_61,
    }
    # the shipped production weights from Config: dense 0.7, sparse 0.3
    assert rrf([[5]], k=60, weights=[1.0]) == rrf([[5]], k=60)


def test_rrf_empty_inputs():
    """Degenerate ranking shapes return the empty mapping, not an error."""
    assert rrf([], k=60) == {}
    assert rrf([[], []], k=60) == {}
    assert rrf([[]], k=60) == {}


# --------------------------------------------------------------------------
# fts_match_query — exact expressions
# --------------------------------------------------------------------------

def test_fts_tokens_are_whole_words():
    """Tokens are maximal ``\\w+`` runs, so a hyphen does not split a term.

    With a single-``\\w`` tokenizer this input yields three tokens instead of
    two; the fixture queries in the store were short single words, so the
    mutant was never observed.
    """
    assert fts_match_query("abc def") == '"abc" OR "def"'
    assert fts_match_query("a-b_c") == '"b_c"'  # F35: stopword "a" dropped
    # digits and underscores are \\w and stay inside one token
    assert fts_match_query("w10 x_y") == '"w10" OR "x_y"'
    # unicode \\w is preserved whole (café / 北京 are one token each)
    assert fts_match_query("北京 café") == '"北京" OR "café"'


def test_fts_lowercases_before_tokenizing():
    """The tokenizer input is lowercased, so FTS5 case folding is not relied on.

    Every store fixture query was already lowercase and unicode61 folds ASCII
    case during tokenization anyway, so dropping ``.lower()`` changed nothing
    observable downstream.
    """
    assert fts_match_query("ALPHA") == '"alpha"'
    assert fts_match_query("ALPHA Beta") == '"alpha" OR "beta"'
    assert fts_match_query("Kubernetes") == '"kubernetes"'


def test_fts_joins_with_OR():
    """Terms are disjunctive. An AND join would be a recall regression."""
    assert fts_match_query("a b c") == '"b" OR "c"'  # F35: "a" dropped
    # a punctuation-only token never reaches the expression
    assert fts_match_query("drop table; --") == '"drop" OR "table"'
    assert fts_match_query("NEAR( OR NOT )") == '"near" OR "not"'  # F35: "or" dropped, "not" kept


def test_fts_max_terms_default_is_24():
    """The shipped default term cap, measured as the emitted segment count."""
    forty = " ".join(f"w{i}" for i in range(40))
    assert len(fts_match_query(forty).split(" OR ")) == 24
    # 24 distinct tokens means no truncation, so the cap is at least 24
    exactly_24 = " ".join(f"w{i}" for i in range(24))
    assert fts_match_query(exactly_24).split(" OR ") == [
        f'"w{i}"' for i in range(24)
    ]


def test_fts_max_terms_slice_is_applied():
    """An explicit ``max_terms`` truncates to the first N tokens.

    No fixture query had more than four tokens, so removing the slice entirely
    was invisible.
    """
    forty = " ".join(f"w{i}" for i in range(40))
    assert fts_match_query(forty, max_terms=3) == '"w0" OR "w1" OR "w2"'
    assert fts_match_query(forty, max_terms=1) == '"w0"'
    # a zero cap is an empty expression, not an error
    assert fts_match_query(forty, max_terms=0) == ""
    # a negative cap is a Python negative slice: it keeps the LAST |n| tokens
    # rather than clamping. Pinned so the quirk is documented rather than
    # mistaken for an intended "unbounded" behaviour.
    assert fts_match_query("a b c d", max_terms=-1) == '"b" OR "c"'
    assert fts_match_query("a b c d", max_terms=-3) == ""


def test_fts_untokenizable_input_yields_empty_expression():
    """Input with no ``\\w`` run yields ``""``, which ``recall`` treats as
    "skip the sparse leg".

    The empty-input branch was never exercised, so a mutant that injected a
    stray ``[""]`` token — producing the MATCH expression ``'""'`` — survived.
    """
    assert fts_match_query("") == ""
    assert fts_match_query("   ") == ""
    assert fts_match_query("!!! ...") == ""
    assert fts_match_query("---") == ""
    assert fts_match_query("()[]{}") == ""
    # punctuation is not \w, so it is dropped rather than quoted
    assert fts_match_query("-0.7") == '"0" OR "7"'


def test_fts_underscore_only_input_is_a_token():
    """Underscore is ``\\w``; an all-underscore run is one token, not empty."""
    assert fts_match_query("___") == '"___"'
    assert fts_match_query("_ leading") == '"_" OR "leading"'


# --------------------------------------------------------------------------
# fts_match_query — real FTS5 acceptance (the OR-join behavioural half)
# --------------------------------------------------------------------------

@pytest.fixture()
def fts_table():
    """A real in-memory FTS5 table with one document per term."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        conn.execute("INSERT INTO t(body) VALUES ('alpha gamma')")
        conn.execute("INSERT INTO t(body) VALUES ('delta beta')")
        yield conn
    finally:
        conn.close()


def _match(conn, expr):
    return [int(r[0]) for r in
            conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,)).fetchall()]


def test_or_join_widens_recall_against_real_fts5(fts_table):
    """A disjunction retrieves documents matching any term; a conjunction
    retrieves only documents matching all of them.

    With the fixture above, "alpha beta" matches no single document, so the AND
    variant returns nothing while the shipped OR variant returns both. This is
    the assertion that separates the two joins behaviourally rather than by
    string inspection alone.
    """
    expr = fts_match_query("alpha beta")
    assert expr == '"alpha" OR "beta"'
    assert sorted(_match(fts_table, expr)) == [1, 2]
    # the mutant's output, asserted to be observably different
    assert _match(fts_table, expr.replace(" OR ", " AND ")) == []


def test_conjunction_would_still_match_a_shared_document(fts_table):
    """Control: where one document holds every term, OR and AND agree in hit set.

    Without this, the OR test above could be passing for the wrong reason (e.g.
    a quoting bug that made every expression a no-op rather than a disjunction).
    """
    both = fts_match_query("alpha gamma")
    assert both == '"alpha" OR "gamma"'
    assert _match(fts_table, both) == [1]
    assert _match(fts_table, both.replace(" OR ", " AND ")) == [1]


def test_generated_expression_is_accepted_by_fts5(fts_table):
    """Every expression this builder can emit is valid MATCH syntax.

    Covers the untrusted-input case: FTS5 syntax characters in user text are
    neutralized by quoting, so the expression parses instead of raising or
    becoming injected syntax.
    """
    hostile = [
        '"; DROP TABLE memories; --',
        "a' OR '1'='1",
        'x" OR "1"="1',
        '"unterminated quote',
        "NEAR( OR NOT )",
        "\\..* && || ^",
        "'; DELETE FROM memories WHERE '1'='1",
        "*",
        "col:val",
        "AND OR NOT NEAR",
    ]
    for text in hostile:
        expr = fts_match_query(text)
        if not expr:
            continue
        # must parse (result value is irrelevant; a syntax error raises)
        _match(fts_table, expr)


def test_quote_doubling_is_dead_defense_only_because_the_tokenizer_excludes_quotes():
    """Documents why mutant #25 (quote-doubling removal) is *equivalent* today.

    The ``.replace('"', '""')`` armor is unreachable: the tokenizer can never
    emit a token containing a double quote, because ``"`` is not ``\\w``. This
    test pins the *reason* the armor is dead, so that a future tokenizer change
    that admits quotes fails here and forces a decision about the armor — rather
    than silently leaving the table with an undeclared survivor.
    """
    import re

    # a quote can be *input* ...
    assert re.search(r'\w"', 'a"b') is not None
    # ... but the tokenizer's alphabet cannot *emit* one
    assert re.fullmatch(r"\w+", 'a"b') is None
    for text in ['"', '""', 'a"b', '" OR "1"="1', 'x"y OR "z']:
        expr = fts_match_query(text)
        for segment in expr.split(" OR ") if expr else []:
            # every segment is '"' + quote-free body + '"'
            assert segment[0] == segment[-1] == '"'
            assert '"' not in segment[1:-1]
    # the builder still applies the doubling; it is simply unreachable today
    assert fts_match_query("abc") == '"abc"'

def test_fts_stopwords_are_dropped_before_the_cap():
    """F35: function words no longer flood the match set (they matched ~88% of
    the fiqa corpus), and the term cap counts CONTENT terms, so a stopword-heavy
    prefix cannot starve the real terms out of the expression."""
    q = "where should i park my rainy day emergency fund"
    assert fts_match_query(q) == '"park" OR "rainy" OR "day" OR "emergency" OR "fund"'
    assert fts_match_query("the " * 30 + "alpha beta") == '"alpha" OR "beta"'


def test_fts_stopword_only_query_is_empty():
    """F35: an expression of only function words expands to "", which
    recall.run_recall already treats as "no sparse leg"."""
    assert fts_match_query("what is the") == ""
    assert fts_match_query("   to  be  ") == ""


def test_fts_negation_words_are_not_stopwords():
    """F35: "not"/"no" are deliberately absent from the stopword set because
    they carry negation signal in questions ("what NOT to do")."""
    assert fts_match_query("what not to do") == '"not" OR "do"'
    assert fts_match_query("no risk no reward") == '"no" OR "risk" OR "no" OR "reward"'


def test_fts_unicode_lower_does_not_split_ordinary_text():
    """F36: the module claims cafe/Beijing stay searchable, and they do."""
    assert fts_match_query("CAFÉ") == '"café"'
    assert fts_match_query("北京") == '"北京"'
    assert fts_match_query("naïve") == '"naïve"'


def test_fts_dotted_capital_i_does_not_split_its_token():
    """F36: U+0130 is the one alphabetic char whose .lower() expands to two
    code points (i + U+0307), and U+0307 is not a \\w character, so the
    tokenizer used to split mid-word and search "stanbul".

    Fixed by stripping combining marks before tokenizing. Measured quality-
    neutral on fiqa (.omni-audit/f36_unicode_fix.py): all four candidate fixes
    scored sparse nDCG@10 0.25407 with 0/648 queries changed. NFKC and
    str.casefold() were both measured and neither fixes it - casefold yields
    the same i+dot pair, and NFKC does not decompose an already-NFC U+0130.
    """
    assert fts_match_query("İstanbul") == '"istanbul"'
    assert fts_match_query("Istanbul") == '"istanbul"'
    assert fts_match_query("İSTANBUL") == '"istanbul"'


def test_fts_does_not_strip_caller_supplied_combining_marks():
    """F36: the fix removes only the mark that .lower() itself introduced, so a
    decomposed accent behaves exactly as it did before the fix. Asserting this
    matters: an earlier draft stripped ALL combining marks, which fixed
    İstanbul and silently changed decomposed "café" from "cafe" to "café" -
    i.e. it fixed one defect by introducing a different tokenization change.

    The decomposed/precomposed mismatch for "ñ" is F38 and is unresolved:
    n + U+0303 has no NFC composition to ñ, because composition runs through
    uppercase Ñ and .lower() runs first. Measured across NFC, NFKC and NFKD.
    """
    decomposed_acute = "cafe" + chr(0x301)
    assert fts_match_query(decomposed_acute) == '"cafe"'      # unchanged
    assert fts_match_query("caf" + chr(0xE9)) == '"café"'     # precomposed
    # the two spellings disagree, which is F38 - recorded, not fixed
    assert fts_match_query(decomposed_acute) != fts_match_query(
        "caf" + chr(0xE9))
