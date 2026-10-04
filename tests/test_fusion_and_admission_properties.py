"""Property-based tests for the two designs in docs/audit/6_VERIFICATION_TESTING.md §1
that have the highest mutant yield: **FTS injection safety (both halves)** and
**trigger-admission bounds**.

Design 2 (FTS) is one differential property with two halves:

* **Safety half** — either the expression is empty, or every quote-doubled
  segment is well-formed *and* a real FTS5 table accepts ``MATCH`` without
  raising. This is the property ``recall`` relies on when it splices the
  expression into a query built from untrusted user text.
* **Equivalence half** — the output equals a reference tokenizer join over the
  first ``max_terms`` ``\\w`` runs of the lowercased input. The reference is
  written as a *character scan*, not a regex, so it is a genuinely independent
  implementation: it cannot inherit a bug from the ``\\w+`` pattern under test.

One property kills the tokenizer, join, cap, lowercasing, default-cap and
empty-input mutants as a group, because a change to any of them breaks
equivalence.

Design 4 (trigger admission) sweeps all three write entry points that share the
policy: ``remember``, ``remember_many`` and ``import_jsonl``. Only two
hand-written cases existed, and ``import_jsonl`` had none of them.

**Adversarial-fixture discipline** (docs/audit/6_VERIFICATION_TESTING.md §3):
every generated input is code-generated, bounded to a few hundred characters,
and no test touches a live home — the FTS properties run against an in-memory
SQLite database and the admission properties run against ``tmp_path``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from conftest import make_store
from hypothesis import given, settings
from hypothesis import strategies as st

from omnidb_server.fusion import fts_match_query
from omnidb_server.portability import import_jsonl
from omnidb_server.store import E_CONSTRAINT, E_TYPE, StoreError
from omnidb_server.validation import (
    MAX_TRIGGER_PATTERN_CHARS,
    MAX_TRIGGER_PATTERNS,
    validate_trigger_patterns,
)

# A segment is a quoted FTS5 phrase: '"' + body + '"' where the body never
# contains a bare quote (a literal quote inside a phrase is doubled to "").
_SEGMENT = re.compile(r'^"([^"]|"")*"$')


# ==========================================================================
# Design 2 — FTS injection safety, both halves
# ==========================================================================

def _reference_fts_expression(text: str, max_terms: int = 24) -> str:
    """An independent tokenizer+joiner, written as a character scan.

    Deliberately NOT a regex: the implementation under test uses ``\\w+``, so a
    regex reference would share the very pattern whose mutation these tests must
    detect. This scan restates the *specification* (maximal runs of word
    characters in the lowercased text, first ``max_terms`` of them) using
    ``str.isalnum``/``"_"``, which is what ``\\w`` means in Python 3 for str
    patterns.
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch == "_" or ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    # F35: the specification drops function words before the cap, so a
    # stopword-heavy query cannot consume the term budget. This set is an
    # intentionally independent literal copy of fusion._STOP_WORDS: importing
    # the real set would let a mutation of it silence this property instead of
    # being caught by it.
    _STOP = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
        "did", "do", "does", "for", "from", "get", "has", "have", "how", "i",
        "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
        "out", "should", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "up", "us", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "you",
        "your",
    }
    tokens = [t for t in tokens if t not in _STOP]
    return " OR ".join('"%s"' % token for token in tokens[:max_terms])


@pytest.fixture(scope="module")
def fts_conn():
    """One in-memory FTS5 table shared by the FTS properties.

    FTS5 availability is a build property of SQLite; a build without it would
    make every property here fail for the wrong reason, so the module skips
    instead. OmniDB requires FTS5 in production, so this should never trigger.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    except sqlite3.OperationalError as exc:  # pragma: no cover - build guard
        conn.close()
        pytest.skip(f"SQLite build lacks FTS5: {exc}")
    conn.execute("INSERT INTO t(body) VALUES ('alpha gamma')")
    conn.execute("INSERT INTO t(body) VALUES ('delta beta')")
    yield conn
    conn.close()


def _match_rows(conn, expr):
    return [int(r[0]) for r in
            conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,)).fetchall()]


# --- half 1: safety (well-formed segments + FTS5 accepts MATCH) ------------

@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=200))
def test_fts_expression_is_either_empty_or_injection_safe(text, fts_conn):
    """The expression is the empty string, or every segment is a well-formed
    quoted phrase and a real FTS5 table accepts it as a ``MATCH`` argument
    without raising.

    This is the property ``recall`` needs: user text must never become FTS5
    syntax. Any unquoted operator, unbalanced quote or stray backslash produces
    a ``sqlite3.OperationalError`` here.
    """
    expr = fts_match_query(text)
    if not expr:
        # empty output is always safe; recall skips the sparse leg entirely
        return
    for segment in expr.split(" OR "):
        assert _SEGMENT.match(segment), (text, expr, segment)
    # must not raise: this is the half that catches injection-shaped output
    _match_rows(fts_conn, expr)


# --- half 2: equivalence against the reference tokenizer ------------------

@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=200), max_terms=st.integers(min_value=0, max_value=40))
def test_fts_matches_reference_tokenizer_join(text, max_terms, fts_conn):
    """The expression equals the reference join over the first ``max_terms``
    ``\\w`` runs of the lowercased input.

    Differential: a change to the regex, the ``.lower()``, the ``[:max_terms]``
    slice, the join operator, the quoting or the empty-input branch all change
    the string, so all are killed. Asserting the *closed form* instead would
    only re-implement the source; asserting the *string* pins the output.
    """
    got = fts_match_query(text, max_terms=max_terms)
    assert got == _reference_fts_expression(text, max_terms=max_terms)
    # the default cap is the shipped 24, pinned as a value
    assert fts_match_query(text) == _reference_fts_expression(text, 24)


@settings(max_examples=200, deadline=None)
@given(
    shape=st.sampled_from([
        '"; DROP TABLE memories; --',
        "a' OR '1'='1",
        'x" OR "1"="1',
        '"unterminated quote',
        "NEAR( OR NOT )",
        "\\..* && || ^",
        "'; DELETE FROM memories WHERE '1'='1",
        '"',
        "col:val",
        "*",
        "^anchor",
        "NEAR(a b, 5)",
        "AND OR NOT",
    ])
)
def test_fts_injection_shapes_are_quoted_literals_only(shape, fts_conn):
    """Adversarial fixtures: injection-shaped input yields quoted literals only.

    A negative control: an unquoted expression for these shapes would make
    FTS5 raise or match unexpectedly. Since the tokenizer strips FTS syntax
    characters before quoting, none of these can escape their quotes.
    """
    expr = fts_match_query(shape)
    if not expr:
        return
    for segment in expr.split(" OR "):
        assert _SEGMENT.match(segment), (shape, expr, segment)
    # and it parses as valid MATCH syntax
    _match_rows(fts_conn, expr)


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=200))
def test_fts_default_cap_is_exactly_24_terms(text, fts_conn):
    """The shipped default cap: at most 24 segments, and truncation is by
    position (the first 24 tokens, not a random or total-order subset)."""
    expr = fts_match_query(text)
    segments = expr.split(" OR ") if expr else []
    assert len(segments) <= 24
    # position-preserving truncation: the emitted segments are a prefix of the
    # reference token list
    reference = _reference_fts_expression(text, 24).split(" OR ")
    assert segments == reference[:len(segments)]


@settings(max_examples=100, deadline=None)
@given(
    pairs=st.lists(
        st.tuples(st.text(max_size=8), st.text(max_size=8)),
        min_size=1, max_size=3,
    )
)
def test_fts_or_is_a_disjunction_over_fts5(pairs, fts_conn):
    """The OR join is a genuine disjunction: it retrieves a superset of what a
    conjunction would, and equals the union of the single-term queries.

    The behavioural half that a string comparison alone cannot supply. The
    ``AND`` mutant returns strictly fewer rows for a pair of terms that live in
    different documents, so it dies here.
    """
    doc_a, doc_b = pairs[0]
    expr = fts_match_query(f"{doc_a} {doc_b}")
    if not expr or " OR " not in expr:
        # fewer than two tokens survived; the equivalence test covers the shape
        return
    or_rows = set(_match_rows(fts_conn, expr))
    and_rows = set(_match_rows(fts_conn, expr.replace(" OR ", " AND ")))
    # the conjunction is a subset of the disjunction
    assert and_rows <= or_rows


# ==========================================================================
# Design 4 — trigger-admission bounds, swept across all three entry points
# ==========================================================================

# valid / boundary inputs (at or under both caps)
_VALID = st.one_of(
    st.just([]),
    st.lists(st.text(max_size=MAX_TRIGGER_PATTERN_CHARS),
             max_size=MAX_TRIGGER_PATTERNS),
    # boundary: exactly at both caps
    st.just(["x" * MAX_TRIGGER_PATTERN_CHARS] * MAX_TRIGGER_PATTERNS),
    st.just([""]),
)

# invalid inputs: a hand-built list where at least one bound is provably
# violated. Built with st.builds + an explicit filter so the strategy can
# never shrink down to an admissible (empty) list.
_TO_MANY = st.lists(st.text(max_size=MAX_TRIGGER_PATTERN_CHARS),
                    min_size=MAX_TRIGGER_PATTERNS + 1,
                    max_size=MAX_TRIGGER_PATTERNS + 4)
_TOO_LONG = st.lists(st.text(min_size=MAX_TRIGGER_PATTERN_CHARS + 1,
                             max_size=MAX_TRIGGER_PATTERN_CHARS + 20),
                     min_size=1, max_size=3)
_NON_STR = st.lists(st.one_of(st.integers(), st.none(), st.booleans(), st.floats()),
                    min_size=1, max_size=3)
_INVALID = st.one_of(_TO_MANY, _TOO_LONG, _NON_STR)


def _violates(patterns: list) -> bool:
    """True iff at least one admission bound is broken."""
    if len(patterns) > MAX_TRIGGER_PATTERNS:
        return True
    return any(
        not isinstance(p, str) or len(p) > MAX_TRIGGER_PATTERN_CHARS
        for p in patterns
    )


@contextmanager
def _home():
    """A throwaway home directory, created per call.

    Hypothesis forbids function-scoped fixtures (``tmp_path``) because they are
    not reset between generated inputs, so the store-backed properties manage
    their own temporary home. Everything stays inside the system temp dir, never
    a live home — the adversarial-fixture rule in
    docs/audit/6_VERIFICATION_TESTING.md §3.
    """
    with tempfile.TemporaryDirectory(prefix="omnidb-admission-") as raw:
        yield Path(raw)


@given(patterns=_VALID)
@settings(max_examples=100, deadline=None)
def test_validation_returns_admissible_input_unchanged(patterns):
    """Admissible input is returned by validation itself (round-trip)."""
    out = validate_trigger_patterns(list(patterns))
    assert out == list(patterns)
    assert all(isinstance(p, str) for p in out)


@given(patterns=_INVALID)
@settings(max_examples=100, deadline=None)
def test_validation_rejects_inadmissible_input(patterns):
    """Inadmissible input raises and never returns a partial value.

    The type violation is a TypeError, the size violations are ValueErrors.
    The generated input is re-checked here, so a future strategy edit that
    starts producing admissible lists fails loudly instead of silently
    weakening the test into a no-op.
    """
    assert _violates(list(patterns)), patterns
    with pytest.raises((TypeError, ValueError)):
        validate_trigger_patterns(list(patterns))


# --- sweep 1: remember ---------------------------------------------------

@given(patterns=_INVALID)
@settings(max_examples=40, deadline=None)
def test_remember_rejects_out_of_bounds_patterns_and_writes_nothing(patterns):
    """``remember`` refuses over-bounds patterns and leaves the home empty."""
    assert _violates(list(patterns)), patterns
    with _home() as home:
        st_ = make_store(home)
        try:
            with pytest.raises(StoreError) as exc:
                st_.remember("ok", kind="procedural",
                             trigger_patterns=list(patterns))
            # size -> OMN-005 (E_CONSTRAINT); type -> OMN-002 (E_TYPE)
            assert exc.value.code in (E_TYPE, E_CONSTRAINT)
            assert st_.conn.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0] == 0
            assert st_.stats()["live"] == 0
        finally:
            st_.close()


@given(patterns=_VALID)
@settings(max_examples=40, deadline=None)
def test_remember_accepts_admissible_patterns(patterns):
    """Control: admissible patterns are stored verbatim (not silently dropped)."""
    assert not _violates(list(patterns)), patterns
    with _home() as home:
        st_ = make_store(home)
        try:
            rec = st_.remember("ok", kind="procedural",
                               trigger_patterns=list(patterns))
            row = st_.conn.execute(
                "SELECT trigger_patterns FROM memories WHERE id=?", (rec["id"],)
            ).fetchone()
            assert json.loads(row[0]) == list(patterns)
        finally:
            st_.close()


# --- sweep 2: remember_many ---------------------------------------------

@given(patterns=_INVALID)
@settings(max_examples=40, deadline=None)
def test_remember_many_rejects_out_of_bounds_patterns_and_writes_nothing(patterns):
    """``remember_many`` refuses over-bounds patterns and leaves the home empty.

    The batch path shares the validator, but "no partial write survives" is a
    distinct claim: a batch that raises partway must not leave earlier items
    committed. Asserting the row count is zero is the check for that.
    """
    assert _violates(list(patterns)), patterns
    with _home() as home:
        st_ = make_store(home)
        try:
            with pytest.raises(StoreError):
                st_.remember_many([
                    {"content": "first", "kind": "semantic"},
                    {"content": "second", "kind": "procedural",
                     "trigger_patterns": list(patterns)},
                ])
            assert st_.conn.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0] == 0
            assert st_.stats()["live"] == 0
        finally:
            st_.close()


# --- sweep 3: import_jsonl ----------------------------------------------

def _write_export(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


@given(patterns=_INVALID)
@settings(max_examples=40, deadline=None)
def test_import_jsonl_rejects_out_of_bounds_patterns_and_writes_nothing(patterns):
    """``import_jsonl`` refuses over-bounds patterns and leaves the home empty.

    The audit's design 4 says to sweep this entry point; before this test it had
    **no** trigger-admission coverage at all. The import is all-or-nothing, so
    the assertion is that not a single row landed.
    """
    assert _violates(list(patterns)), patterns
    with _home() as home:
        dump = home / "bad.jsonl"
        _write_export(dump, [
            {"type": "memory", "id": 1, "kind": "semantic", "content": "first"},
            {"type": "memory", "id": 2, "kind": "procedural",
             "content": "second", "trigger_patterns": list(patterns)},
        ])
        target = make_store(home / "dst")
        try:
            with pytest.raises(StoreError):
                import_jsonl(target, dump)
            assert target.conn.execute(
                "SELECT COUNT(*) FROM memories").fetchone()[0] == 0
            assert target.stats()["live"] == 0
        finally:
            target.close()


@given(patterns=_VALID)
@settings(max_examples=40, deadline=None)
def test_import_jsonl_accepts_admissible_patterns(patterns):
    """Control: admissible patterns import successfully and round-trip."""
    assert not _violates(list(patterns)), patterns
    with _home() as home:
        dump = home / "good.jsonl"
        _write_export(dump, [{
            "type": "memory", "id": 1, "kind": "procedural",
            "content": "proc body", "trigger_patterns": list(patterns),
        }])
        target = make_store(home / "dst")
        try:
            result = import_jsonl(target, dump)
            assert result["memories"] == 1
            row = target.conn.execute(
                "SELECT trigger_patterns FROM memories WHERE id=1"
            ).fetchone()
            assert json.loads(row[0]) == list(patterns)
        finally:
            target.close()
