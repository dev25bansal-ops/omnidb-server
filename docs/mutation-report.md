# Mutation Analysis — Pure Ranking Modules (fusion.py + scoring.py)

Analysis §6.3 target: the two pure ranking modules that carry ALL ranking
behavior in ~70 lines. Goal: measure how well the scoped test suite detects
deliberately broken ranking logic.

## Tool used

**mutmut 2.5.1 was attempted first and failed on Python 3.14.4** (repo runs
3.14). Two distinct failures were reproduced:

1. `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f389'` —
   mutmut's banner prints an emoji and crashes under Windows cp1252 stdout.
   Bypassed with `PYTHONIOENCODING=utf-8`.
2. After bypassing the banner: `TypeError: cannot pickle 'itertools.count'
   object` from inside mutmut's trampoline/deepcopy machinery. Fatal; no
   workaround without editing files we don't own.

Command attempted:

```
python -m pip install --quiet mutmut   # -> installs 2.5.1
PYTHONIOENCODING=utf-8 python -m mutmut run \
  --paths-to-mutate omnidb_server/fusion.py --paths-to-mutate omnidb_server/scoring.py \
  --tests-dir tests \
  --runner "python -m pytest tests/test_scoring_properties.py tests/test_memory.py -x -q"
```

(The fallback runner omits `tests/test_reranker.py`: no such file exists in
this tree; the two files above are the ones that exercise these modules.)

**Fallback executed:** `scripts/mutation_check.py` — self-contained script
applying a fixed table of 30 hand-written mutations (14 scoring.py,
16 fusion.py) one at a time, running the scoped suite per mutant with a 60 s
subprocess timeout, recording KILLED/SURVIVED, and restoring originals in a
`finally` block so no mutated file can survive an interrupted run.

> **Status note (2026-10-07).** This report records the audit's *starting*
> state — 33% with 20 survivors — and the recommendations that followed it.
> Those recommendations were adopted: the gate now runs 38 mutants with 37
> killed and 1 genuinely equivalent, and it exits non-zero when the score falls
> below `--min-score` or when any anchor stops matching the source (a stale
> anchor used to look identical to an unkilled mutant). One declaration since
> withdrawn is worth noting: mutant 25 here (`quote-doubling`) was accepted as
> equivalent; it remains equivalent and is now numbered 33. A *different*
> mutant — the finalised-check removal in `graph_leg.py` — was declared
> equivalent on the same style of argument and proved observable under the
> hard pop budget, and is now killed by a pinned counterexample. See
> `scripts/mutation_check.py` for the current table and `EQUIVALENT` rationale.

## Results

Baseline: `12 passed` (tests/test_scoring_properties.py +
tests/test_memory.py, pristine sources).

**30 mutants: 10 killed, 20 survived — mutation score 33%.**

### Killed (10)

| # | Module | Mutation | Killed by |
|---|--------|----------|-----------|
| 01 | scoring | `exp(-age/hl)` → `exp(age/hl)` (decay sign flip) | `test_recency_decays_with_age` + boundedness property |
| 02 | scoring | drop `max(age_ms, 0)` clamp | `test_clock_skew_clamped` / totality property |
| 05 | scoring | `log1p` → `log` (access_count=0 → domain error) | finiteness in `test_total_and_recency_bounded` |
| 09 | scoring | return drops recency component | `test_remember_recall_roundtrip` field presence |
| 10 | scoring | return drops frequency component | same |
| 11 | scoring | score omits recency term | `test_recency_decays_with_age` (strict score gap collapses to tie) |
| 13 | scoring | score omits rrf term | `test_fusion_orders_relevant_above_irrelevant` |
| 17 | fusion | rrf denominator → `position+1` (rank weights inverted) | ordering assertions via `store.recall` |
| 19 | fusion | rrf scores all 0.0 (signal destroyed) | recall ordering |
| 23 | fusion | rrf returns empty dict | `test_remember_recall_roundtrip` |

### Survived (20) — each survivor is a missing assertion

#### A. No direct unit tests for `fusion.rrf` (6 survivors: 15, 16, 18, 20, 21, 22)

Nothing imports `rrf` directly; it is reached only through `store.recall`,
where dense-leg hits mask pure sparse-side rank arithmetic.

| # | Mutation | Why it survives |
|---|----------|-----------------|
| 15 | denominator `k+p+1` → `k+p` | Uniform shift of all RRF weights preserves relative order; no numeric assertion on fused values exists |
| 16 | `(k+p)^2` quadratic damping | Strictly decreasing transform of rank — order-preserving, so order-only assertions pass |
| 18 | accumulation → overwrite (later list wins) | Cross-list duplicate ids are never asserted numerically through recall |
| 20 | enumerate offset shifted (`+2`) | Monotone shift again |
| 21 | default `k=60` → `k=1` | Store always passes `cfg.rrf_k` explicitly, so the default arg is dead in production paths and untested |
| 22 | second ranking list ignored (`rankings[:1]`) | Dense leg alone still orders the fixtures correctly |

Killing tests (proposed new `tests/test_fusion_units.py`):

```python
from omnidb_server.fusion import rrf

def test_rrf_accumulates_across_lists():
    assert rrf([[7], []], k=60) == {7: 1/61}
    assert abs(rrf([[7], [7]], k=60)[7] - 2/61) < 1e-12   # kills 18

def test_rrf_exact_rank_weights():
    out = rrf([[10, 20, 30]], k=60)
    assert abs(out[10] - 1/61) < 1e-12                    # kills 15, 16, 20
    assert abs(out[20] - 1/62) < 1e-12
    assert out[10] > out[20] > out[30]

def test_rrf_default_k_is_60():
    assert rrf([[1]]) == {1: 1/61}                        # kills 21

def test_rrf_uses_every_list():
    out = rrf([[1], [2]], k=60)
    assert set(out) == {1, 2}                             # kills 22
```

#### B. No direct unit tests for `fts_match_query` (7 survivors: 24–30)

The FTS query builder is exercised only indirectly through SQLite round-trips
whose fixtures contain every query token, making conjunction/disjunction and
tokenizer behavior unobservable.

| # | Mutation | Why it survives |
|---|----------|-----------------|
| 24 | regex `\w+` → `\w` | Single-char tokens still match inside fixture words; bm25 still ranks the intended doc top; assertions check membership/order, not the MATCH expression itself |
| 25 | quote-doubling removed | **Equivalent mutant for all currently reachable inputs**: `\w+` can never emit a token containing `"`, so `.replace('"', '""')` is dead defense today (verified: regex probe emits no quotes). Only a future tokenizer change could expose it |
| 26 | OR join → AND join | Fixture docs contain all query tokens, so conjunctive matching returns the same top doc (empirically verified on FTS5: AND matches a doc holding both tokens; OR additionally matches subset docs) |
| 27 | `[:max_terms]` slice removed | No fixture query exceeds 24 tokens |
| 28 | `.lower()` dropped | Fixture queries are already lowercase; FTS5 unicode61 folds ASCII case during tokenization anyway |
| 29 | default `max_terms=24` → 1 | `recall()` does rely on the default, but fixture queries are 2–4 tokens, indistinguishable from capped-at-1 behavior only when the first token alone still hits |
| 30 | empty text yields `['']` → MATCH `'""'` | Empty-query branch never tested; FTS5 treats `'""'` as a no-hit phrase (verified), so downstream behavior happens to match the empty-expression early-return in fixtures |

Killing tests:

```python
from omnidb_server.fusion import fts_match_query

def test_fts_tokens_are_whole_words():
    assert fts_match_query("abc def") == '"abc" OR "def"'      # kills 24

def test_fts_lowercases():
    assert fts_match_query("ALPHA") == '"alpha"'               # kills 28

def test_fts_caps_terms():
    expr = fts_match_query(" ".join(f"w{i}" for i in range(40)), max_terms=3)
    assert expr == '"w0" OR "w1" OR "w2"'                      # kills 27
    assert fts_match_query("a b c") == '"a" OR "b" OR "c"'     # kills 29 (default cap)

def test_or_join_widens_recall():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    conn.execute("INSERT INTO t VALUES ('alpha gamma')")
    hits = conn.execute("SELECT rowid FROM t WHERE t MATCH ?",
                        (fts_match_query("alpha beta"),)).fetchall()
    assert hits                                                # kills 26 (AND variant finds nothing)

def test_empty_text_gives_empty_expr():
    assert fts_match_query("!!! ...") == ""                    # kills 30
```

(#25 stays surviving by design — document as an accepted equivalent mutant.)

#### C. Weight defaults, floors, and clamps in scoring.py are unpinned (7 survivors: 03, 04, 06, 07, 08, 12, 14)

Property tests pin *shape* (monotonicity, boundedness, totality), not
*values*. Non-strict monotonicity is trivially satisfied when a term vanishes,
which is why term-dropping mutants slip through.

| # | Mutation | Why it survives |
|---|----------|-----------------|
| 03 | drop `max(half_life_ms, 1e-3)` clamp | Property strategy draws `half_life_ms ≥ 1e-3`, making the clamp unreachable; store config always positive |
| 04 | drop `max(access_count, 0)` clamp | DB schema keeps access_count non-negative; property strategy too |
| 06 | default `recency_weight=0.15` → `0.0` | Store passes `cfg.recency_weight` explicitly — default dead in prod, never asserted |
| 07 | default `freq_weight=0.05` → `0.0` | Same |
| 08 | `recency_weight * recency` → bare `recency` | Changes magnitudes but not directions; no absolute-value assertion anywhere |
| 12 | score omits frequency term | Property asserts non-decreasing in access_count — a constant-in-count score satisfies `more >= fewer - eps` vacuously; store-level frequency test's key assertion is conditional (`if counts[first] > counts[second]`) |
| 14 | half-life floor `1e-3` → `1e6` | Only reachable when caller passes `half_life_ms < 1e6`; store uses hours-scale ms far above the floor |

Killing tests (add to `tests/test_scoring_properties.py`):

```python
def test_exact_values_pin_weights_and_defaults():
    score, recency, frequency = final_score(rrf=1.0, age_ms=0.0, access_count=9,
                                            half_life_ms=1000.0)
    assert recency == 1.0 and frequency == math.log(10)
    assert abs(score - (1.0 + 0.15 + 0.05 * math.log(10))) < 1e-12   # kills 08

    d, _, _ = final_score(rrf=0.0, age_ms=0.0, access_count=9, half_life_ms=1e9)
    assert abs(d - (0.15 + 0.05 * math.log(10))) < 1e-12             # kills 06, 07

def test_frequency_term_strictly_contributes():
    s0, _, _ = final_score(rrf=1.0, age_ms=0.0, access_count=0, half_life_ms=1e6)
    s9, _, _ = final_score(rrf=1.0, age_ms=0.0, access_count=9, half_life_ms=1e6)
    assert s9 > s0 + 0.01                                            # kills 12

def test_half_life_floor_protects_zero():
    score, recency, _ = final_score(rrf=0.0, age_ms=1.0, access_count=0,
                                    half_life_ms=0.0)
    assert math.isfinite(score)
    assert recency == math.exp(-1.0 / 1e-3)                          # kills 03, 14

def test_negative_access_count_clamped():
    _, _, frequency = final_score(rrf=0.0, age_ms=0.0, access_count=-5,
                                  half_life_ms=1e6)
    assert frequency == 0.0                                          # kills 04
```

## Verdict

- The property suite is strong on *behavioral* mutations (sign flips, term
  omission from the tuple, hot-path clamp removal, signal destruction):
  those die 10-for-10.
- The blind spots are systematic, not random: **neither pure function has a
  single direct unit test** — everything flows through `store.recall`, whose
  fixtures are friendly (short lowercase queries, docs containing all query
  tokens, dense leg dominating). All 20 survivors fall into three named
  groups above.
- One accepted equivalent mutant (#25, quote-doubling): unreachable by
  construction today; keep as cheap armor or delete with the tokenizer's
  contract documented.
- Highest-value additions in order: exact-value pins on `final_score`
  (kills 6 survivors with two tests), `rrf` structural unit tests (kills 6),
  `fts_match_query` string-equality tests (kills 6). Full adoption moves the
  score from 33% to >95% (29/30, #25 excepted).

## Integrity verification

After the run, SHA-256 of both sources matched their pre-run backups
byte-for-byte. The crashed mutmut attempt left no mutated files either (only a
`.mutmut-cache`, which was removed):

```
fusion.py  5ae71f86d70ac6696297415cfda2aea8f1e4da7b25980a8d706fc98b70373ccb
scoring.py 5d8f6e7f72fb80ebca7cd5d1acac0840f6d0b7f1ca9ae5461797e45d0a61d2f9
```

No mutated file left behind. Scoped suite green on pristine sources:
**12 passed** (tests/test_scoring_properties.py + tests/test_memory.py).
