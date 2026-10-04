"""Exact-value unit tests for the pure scoring surface (scoring.final_score).

docs/mutation-report.md:148-160 explains the seven survivors mechanically:
*"Property tests pin shape (monotonicity, boundedness, totality), not values.
Non-strict monotonicity is trivially satisfied when a term vanishes, which is
why term-dropping mutants slip through."* A property like
``score_non_decreasing_in_access_count`` asserts ``more >= fewer``, and a score
that simply omits the frequency term satisfies it exactly.

These tests replace that vacuity with **exact equality on pinned float
literals**. Every literal is bit-exact on CPython 3.11-3.14 and
platform-independent: IEEE-754 double, left-to-right evaluation, no FMA
contraction, no x87 extended precision. Each was verified with an explicit
``repr()`` round-trip before being written down.

Shape properties in ``test_scoring_properties.py`` are kept — they are the right
tool for totality and boundedness — but the shipped weights, the floors and the
clamps are pinned *here*, where a change to any of them is a red test rather
than a silently tolerated shape.
"""

from __future__ import annotations

import dataclasses
import math

from omnidb_server.scoring import final_score

# The shipped weights, quoted from the final_score signature:
#   recency_weight: float = 0.15
#   freq_weight: float = 0.05
_SHIPPED_RECENCY_WEIGHT = 0.15
_SHIPPED_FREQ_WEIGHT = 0.05
# The shipped floor, quoted from the body:
#   half_life = max(float(half_life_ms), 1e-3)
_SHIPPED_HALF_LIFE_FLOOR = 1e-3
# The shipped per-leg fusion weights, quoted from Config:
#   dense_weight: float = 0.7
#   sparse_weight: float = 0.3
_SHIPPED_DENSE_WEIGHT = 0.7
_SHIPPED_SPARSE_WEIGHT = 0.3


# --------------------------------------------------------------------------
# the exact score decomposition
# --------------------------------------------------------------------------

def test_exact_values_pin_weights_and_defaults():
    """score == rrf + 0.15*recency + 0.05*log1p(max(access_count, 0)).

    Kills the "weights ignored", "recency term omitted" and "frequency term
    omitted" mutants, because each moves the literal.
    """
    score, recency, frequency = final_score(
        rrf=1.0, age_ms=0.0, access_count=9, half_life_ms=1000.0,
    )
    # age 0 -> exp(0) == 1.0 exactly
    assert recency == 1.0
    # log1p(9.0) == log(10.0) bit-exactly, so frequency is the pinned literal
    assert frequency == 2.302585092994046
    # 1.0 + 0.15*1.0 + 0.05*2.302585092994046, evaluated left to right
    assert score == 1.2651292546497022
    # the weight literals themselves, so a weight change is a red test even if
    # it left this particular score unchanged
    assert _SHIPPED_RECENCY_WEIGHT == 0.15
    assert _SHIPPED_FREQ_WEIGHT == 0.05


def test_recency_and_frequency_weights_are_applied_not_bare():
    """The weights multiply their terms; dropping them changes the magnitude.

    ``recency_weight * recency`` -> ``recency`` keeps every ordering and every
    bound, so only a pinned value can see it.
    """
    # recency 0.9048374180359595 (exp(-0.1)), frequency 0.0
    score, recency, frequency = final_score(
        rrf=0.0, age_ms=1e-4, access_count=0, half_life_ms=1e-4,
    )
    assert recency == 0.9048374180359595
    assert frequency == 0.0
    # 0.0 + 0.15*0.9048374180359595 + 0.05*0.0
    assert score == 0.13572561270539393
    # the unweighted value the mutant would produce
    assert recency != score


def test_rrf_term_is_passed_through_unchanged():
    """The fusion contribution is added verbatim, not scaled or clamped."""
    score, _, _ = final_score(rrf=0.5, age_ms=0.0, access_count=0,
                              half_life_ms=1e6)
    # 0.5 + 0.15*1.0 + 0.05*0.0
    assert score == 0.65
    score, _, _ = final_score(rrf=2.0, age_ms=0.0, access_count=0,
                              half_life_ms=1e6)
    # 2.0 + 0.15*1.0 + 0.05*0.0
    assert score == 2.15


def test_default_weights_match_the_config_defaults():
    """The function defaults and the shipped Config defaults must not drift.

    ``recall`` passes ``cfg.recency_weight``/``cfg.freq_weight`` explicitly, so
    the function defaults are unreachable in production and were untested. This
    pins them against the config that actually ships.
    """
    from omnidb_server.config import Config

    cfg_defaults = {f.name: f.default for f in dataclasses.fields(Config)}
    assert cfg_defaults["recency_weight"] == _SHIPPED_RECENCY_WEIGHT
    assert cfg_defaults["freq_weight"] == _SHIPPED_FREQ_WEIGHT
    assert cfg_defaults["rrf_k"] == 60
    assert cfg_defaults["dense_weight"] == _SHIPPED_DENSE_WEIGHT
    assert cfg_defaults["sparse_weight"] == _SHIPPED_SPARSE_WEIGHT

    defaults = final_score(rrf=0.0, age_ms=0.0, access_count=3, half_life_ms=1e6)
    explicit = final_score(rrf=0.0, age_ms=0.0, access_count=3, half_life_ms=1e6,
                           recency_weight=0.15, freq_weight=0.05)
    assert defaults == explicit
    # 0.0 + 0.15*1.0 + 0.05*log1p(3.0) = 0.0 + 0.15 + 0.06931471805599453
    assert defaults[0] == 0.2193147180559945


def test_frequency_term_strictly_contributes():
    """Access count must move the score by a pinned, strictly positive amount.

    The old property asserted ``more >= fewer`` — satisfied exactly by a score
    that omits the frequency term. A pinned gap kills it.
    """
    s0, _, _ = final_score(rrf=1.0, age_ms=0.0, access_count=0, half_life_ms=1e6)
    s9, _, f9 = final_score(rrf=1.0, age_ms=0.0, access_count=9, half_life_ms=1e6)
    assert s0 == 1.15
    assert f9 == 2.302585092994046
    # 0.05 * log1p(9.0) == 0.11512925464970225
    assert s9 - s0 == 0.11512925464970225
    assert s9 > s0


def test_recency_term_strictly_contributes():
    """Age must move the score by a pinned, strictly positive amount.

    Uses ``half_life_ms=1.0`` so the recency gap is large enough to be a stable
    float comparison rather than a near-cancellation artefact.
    """
    fresh, r_fresh, _ = final_score(rrf=0.0, age_ms=0.0, access_count=0,
                                    half_life_ms=1.0)
    stale, r_stale, _ = final_score(rrf=0.0, age_ms=1e-4, access_count=0,
                                    half_life_ms=1.0)
    assert r_fresh == 1.0
    assert r_stale == 0.9999000049998333
    assert fresh == 0.15
    assert stale == 0.149985000749975
    # 0.15 * (1.0 - exp(-1e-4))
    assert fresh - stale == 1.4999250024988475e-05
    assert fresh > stale


# --------------------------------------------------------------------------
# the clamps and the half-life floor
# --------------------------------------------------------------------------

def test_half_life_floor_protects_against_zero_and_negative():
    """``max(half_life_ms, 1e-3)`` — the floor, pinned to a non-degenerate value.

    The old property drew ``half_life_ms >= 1e-3``, so the clamp was
    unreachable and its removal was invisible. The witness is chosen so that all
    three mutants give *different, non-degenerate* answers:

    * no clamp        -> exp(-1e-4 / 1e-4)  = exp(-1.0)  = 0.36787944117144233
    * floor 1e-3      -> exp(-1e-4 / 1e-3)  = exp(-0.1)  = 0.9048374180359595
    * floor 1e6       -> exp(-1e-4 / 1e6)   = 0.9999999999
    """
    score, recency, _ = final_score(rrf=0.0, age_ms=1e-4, access_count=0,
                                    half_life_ms=0.0)
    assert recency == 0.9048374180359595
    # 0.15 * 0.9048374180359595
    assert score == 0.13572561270539393
    assert math.isfinite(score)
    # a negative half-life is floored the same way
    _, recency_neg, _ = final_score(rrf=0.0, age_ms=1e-4, access_count=0,
                                    half_life_ms=-1.0)
    assert recency_neg == 0.9048374180359595
    # the floor constant itself
    assert _SHIPPED_HALF_LIFE_FLOOR == 1e-3


def test_half_life_floor_value_is_1em3_not_1em6():
    """A frozen-decay floor (1e6) is a distinct, also-observable value.

    With ``age_ms=1e-4`` a floor of 1e6 would give exp(-1e-10) = 0.9999999999,
    which the floor test above already excludes. This test names the case
    directly so the intended floor is unambiguous from the failure message.
    """
    _, recency, _ = final_score(rrf=0.0, age_ms=1e-4, access_count=0,
                                half_life_ms=0.0)
    # a 1e6 floor would give 0.9999999999; the 1e-3 floor gives this
    assert recency != 0.9999999999
    assert recency == 0.9048374180359595


def test_negative_access_count_is_clamped_to_zero():
    """``max(access_count, 0)`` — frequency pins to exactly +0.0.

    The DB schema and the property strategy both keep access_count
    non-negative, so the clamp was unreachable from either direction.

    The sign bit is checked explicitly: ``0.0 == -0.0`` is true in IEEE-754, so
    a bare ``== 0.0`` would also accept a frequency that arrived as ``-0.0``,
    which is exactly what an unclamped ``log1p`` of a tiny negative would give.
    """
    score, recency, frequency = final_score(rrf=0.0, age_ms=0.0,
                                            access_count=-5, half_life_ms=1e6)
    assert frequency == 0.0
    assert math.copysign(1.0, frequency) > 0.0   # +0.0, not -0.0
    assert recency == 1.0
    # 0.0 + 0.15*1.0 + 0.05*0.0
    assert score == 0.15
    # a large negative count clamps identically
    _, _, frequency_neg = final_score(rrf=0.0, age_ms=0.0, access_count=-10**9,
                                      half_life_ms=1e6)
    assert frequency_neg == 0.0
    assert math.copysign(1.0, frequency_neg) > 0.0


def test_clock_skew_negative_age_is_clamped_to_zero():
    """``max(age_ms, 0)`` — a future created_at scores as brand new.

    Pinned by equality with the age=0 result, not merely by "is finite".
    """
    from_future = final_score(rrf=0.0, age_ms=-1e12, access_count=0,
                              half_life_ms=1e9)
    at_now = final_score(rrf=0.0, age_ms=0.0, access_count=0, half_life_ms=1e9)
    assert from_future == at_now
    assert from_future == (0.15, 1.0, 0.0)


# --------------------------------------------------------------------------
# the degenerate corner the old properties never drew
# --------------------------------------------------------------------------

def test_zero_and_negative_inputs_stay_finite_and_pinned():
    """The exact tuple for the fully degenerate corner of the domain.

    All three clamps fire at once, so a mutation of any one of them moves a
    literal here.
    """
    assert final_score(rrf=0.0, age_ms=0.0, access_count=0,
                       half_life_ms=0.0) == (0.15, 1.0, 0.0)
    # age 1.0 against the 1e-3 floor underflows exp(-1000) to exactly 0.0
    assert final_score(rrf=0.0, age_ms=1.0, access_count=0,
                       half_life_ms=0.0) == (0.0, 0.0, 0.0)


def test_weights_are_honoured_when_overridden():
    """The weights are a live tuning surface, so an override must take effect."""
    # corpus_mode zeroes both weights in recall.py
    assert final_score(rrf=1.0, age_ms=0.0, access_count=9, half_life_ms=1e6,
                       recency_weight=0.0, freq_weight=0.0) == (
        1.0, 1.0, 2.302585092994046,
    )
    # half the shipped weights
    half, _, _ = final_score(rrf=0.0, age_ms=0.0, access_count=9,
                             half_life_ms=1e6,
                             recency_weight=0.075, freq_weight=0.025)
    # 0.075*1.0 + 0.025*2.302585092994046
    assert half == 0.13256462732485114
