"""Property-based tests for the pure scoring surface (scoring.final_score).

These encode the invariants the recall reranker depends on: totality (finite
output for any finite input), recency bounded in [0,1], monotonicity in age
and access count, and clock-skew safety.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from omnidb_server.scoring import final_score

numbers = st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
small_positive = st.floats(min_value=1e-3, max_value=1e9, allow_nan=False)


@settings(max_examples=300)
@given(rrf=st.floats(-5, 5, allow_nan=False),
       age_ms=st.floats(-1e12, 1e15, allow_nan=False),
       access_count=st.integers(min_value=0, max_value=10**12),
       half_life_ms=small_positive,
       recency_weight=st.floats(0, 2, allow_nan=False),
       freq_weight=st.floats(0, 2, allow_nan=False))
def test_total_and_recency_bounded(rrf, age_ms, access_count, half_life_ms,
                                   recency_weight, freq_weight):
    score, recency, frequency = final_score(
        rrf=rrf, age_ms=age_ms, access_count=access_count,
        half_life_ms=half_life_ms,
        recency_weight=recency_weight, freq_weight=freq_weight,
    )
    assert math.isfinite(score)
    assert math.isfinite(recency) and math.isfinite(frequency)
    assert 0.0 <= recency <= 1.0
    assert frequency >= 0.0


@settings(max_examples=200)
@given(base=numbers, delta=st.floats(0.01, 1e9), access_count=st.integers(0, 1000),
       half_life_ms=small_positive)
def test_score_non_increasing_in_age(base, delta, access_count, half_life_ms):
    younger, _, _ = final_score(rrf=base, age_ms=1_000, access_count=access_count,
                             half_life_ms=half_life_ms)
    older, older_recency, _ = final_score(rrf=base, age_ms=1_000 + delta,
                                          access_count=access_count,
                                          half_life_ms=half_life_ms)
    assert older <= younger + 1e-12
    assert older_recency <= 1.0


@settings(max_examples=200)
@given(base=numbers, age_ms=st.floats(0, 1e9), acc1=st.integers(0, 10**6),
       extra=st.integers(1, 10**6), half_life_ms=small_positive)
def test_score_non_decreasing_in_access_count(base, age_ms, acc1, extra,
                                              half_life_ms):
    fewer, _, _ = final_score(rrf=base, age_ms=age_ms, access_count=acc1,
                           half_life_ms=half_life_ms)
    more, _, _ = final_score(rrf=base, age_ms=age_ms, access_count=acc1 + extra,
                             half_life_ms=half_life_ms)
    assert more >= fewer - 1e-12


def test_clock_skew_clamped():
    """created_at after 'now' must not explode the exponent."""
    score, recency, _ = final_score(rrf=1.0, age_ms=-10**12, access_count=0,
                                    half_life_ms=168 * 3600_000)
    assert math.isfinite(score)
    assert recency == 1.0
