"""Pure scoring math for recall reranking — the tuning surface.

Kept dependency-free and total (no exceptions for any finite input) so it can
be property-tested directly and tuned without touching SQL or index code.
"""

from __future__ import annotations

import math


def final_score(*, rrf: float, age_ms: float, access_count: int,
                half_life_ms: float, recency_weight: float = 0.15,
                freq_weight: float = 0.05) -> tuple[float, float, float]:
    """Combine fusion rank, recency decay, and log-frequency into one score.

    Returns (score, recency, frequency). Properties (property-tested):
    - finite for all finite inputs (clock skew clamped at zero age)
    - recency in [0, 1], non-increasing in age
    - score non-decreasing in access_count
    """
    age = max(float(age_ms), 0.0)
    half_life = max(float(half_life_ms), 1e-3)
    accesses = max(float(access_count), 0.0)

    recency = math.exp(-age / half_life)
    frequency = math.log1p(accesses)
    score = float(rrf) + recency_weight * recency + freq_weight * frequency
    return score, recency, frequency
