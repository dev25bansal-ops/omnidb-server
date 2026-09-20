from __future__ import annotations

import re

# \w is unicode-aware by default in Python 3: keeps café/北京 searchable,
# matching the unicode61 FTS5 tokenizer
_TOKEN = re.compile(r"\w+")


def rrf(rankings: list[list[int]], k: int = 60,
        weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion over ranked id lists (Cormack et al. 2009).

    weights: per-leg multipliers (must match len(rankings) when given).
    Equal weights are the classical formulation. Measured behavior (see
    bench/comparative-results.md and bench/research-results.md): with a
    weak/offline embedder, equal-weight fusion lets a noisy lexical leg
    outvote correct dense evidence (recall@5 0.97 -> 0.42; repaired to
    ~0.97 dense-equivalent by weighting the dense leg); with a strong
    embedder on BEIR corpora, the lexical leg adds noise at ANY weight and
    hybrid trails vector-only — the optimal weight is dataset-dependent
    (0.7 on nfcorpus, <0.3 on scifact). Treat weights as a per-deployment
    tuning surface, not a universal default.
    """
    scores: dict[int, float] = {}
    if weights is None:
        weights = [1.0] * len(rankings)
    for ranked, weight in zip(rankings, weights):
        for position, mid in enumerate(ranked):
            scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 1)
    return scores


def fts_match_query(text: str, max_terms: int = 24) -> str:
    """Build a safe FTS5 MATCH expression: quoted OR-ed tokens.

    Quoting neutralizes FTS syntax characters from user input; OR widens
    recall and lets bm25() do the ranking.
    """
    tokens = _TOKEN.findall(text.lower())[:max_terms]
    return " OR ".join('"%s"' % token.replace('"', '""') for token in tokens)
