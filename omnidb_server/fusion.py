from __future__ import annotations

import re
import unicodedata

# \w is unicode-aware by default in Python 3: keeps café/北京 searchable,
# matching the unicode61 FTS5 tokenizer
_TOKEN = re.compile(r"\w+")


def _fold(text: str) -> str:
    """Lowercase, then drop only the marks that `.lower()` itself introduced.

    F36. `str.lower()` maps U+0130 (İ) to two code points - i plus U+0307
    COMBINING DOT ABOVE - and U+0307 is not a \\w character, so the tokenizer
    split mid-word and "İstanbul" was searched as "stanbul".

    Only marks that arrive as a result of the case fold are removed. Stripping
    every combining mark was measured and REJECTED: it fixes İstanbul but makes
    a decomposed "ñ" (n + U+0303) fold to "n" while the precomposed "ñ" keeps
    its tilde, so one word then tokenizes two different ways depending on the
    caller's spelling. That is a worse defect than the one being fixed, and it
    is not an ordering mistake - n + U+0303 has no NFC composition to
    precomposed ñ at all, because composition runs through uppercase Ñ. NFC,
    NFKC and NFKD were all measured and none of them resolve the pair.

    Measured 2026-10-02 (.omni-audit/f36_unicode_fix.py) on fiqa: four candidate
    fixes all scored sparse nDCG@10 0.25407 with 0/648 queries changed, so the
    change is quality-neutral on this corpus and needs no re-run.

    The residual limitation is stated rather than papered over: a decomposed
    "ñ" (n + U+0303) still does NOT match a precomposed "ñ", because .lower()
    runs first and precomposing that pair requires the uppercase base. That is
    F38, and it is left as-is here - fixing it would mean changing what the
    caller must send, not changing this function.
    """
    # Marks that .lower() itself creates: compare the fold against the source
    # so pre-existing combining marks are preserved.
    lowered = text.lower()
    if len(lowered) == len(text):
        return lowered                      # no expansion, so no introduced mark
    # .lower() expanded, so at least one char folded to >1. Drop combining
    # marks only when they are a by-product of that expansion.
    return "".join(c for c in lowered if not unicodedata.combining(c))

# F35: the old OR-of-every-token expression matched ~88% of the corpus on fiqa,
# because function words hit nearly every document, forcing bm25() to score the
# whole collection to return 50 rows. Dropping them was measured 2026-10-02
# (.omni-audit/f35_fts_quality.py): the sparse leg runs ~3x faster and sparse
# nDCG@10 is unchanged-to-slightly-better (0.2479 -> 0.2545 on fiqa, minilm
# vectors, n=648 queries). "not"/"no" are deliberately NOT here: they carry
# negation signal in questions ("what NOT to do").
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "for", "from", "get", "has", "have", "how", "i", "if",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "out",
    "should", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "up", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
})


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
    elif len(weights) != len(rankings):
        # The loop below zips the two, and `zip` stops at the shorter iterable.
        # A mismatch therefore discarded a whole leg silently: its ids vanished
        # from the result and shared ids lost their contribution, with no
        # exception and no warning. The docstring above already said the
        # lengths must match; this makes it true (audit C10).
        raise ValueError(
            f"rrf: len(weights)={len(weights)} != len(rankings)={len(rankings)}"
        )
    for ranked, weight in zip(rankings, weights):
        for position, mid in enumerate(ranked):
            scores[mid] = scores.get(mid, 0.0) + weight / (k + position + 1)
    return scores


def fts_match_query(text: str, max_terms: int = 24) -> str:
    """Build a safe FTS5 MATCH expression: quoted OR-ed CONTENT tokens.

    Quoting neutralizes FTS syntax characters from user input. Stopwords in
    ``_STOP_WORDS`` are dropped so a query of only function words yields "" —
    callers (``recall.run_recall``) already treat a falsey expression as "no
    sparse leg" — and the cap counts content terms, not raw ones. Since only a
    token's *membership* changes, quoting is unaffected (it never was the
    stopword list).
    """
    tokens = [t for t in _TOKEN.findall(_fold(text))
              if t not in _STOP_WORDS][:max_terms]
    return " OR ".join('"%s"' % token.replace('"', '""') for token in tokens)
