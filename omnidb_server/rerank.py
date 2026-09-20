"""Pluggable reranker hook applied by store.recall (analysis §4.2).

Contract: a reranker is a callable (query, results) -> results. store.recall
re-sorts by (-score, id) after the hook returns, so a reranker whose ordering
is not derivable from "score" alone must encode its chosen order back into
each item's "score" (recency_first writes ordinal ranks). Custom strategies
plug in via @register("name") and are selected by name at recall time.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .store import E_TYPE, StoreError

Reranker = Callable[[str, list[dict]], list[dict]]

REGISTRY: dict[str, Reranker] = {}


def register(name: str) -> Callable[[Reranker], Reranker]:
    def wrap(fn: Reranker) -> Reranker:
        REGISTRY[name] = fn
        return fn

    return wrap


_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(str(text).lower()))


@register("none")
def _rerank_none(query: str, results: list[dict]) -> list[dict]:
    return results


@register("lexical_overlap")
def _rerank_lexical(query: str, results: list[dict]) -> list[dict]:
    query_tokens = _tokens(query)
    for item in results:
        overlap = len(query_tokens & _tokens(item.get("content", "")))
        item["score"] = float(item.get("score", 0.0)) * (
            1.0 + overlap / max(1, len(query_tokens))
        )
    results.sort(key=lambda item: (-item["score"], item["id"]))
    return results


@register("recency_first")
def _rerank_recency(query: str, results: list[dict]) -> list[dict]:
    ranked = sorted(
        results,
        key=lambda item: (
            -float(item.get("recency", 0.0)),
            -float(item.get("score", 0.0)),
            item["id"],
        ),
    )
    # store.recall re-sorts by (-score, id) after this hook, so the chosen
    # order must be written back into "score" to survive it
    for position, item in enumerate(ranked):
        item["score"] = float(len(ranked) - position)
    return ranked


def apply_reranker(name: str, query: str, results: list[dict]) -> list[dict]:
    try:
        reranker = REGISTRY[name]
    except KeyError:
        # QA-02: typed like every other store-surface error, not a bare ValueError
        raise StoreError(f"unknown reranker '{name}'", E_TYPE) from None
    return reranker(query, results)
