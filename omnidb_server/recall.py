from __future__ import annotations

import json
import re
import sqlite3
import time

from .fusion import fts_match_query, rrf
from .scoring import final_score
from .store import _SELECTIVE, E_TYPE, StoreError, _commit_or_rollback, _iso, _record_query

# BUG-12: k and k_fetch are clamped so a hostile/buggy client can neither
# blow past SQLite's variable limit with fused-id placeholders nor allocate
# an unbounded candidate pool.
_MAX_RECALL_K = 1000

# PERF-03: bounded compiled-regex cache for skill triggers, sized above
# re's default 512-entry cache so large procedural corpora don't thrash it.
_PATTERN_CACHE: dict[str, re.Pattern | None] = {}
_PATTERN_CACHE_LIMIT = 4096

# AUDIT-I06: a compiled-pattern cache bounds memory, not matching work.
# Python's backtracking engine has no execution-time interrupt that can be
# relied on from this thread, so the recall path caps its own inputs:
# queries longer than the admission bound skip trigger matching entirely,
# and at most _MAX_TRIGGER_EVALUATIONS patterns are evaluated per recall.
# Admission limits on what can ever be STORED (pattern count/length) live
# in validation.py and apply to creation and import alike.
_MAX_TRIGGER_QUERY_CHARS = 4096
_MAX_TRIGGER_EVALUATIONS = 2048


def _compiled_pattern(pat: str) -> re.Pattern | None:
    """Compile a skill-trigger regex; invalid patterns cache as None so an
    importer-supplied bad regex is skipped, not fatal, and not re-parsed."""
    if pat in _PATTERN_CACHE:
        return _PATTERN_CACHE[pat]
    if len(_PATTERN_CACHE) >= _PATTERN_CACHE_LIMIT:
        _PATTERN_CACHE.clear()
    try:
        compiled: re.Pattern | None = re.compile(pat)
    except re.error:
        compiled = None
    _PATTERN_CACHE[pat] = compiled
    return compiled


def run_recall(store, query: str, k: int, kind, k_fetch,
               agent_id, reranker, log_query: bool = False) -> list[dict]:
    """AUDIT-I08: all legs select live rows of the requested kind/scope.

    None means no kind/scope restriction; an empty agent_id is a supplied
    scope and matches only rows carrying that exact empty string.
    """
    t0 = time.perf_counter()
    if k < 1:
        raise StoreError("k must be >= 1", E_TYPE)
    k = min(int(k), _MAX_RECALL_K)
    fetch = min(max(int(k_fetch) if k_fetch else store.cfg.k_fetch, k),
                _MAX_RECALL_K)
    if agent_id is not None and store.cfg.agent_overfetch > 1.0:
        # PERF-08: the dense leg is global-ANN-then-filter, so when scoped by
        # agent_id the foreign rows it returns will be dropped. Over-fetch
        # proportionally so one dominant agent cannot crowd a scoped recall
        # below k; the cap keeps the placeholder count bounded.
        fetch = min(_MAX_RECALL_K, int(fetch * store.cfg.agent_overfetch))
    qvec = store.embedder.encode([query])[0]
    dense_ids = [mid for mid, _score in store.index.query(qvec, fetch)]

    agent_cond = "AND (? IS NULL OR m.agent_id = ?) "
    agent_params: list = [agent_id, agent_id]
    match_expr = fts_match_query(query)
    sparse_ids: list[int] = []
    if match_expr:
        rows = store.conn.execute(
            "SELECT f.rowid FROM memory_fts f "
            "JOIN memories m ON m.id = f.rowid "
            f"WHERE memory_fts MATCH ? AND m.{_SELECTIVE} "
            "AND (? IS NULL OR m.kind = ?) "
            + agent_cond +
            "ORDER BY bm25(memory_fts) LIMIT ?",
            (match_expr, kind, kind, *agent_params, fetch),
        ).fetchall()
        sparse_ids = [int(r["rowid"]) for r in rows]

    fused = rrf([dense_ids, sparse_ids], k=store.cfg.rrf_k,
                weights=[store.cfg.dense_weight, store.cfg.sparse_weight])
    scored = sorted(fused.items(), key=lambda pair: -pair[1])
    fused_rows: list[sqlite3.Row] = []
    if scored:
        # AUDIT-I09: filter the complete fetched pool before windowing.
        top_ids = [mid for mid, _ in scored]
        batch_size = min(500, store.conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) - 4)
        for start in range(0, len(top_ids), max(1, batch_size)):
            part = top_ids[start:start + max(1, batch_size)]
            placeholders = ",".join("?" * len(part))
            fused_rows.extend(store.conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders}) "
                f"AND {_SELECTIVE} AND (? IS NULL OR kind = ?) "
                "AND (? IS NULL OR agent_id = ?)",
                [*part, kind, kind, agent_id, agent_id],
            ).fetchall())
        fused_rows.sort(key=lambda row: (-fused[row["id"]], row["id"]))
        fused_rows = fused_rows[:max(k * 4, k)]

    now_ms = int(store.now_fn() * 1000)
    half_life_ms = max(store.cfg.half_life_hours, 0.001) * 3600_000
    # corpus_mode disables the memory rerank (see config.py: measured 40-56%
    # nDCG loss on corpus workloads); fusion order is the final order.
    rec_w = 0.0 if store.cfg.corpus_mode else store.cfg.recency_weight
    freq_w = 0.0 if store.cfg.corpus_mode else store.cfg.freq_weight
    results = []
    for row in fused_rows:
        score, recency, freq = final_score(
            rrf=fused[row["id"]],
            age_ms=now_ms - row["created_ms"],
            access_count=row["access_count"],
            half_life_ms=half_life_ms,
            recency_weight=rec_w,
            freq_weight=freq_w,
        )
        results.append({
            "id": row["id"],
            "kind": row["kind"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]),
            "created_at": _iso(row["created_ms"]),
            "access_count": row["access_count"],
            "rrf": round(fused[row["id"]], 6),
            "recency": round(recency, 6),
            "frequency": round(freq, 6),
            "score": round(score, 6),
        })

    # live procedural skills fire on regex triggers regardless of vector/FTS
    # rank. Two-phase (PERF-03): match on (id, trigger_patterns) only, then
    # fetch full rows for matches — the common no-match case stops moving
    # every procedural row's full content on every recall.
    # AUDIT-I08: triggers obey the same kind and exact-scope contract.
    pat_rows = []
    if len(query) <= _MAX_TRIGGER_QUERY_CHARS:
        # AUDIT-I06: oversized queries skip trigger matching rather than
        # feeding unbounded input to a backtracking engine.
        pat_rows = store.conn.execute(
            "SELECT id, trigger_patterns FROM memories "
            "WHERE trigger_patterns IS NOT NULL "
            f"AND {_SELECTIVE} AND kind = 'procedural' "
            "AND (? IS NULL OR kind = ?) AND (? IS NULL OR agent_id = ?)",
            [kind, kind, agent_id, agent_id],
        ).fetchall()
    triggered: set[int] = set()
    evaluations = 0
    for row in pat_rows:
        try:
            patterns = json.loads(row["trigger_patterns"])
        except (TypeError, ValueError):
            continue
        if not isinstance(patterns, list):
            continue
        for pat in patterns:
            if evaluations >= _MAX_TRIGGER_EVALUATIONS:
                break
            compiled = _compiled_pattern(str(pat))
            if compiled is None:
                continue
            evaluations += 1
            if compiled.search(query):
                triggered.add(int(row["id"]))
                break
        if evaluations >= _MAX_TRIGGER_EVALUATIONS:
            break

    seen = {r["id"] for r in results}
    missing = triggered - seen
    if missing:
        placeholders = ",".join("?" * len(missing))
        extra_rows = store.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            sorted(missing),
        ).fetchall()
        for row in extra_rows:
            score, recency, freq = final_score(
                rrf=0.0,
                age_ms=now_ms - row["created_ms"],
                access_count=row["access_count"],
                half_life_ms=half_life_ms,
                recency_weight=rec_w,
                freq_weight=freq_w,
            )
            results.append({
                "id": row["id"],
                "kind": row["kind"],
                "content": row["content"],
                "metadata": {**json.loads(row["metadata"]), "skill_trigger": True},
                "created_at": _iso(row["created_ms"]),
                "access_count": row["access_count"],
                "rrf": round(0.0, 6),
                "recency": round(recency, 6),
                "frequency": round(freq, 6),
                "score": round(score, 6),
            })
            seen.add(row["id"])
    for item in results:
        if item["id"] in triggered and "skill_trigger" not in item["metadata"]:
            item["metadata"] = {**item["metadata"], "skill_trigger": True}

    if reranker is not None:
        from .rerank import apply_reranker

        results = apply_reranker(reranker, query, results)

    results.sort(key=lambda item: (-item["score"], item["id"]))
    results = results[:k]

    final_ids = [r["id"] for r in results]
    if final_ids:
        ph = ",".join("?" * len(final_ids))
        store.conn.execute(
            f"UPDATE memories SET access_count = access_count + 1, "
            f"last_accessed_ms = ? WHERE id IN ({ph})",
            [now_ms, *final_ids],
        )
    # PERF-04: the access bump and the query-log row share one commit (and
    # recall no longer commits at all when it neither bumps nor logs)
    if log_query:
        _record_query(store, "recall", query, len(results),
                      round((time.perf_counter() - t0) * 1000, 2), agent_id,
                      commit=False)
    if final_ids or log_query:
        try:
            store.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(store.conn, exc) from exc
    return results
