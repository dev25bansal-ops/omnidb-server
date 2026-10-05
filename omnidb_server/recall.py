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


def _as_mapping(value) -> dict:
    """Coerce a parsed or unparsed metadata value into a dict.

    `metadata` is a JSON column, so a row can legitimately contain a list, a
    string or a number. The main result path passes `json.loads(...)` through
    unchanged and is fine with that; the skill-trigger merge is not, because it
    splats the value (`{**parsed}`) and a non-mapping raises
    `TypeError: 'list' object is not a mapping` (audit C11).

    `skills.py` and `portability.py` already `isinstance(..., dict)` before
    touching `.get()`; this brings recall in line with them. A non-object
    becomes `{}` so the merge can still add `skill_trigger` — tolerating the row
    by *dropping* it would be a quieter bug than the crash, because the row
    would then vanish from skill recall with no error at all.

    Accepts both forms because the two merge sites differ: one reads the raw
    column, the other reads a value `run_recall` already parsed.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def run_recall(store, query: str, k: int, kind, k_fetch,
               agent_id, reranker, log_query: bool = False,
               tenant: str = "default", max_hops: int = 2) -> list[dict]:
    """AUDIT-I08: all legs select live rows of the requested kind/scope.

    None means no kind/scope restriction; an empty agent_id is a supplied
    scope and matches only rows carrying that exact empty string.

    ADR-08: `tenant` is a hard equality on every leg. It is passed down from
    the authenticated request (never sourced from the caller), and every query
    here fences rows by `tenant_id = ?` BEFORE any other predicate — so there
    is no query shape that escapes the tenant column.
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
    # ADR-08 / W3-3: the dense leg is tenant-scoped at the index, not
    # global-ANN-then-filter. Its postings live in the tenant's own sidecar,
    # so a foreign row is never a candidate and `fetch` is spent entirely on
    # this tenant's rows.
    dense_ids = [mid for mid, _score in store.index.query(qvec, fetch, tenant)]

    # ADR-08: tenant is fenced first — every candidate is narrowed to the
    # tenant's own rows before kind or agent scoping, so a foreign tenant's row
    # can never surface regardless of the other predicates.
    tenant_cond = "AND m.tenant_id = ? "
    agent_cond = "AND (? IS NULL OR m.agent_id = ?) "
    agent_params: list = [agent_id, agent_id]
    match_expr = fts_match_query(query)
    sparse_ids: list[int] = []
    # `_fts_table_existing`, not `_fts_table`: a read must not run DDL. On a
    # home with no rows for this tenant there is no FTS table, and there is
    # nothing for the sparse leg to contribute anyway — so the leg is skipped
    # rather than materialising an empty index to query.
    fts = store._fts_table_existing(tenant)
    if match_expr and fts:
        # ADR-08 / W3-3: the lexical leg queries the *tenant's own* FTS table,
        # so bm25 derives IDF from that tenant's corpus alone. On the shared
        # table every tenant's document frequencies reweighted every other
        # tenant's rare terms — measured: one tenant's score for a term moved
        # from -1e-06 to -1.256e-06 when an unrelated tenant added 400 rows
        # containing it, with this tenant's own data untouched.
        #
        # The `m.tenant_id = ?` fence below stays. It is now redundant against
        # a per-tenant table, and that is the point: it also excludes
        # soft-deleted and superseded rows via `_SELECTIVE`, which table
        # separation does not do.
        rows = store.conn.execute(
            f"SELECT f.rowid FROM {fts} f "
            "JOIN memories m ON m.id = f.rowid "
            f"WHERE {fts} MATCH ? AND m.{_SELECTIVE} "
            + tenant_cond +
            "AND (? IS NULL OR m.kind = ?) "
            + agent_cond +
            f"ORDER BY bm25({fts}) LIMIT ?",
            (match_expr, tenant, kind, kind, *agent_params, fetch),
        ).fetchall()
        sparse_ids = [int(r["rowid"]) for r in rows]

    seed_scores = rrf([dense_ids, sparse_ids], k=store.cfg.rrf_k,
                  weights=[store.cfg.dense_weight, store.cfg.sparse_weight])

    # ADR-08 / W3-3: the graph leg. Off when graph_weight == 0.0, so the guard
    # is the weight itself — no separate flag that can disagree with it and
    # produce silently dead config.
    #
    # Seeded from the *fused* ranking rather than a single leg, because the
    # fused ranking is the only place in the pipeline where leg agreement is
    # expressed and the thing a graph leg should amplify is exactly that
    # agreement, not whichever leg happened to be stronger.
    graph_ids: list[int] = []
    if store.graph_enabled and seed_scores:
        # One extra subtlety: fused ordering is score-descending with no id
        # tie-break, so the seeds get their own explicit (-score, id) order
        # rather than inheriting a tie-break-free one. A derived leg must
        # never be *less* deterministic than what it is seeded from.
        seeds = [mid for mid, _ in sorted(
            seed_scores.items(), key=lambda pair: (-pair[1], pair[0]))]
        budget = store.cfg.graph_expand_budget or max(4 * k, 256)
        graph_ids = store.graph_ranking(
            seeds[:20], kind=kind, agent_id=agent_id, tenant=tenant,
            depth=max_hops, budget=budget)
        # Provenance for the unified surface: which rows the graph leg added.
        # Store-side attribute so query() can annotate without a second run.
        try:
            store.graph_leg_last = list(graph_ids)
        except AttributeError:
            pass

    # Two rrf() calls rather than seeding dense_ids.extend(graph_ids), because
    # the two-call form makes the bit-identity guarantee trivially provable:
    # with an empty leg, fused is literally today's expression. And for an id not
    # in graph_ids the accumulated float is bit-identical, because rrf()
    # accumulates in list order and the first two lists are unchanged.
    if graph_ids:
        fused = rrf([dense_ids, sparse_ids, graph_ids], k=store.cfg.rrf_k,
                    weights=[store.cfg.dense_weight, store.cfg.sparse_weight,
                             store.cfg.graph_weight])
    else:
        fused = seed_scores
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
                f"AND {_SELECTIVE} AND tenant_id = ? "
                f"AND (? IS NULL OR kind = ?) "
                "AND (? IS NULL OR agent_id = ?)",
                [*part, tenant, kind, kind, agent_id, agent_id],
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
        # The two kind predicates in this query are a conjunction and BOTH are
        # load-bearing, which is not obvious: one restricts rows to procedural
        # skills, the other is the caller scope filter. Together they mean
        # "procedural rows, when the caller asked for procedural or for no kind
        # at all". Deleting either changes behaviour - dropping the hardcoded one
        # lets a kind=semantic recall return procedural trigger rows, because the
        # surviving clause compares the COLUMN to the literal and so is true for
        # every procedural row. Kept deliberately, not as an oversight.
        pat_rows = store.conn.execute(
            "SELECT id, trigger_patterns FROM memories "
            "WHERE trigger_patterns IS NOT NULL "
            f"AND {_SELECTIVE} AND tenant_id = ? AND kind = 'procedural' "
            "AND (? IS NULL OR kind = ?) AND (? IS NULL OR agent_id = ?)",
            [tenant, kind, kind, agent_id, agent_id],
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
            f"SELECT * FROM memories WHERE id IN ({placeholders}) "
            f"AND {_SELECTIVE} AND tenant_id = ?",
            [*sorted(missing), tenant],
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
                "metadata": {**_as_mapping(row["metadata"]), "skill_trigger": True},
                "created_at": _iso(row["created_ms"]),
                "access_count": row["access_count"],
                "rrf": round(0.0, 6),
                "recency": round(recency, 6),
                "frequency": round(freq, 6),
                "score": round(score, 6),
            })
            seen.add(row["id"])
    for item in results:
        if item["id"] in triggered and not (isinstance(item["metadata"], dict)
                                             and "skill_trigger" in item["metadata"]):
            # A non-object metadata is coerced here rather than skipped: the tag
            # is what makes the row reachable through skill recall.
            item["metadata"] = {**_as_mapping(item["metadata"]),
                                "skill_trigger": True}

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
                      tenant=tenant, commit=False)
    if final_ids or log_query:
        try:
            store.conn.commit()
        except sqlite3.Error as exc:
            raise _commit_or_rollback(store.conn, exc) from exc
    return results
