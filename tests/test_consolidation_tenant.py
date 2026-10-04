"""F12 - consolidation is tenant-scoped end to end (ADR-08).

`consolidate` was the one mutating tool that passed no tenant. It read episodes
with no tenant predicate, deduped against every tenant's semantic rows, wrote
facts through `store.remember` with the default tenant, and marked episodes consumed
by id alone. On a multi-tenant home one call therefore distilled every tenant's
episodes into the default tenant, could rewrite another tenant's facts, and
could send another tenant's transcript to the configured LLM endpoint. That is
attack A6/G6 in `docs/MULTI_TENANCY_THREAT_MODEL.md`, and it is the one item the
Wave 3 report left open next to the now-fixed fail-open resolver.

Each test below states one of those leaks. Every one goes red if the tenant
argument is dropped from any single call site, so the coverage is per-leak
rather than one aggregate smoke test.
"""

from __future__ import annotations

import json

from conftest import make_store

import omnidb_server.consolidate as consolidate
from omnidb_server.server import attach_tools

TENANT = "acme"
OTHER = "globex"


def _episodes(st, tenant, n=3, prefix="episode"):
    return [
        st.remember(f"{prefix} {i}: the team ships on blue widgets",
                    kind="episodic", tenant=tenant)["id"]
        for i in range(n)
    ]


def _semantic_by_tenant(st):
    rows = st.conn.execute(
        "SELECT tenant_id, content, metadata FROM memories WHERE kind = 'semantic' "
        "AND deleted_ms IS NULL AND superseded_by IS NULL"
    ).fetchall()
    out = {}
    for tenant, content, metadata in rows:
        out.setdefault(tenant, []).append((content, json.loads(metadata)))
    return out


def _consumed_flags(st, ids):
    flags = {}
    for i in ids:
        row = st.conn.execute(
            "SELECT metadata FROM memories WHERE id = ?", (i,)
        ).fetchone()
        flags[i] = bool(json.loads(row[0]).get("consolidated"))
    return flags


# --- leak 1: reading another tenant's episodes ------------------------------


def test_consolidation_reads_only_the_callers_episodes(tmp_path):
    """The episode scan must be fenced, or one run distils every tenant."""
    st = make_store(tmp_path)
    _episodes(st, TENANT, 3, "acme-episode")
    _episodes(st, OTHER, 3, "globex-episode")

    report = consolidate.run_consolidation(st, hours=24, tenant=TENANT)

    assert report["episodes_scanned"] == 3
    st.close()


# --- leak 2: writing facts into the wrong tenant ----------------------------


def test_facts_are_written_into_the_callers_tenant(tmp_path):
    """store.remember defaults to the default tenant; consolidation must not
    inherit that default, or a run for acme deposits its facts in default."""
    st = make_store(tmp_path)
    _episodes(st, TENANT, 3, "acme-episode")
    _episodes(st, OTHER, 3, "globex-episode")

    report = consolidate.run_consolidation(st, hours=24, tenant=TENANT)

    assert report["facts_created"] == 3
    assert set(_semantic_by_tenant(st)) == {TENANT}
    st.close()


# --- leak 3: marking another tenant's rows consumed -------------------------


def test_consumption_is_marked_for_this_tenant_only(tmp_path):
    """_mark_consolidated writes by id alone. A foreign id in consumed_ids
    would be silently retired, so that run can never happen again.

    Both directions are asserted on purpose. Asserting only the negative
    passes just as well when the update is fenced to the WRONG tenant, which
    retires nothing at all and stalls consolidation forever."""
    st = make_store(tmp_path)
    mine = _episodes(st, TENANT, 3, "acme-episode")
    theirs = _episodes(st, OTHER, 3, "globex-episode")

    consolidate.run_consolidation(st, hours=24, tenant=TENANT)

    assert all(_consumed_flags(st, mine).values()), (
        "the caller tenant own episodes were not marked, so the run repeats"
    )
    foreign = _consumed_flags(st, theirs)
    assert foreign and not any(foreign.values()), (
        "consolidation for %s marked %s episodes consumed" % (TENANT, OTHER)
    )
    st.close()


# --- leak 4: dedup rewriting another tenant's fact -------------------------


def test_dedup_works_within_a_tenant_and_never_crosses(tmp_path):
    """The dedup pool is the highest-risk read: a matching foreign row is
    handed to store.update_memory, which rewrites the row in place.

    Two rows carry the same text, one per tenant. The same-tenant match must
    still be found and folded, otherwise a pool that is fenced too hard just
    looks safe while silently deduplicating nothing."""
    st = make_store(tmp_path)
    shared = "the team ships on blue widgets"
    st.remember(shared, kind="semantic", tenant=OTHER)
    st.remember(shared, kind="semantic", tenant=TENANT)
    st.remember(shared, kind="episodic", tenant=TENANT)
    st.remember("second acme note about red widgets", kind="episodic", tenant=TENANT)
    st.remember("third acme note about green widgets", kind="episodic", tenant=TENANT)

    report = consolidate.run_consolidation(st, hours=24, tenant=TENANT)

    by_tenant = _semantic_by_tenant(st)
    foreign = by_tenant.get(OTHER, [])
    assert len(foreign) == 1, "the foreign row was consumed or duplicated"
    assert foreign[0][1] == {}, "the foreign fact was rewritten by dedup"
    assert report["facts_deduplicated"] == 1, (
        "the same-tenant match was missed, so the fence is too coarse"
    )
    assert report["facts_created"] == 2
    st.close()


# --- leak 5: the transcript that leaves the process -----------------------


def test_the_llm_transcript_never_carries_another_tenant(tmp_path, monkeypatch):
    """The exfiltration path: whatever the scan is fenced to is what a
    third-party LLM endpoint receives. This is A6/G6."""
    st = make_store(tmp_path, llm_base_url="http://llm.local/v1",
                    llm_model="test-model")
    _episodes(st, TENANT, 3, "acme-episode")
    _episodes(st, OTHER, 3, "globex-payroll")

    seen = []

    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {
                "content": '[{"fact": "distilled", "confidence": 0.9}]'}}]}

    def _fake_post(url, json=None, headers=None, timeout=None):
        seen.append(json["messages"][1]["content"])
        return _R()

    monkeypatch.setattr(consolidate.httpx, "post", _fake_post)
    consolidate.run_consolidation(st, hours=24, tenant=TENANT)

    assert seen, "the LLM path was never exercised, so this proves nothing"
    for body in seen:
        assert "globex-payroll" not in body
    st.close()


# --- the reachable surface: the MCP tool, not the library default ----------


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_a, **_k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_the_tool_passes_the_resolved_tenant(tmp_path):
    """run_consolidation's default is the single-tenant local path. The HTTP
    transport must override it, exactly as it does for the other nine tools."""
    import asyncio

    st = make_store(tmp_path)
    _episodes(st, TENANT, 3, "acme-episode")
    mcp = _FakeMCP()
    attach_tools(mcp, st, tenant_resolver=lambda: TENANT)

    asyncio.run(mcp.tools["consolidate"]())

    assert set(_semantic_by_tenant(st)) == {TENANT}
    st.close()
