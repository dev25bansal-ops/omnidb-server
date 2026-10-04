"""ADR-08 / W3-3: recall fences by tenant across every leg.

The four MT-02 bypasses exist because `agent_id` was a self-declared label. The
fix is a hard `tenant_id = ?` equality on the lexical leg, the dense-leg
hydration, and the trigger matcher — plus a write path that stamps the row with
its tenant. These tests write two tenants into one home and assert neither can
see the other through any retrieval leg.
"""

from __future__ import annotations

import pytest
from conftest import make_store

from omnidb_server.store import MemoryStore


@pytest.fixture()
def two_tenant_store(tmp_path):
    """One home, two tenants, with distinctive per-tenant vocabulary."""
    st: MemoryStore = make_store(tmp_path)
    st.remember("acme quarterly roadmap for vector database", tenant="acme")
    st.remember("globex quarterly roadmap for billing service", tenant="globex")
    st.remember("acme secret launch date in october", tenant="acme")
    st.remember("globex internal pricing tiers", tenant="globex")
    yield st
    st.close()


def test_lexical_leg_is_tenant_fenced(two_tenant_store):
    """A query matching only the foreign tenant's vocabulary returns nothing."""
    st = two_tenant_store
    # 'billing' matches globex's row; acme asks, gets nothing
    hits = st.recall("billing service", k=5, tenant="acme")
    assert all(h["id"] in _ids(st, "acme") for h in hits)
    assert not any("globex" in h["content"].split() for h in hits)
    # the reverse: globex asking for acme's vocabulary
    hits_globex = st.recall("vector database", k=5, tenant="globex")
    assert all(h["id"] in _ids(st, "globex") for h in hits_globex)


def test_dense_leg_is_tenant_fenced(two_tenant_store):
    """The dense leg (ANN) is filtered by tenant before returning."""
    st = two_tenant_store
    # a semantically overlapping query must still come back tenant-scoped
    hits = st.recall("roadmap and planning", k=5, tenant="acme")
    assert hits, "acme should match its own roadmap row"
    assert all(h["id"] in _ids(st, "acme") for h in hits)


def test_trigger_matcher_is_tenant_fenced(two_tenant_store):
    """Procedural skill triggers fire only inside the requesting tenant."""
    st = two_tenant_store
    st.remember("globex deploy procedure", kind="procedural",
                trigger_patterns=["deploy"], tenant="globex")
    acme_hits = st.recall("deploy now", k=5, tenant="acme")
    assert not any(
        h.get("metadata", {}).get("skill_trigger", False) for h in acme_hits
    )
    globex_hits = st.recall("deploy now", k=5, tenant="globex")
    assert any(
        h.get("metadata", {}).get("skill_trigger", False) for h in globex_hits
    )


def test_default_tenant_is_still_isolated(two_tenant_store):
    """Rows written without an explicit tenant live in 'default', not 'acme'."""
    st = two_tenant_store
    st.remember("default tenant standalone note", tenant="default")
    hits = st.recall("standalone note", k=5, tenant="acme")
    assert not any("default" in h["content"] for h in hits)
    hits_default = st.recall("standalone note", k=5, tenant="default")
    assert any("default" in h["content"] for h in hits_default)


def _ids(store, tenant):
    return {r[0] for r in store.conn.execute(
        "SELECT id FROM memories WHERE tenant_id = ?", (tenant,))}