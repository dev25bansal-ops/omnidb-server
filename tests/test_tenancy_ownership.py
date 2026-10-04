"""ADR-08 / W3-4: id-addressed operations are tenant-scoped (SEC-02, 9.5).

The audit found `agent_id` was a label, not a boundary: `forget_many([1..N])`
could tombstone every row, `get(id)` read any row, and `related()`/`stats()`
leaked across the home. These tests write two tenants and assert that no
id-addressed read or write crosses the boundary.
"""

from __future__ import annotations

import pytest
from conftest import make_store

from omnidb_server.store import StoreError


@pytest.fixture()
def two_tenants(tmp_path):
    st = make_store(tmp_path)
    a = st.remember("acme alpha fact", tenant="acme")["id"]
    b = st.remember("globex beta fact", tenant="globex")["id"]
    yield st, a, b
    st.close()


def test_get_is_tenant_scoped(two_tenants):
    st, a, b = two_tenants
    assert st.get(a, "acme") is not None
    assert st.get(b, "acme") is None          # foreign id is invisible
    assert st.get(b, "globex") is not None


def test_forget_many_cannot_tombstone_a_foreign_tenant(two_tenants):
    st, a, b = two_tenants
    result = st.forget_many([a, b], tenant="acme")
    assert result["forgotten"] == [a]         # only its own row
    assert b in result["missing"]             # the foreign id is "missing"
    assert st.get(a, "acme")["deleted_ms"] is not None   # own row soft-deleted
    assert st.get(b, "globex")["deleted_ms"] is None     # foreign row untouched


def test_update_memory_cannot_edit_a_foreign_tenant(two_tenants):
    st, a, b = two_tenants
    with pytest.raises(StoreError, match="does not exist"):
        st.update_memory(b, content="overwritten by acme", tenant="acme")
    assert st.get(b, "globex")["content"] == "globex beta fact"


def test_supersede_cannot_cross_tenants(two_tenants):
    st, a, b = two_tenants
    with pytest.raises(StoreError):
        st.supersede(a, b, tenant="acme")     # new_id is foreign
    assert st.get(b, "globex")["superseded_by"] is None


def test_link_cannot_bridge_tenants(two_tenants):
    st, a, b = two_tenants
    with pytest.raises(StoreError, match="does not exist"):
        st.link(a, b, tenant="acme")          # dst is foreign


def test_related_walk_stays_within_tenant(two_tenants):
    st, a, b = two_tenants
    # even if a cross-tenant edge somehow existed, the walk would not follow it
    st.conn.execute(
        "INSERT INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, ?)",
        (a, b, "forged", 1),
    )
    st.conn.commit()
    neighbors = st.related(a, max_hops=1, tenant="acme")
    assert b not in {row["id"] for row in neighbors}


def test_stats_is_tenant_scoped(two_tenants):
    st, a, b = two_tenants
    acme = st.stats("acme")
    assert acme["tenant"] == "acme"
    assert acme["live"] == 1
    # globex's row is not in acme's counts
    assert acme["live"] == 1 and st.stats("globex")["live"] == 1


def test_purge_is_tenant_scoped(tmp_path):
    st = make_store(tmp_path)
    acme_id = st.remember("acme to delete", tenant="acme")["id"]
    globex_id = st.remember("globex to delete", tenant="globex")["id"]
    st.forget(acme_id, "acme")
    st.forget(globex_id, "globex")

    result = st.purge(tenant="acme")
    assert result["purged"] == 1                # only acme's row
    # globex's soft-deleted row survives acme's purge
    assert st.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = ?", (globex_id,)
    ).fetchone()[0] == 1
    st.close()