"""MemoryStore.tenants() operator view of per-tenant live/soft-deleted counts."""
from __future__ import annotations

from pathlib import Path

from conftest import make_store


def test_tenants_reports_each_tenant_with_counts(tmp_path: Path):
    """Multi-tenant data is accurately counted by tenants()."""
    store = make_store(tmp_path)
    try:
        # Add acme entries
        store.remember("acme semantic", tenant="acme")
        store.remember("acme episodic", kind="episodic", tenant="acme")
        # Add globex entry
        gone_id = store.remember("globex about to forget", tenant="globex")["id"]
        store.forget(gone_id, tenant="globex")  # mark as soft-deleted

        tenants = {t["tenant"]: t for t in store.tenants()}
        assert set(tenants) == {"acme", "globex"}
        # acme: 2 live (semantic + episodic), 0 soft-deleted
        assert tenants["acme"]["live"] == 2
        assert tenants["acme"]["soft_deleted"] == 0
        assert tenants["acme"]["superseded"] == 0
        assert tenants["acme"]["total"] == 2
        assert set(tenants["acme"]["by_kind"]) == {"semantic", "episodic"}
        # globex: 0 live (forgotten), 1 soft-deleted
        assert tenants["globex"]["live"] == 0
        assert tenants["globex"]["soft_deleted"] == 1
        assert tenants["globex"]["total"] == 1
    finally:
        store.close()


def test_tenants_empty_for_fresh_home(tmp_path: Path):
    """A home with zero memories returns an empty list."""
    store = make_store(tmp_path)
    try:
        assert store.tenants() == []
    finally:
        store.close()


def test_tenants_ordering_by_live_then_name(tmp_path: Path):
    """Results are ordered by live count descending, then tenant name ascending."""
    store = make_store(tmp_path)
    try:
        # Three tenants: one with 3 live, two with 1 each
        for _ in range(3):
            store.remember("x", tenant="zulu")
        store.remember("y", tenant="alpha")
        store.remember("w", tenant="beta")

        tenants_list = store.tenants()
        assert [t["tenant"] for t in tenants_list] == ["zulu", "alpha", "beta"]
    finally:
        store.close()
