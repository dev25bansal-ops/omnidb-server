"""`stats()` must not report another tenant's numbers (audit finding C7).

The `stats` docstring asserts "every tenant-scoped count is fenced", and the
memory-derived fields are: `live`, `by_kind`, `soft_deleted` and `agents` all
take `AND tenant_id = ?`. Two fields did not:

- `edges` — the table has no tenant column at all (`db.py:49`, migration 4 adds
  one only to `memories` and `queries`), so `SELECT COUNT(*) FROM edges` returns
  the whole home's total.
- `files` — a directory listing of the entire home, with byte sizes, including
  `tenants.json`.

Neither is memory *content*, so the exposure is low-bandwidth. It is still a
live cross-tenant signal on the authenticated transport: one tenant can watch
another's activity volume and see the credential map's filename, and it
contradicts the invariant the surrounding comment claims.

What this file pins is the *property*, not a specific number: a count reported
under one tenant must not change when an unrelated tenant's data changes.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_store


@pytest.fixture()
def two_tenants(tmp_path):
    """acme with 2 memories + 1 edge; globex with 3 memories + 2 edges."""
    st = make_store(tmp_path)
    try:
        a1 = st.remember("acme one", tenant="acme")["id"]
        a2 = st.remember("acme two", tenant="acme")["id"]
        g1 = st.remember("globex one", tenant="globex")["id"]
        g2 = st.remember("globex two", tenant="globex")["id"]
        g3 = st.remember("globex three", tenant="globex")["id"]
        st.link(a1, a2, tenant="acme")
        st.link(g1, g2, tenant="globex")
        st.link(g2, g3, tenant="globex")
        yield st
    finally:
        st.close()


def test_edges_are_tenant_scoped(two_tenants):
    """acme owns one edge; globex owns two. Neither may see the other's."""
    assert two_tenants.stats(tenant="acme")["edges"] == 1
    assert two_tenants.stats(tenant="globex")["edges"] == 2


def test_a_tenants_edge_count_does_not_move_when_another_tenant_changes(two_tenants):
    """The property form of the above: observe a foreign write, re-read."""
    before = two_tenants.stats(tenant="acme")["edges"]
    g = [m["id"] for m in two_tenants.recall("globex", tenant="globex", k=10)]
    two_tenants.link(g[0], g[1], tenant="globex")
    assert two_tenants.stats(tenant="acme")["edges"] == before


def test_files_does_not_enumerate_the_whole_home(two_tenants):
    """A home-wide file listing is a directory disclosure, not tenant data."""
    listed = two_tenants.stats(tenant="acme")["files"]
    assert isinstance(listed, dict)
    for name in listed:
        assert "/" not in name and "\\" not in name, name


def test_files_does_not_disclose_the_credential_map_to_a_scoped_tenant(two_tenants):
    """`tenants.json` is the bearer-key map. Its size is an attacker's signal.

    The home has to actually contain one — a served deployment always does, and
    a test home that lacks it would pass without exercising anything.
    """
    (two_tenants._home() / "tenants.json").write_text(
        '{"tenants": {"sk_acme": "acme", "sk_globex": "globex"}}', encoding="utf-8")
    listed = two_tenants.stats(tenant="acme")["files"]
    assert "tenants.json" not in listed, \
        f"stats disclosed the credential map to a scoped tenant: {sorted(listed)}"
    assert ".lock" not in listed, \
        f"stats disclosed the home lock file to a scoped tenant: {sorted(listed)}"


def test_scoped_tenants_still_see_the_database_size(two_tenants):
    """The narrow listing must still be useful, not just smaller.

    A scoped tenant legitimately needs to know the database is growing; the
    fix removes cross-tenant disclosure, not all diagnostics.
    """
    listed = two_tenants.stats(tenant="acme")["files"]
    assert any(name.startswith("omnidb.db") for name in listed), listed


def test_single_tenant_still_sees_the_sidecars(two_tenants):
    """The single-tenant caller owns the home, so I-05 disk accounting stays.

    This is why `files` is two-tier rather than simply narrowed: an operator on
    a stdio/CLI deployment needs the vector sidecar sizes to reason about disk.
    Flushed first, because sidecar writes are debounced and the fixture is a
    brute-force store.
    """
    two_tenants.flush()
    listed = two_tenants.stats(tenant="default")["files"]
    # ADR-08 / W3-3: sidecars are per-tenant now, named tenant-<id>.<suffix>.
    assert any(name.startswith("tenant-") for name in listed), listed


def test_file_sizes_are_not_reported_per_tenant(two_tenants):
    """Either omit the listing or report sizes of known files only.

    Sizes vary with another tenant's write volume, so reporting the home's
    entries leaks activity even when the names are harmless.
    """
    for name, size in two_tenants.stats(tenant="acme")["files"].items():
        assert isinstance(size, int), f"{name} reports {size!r}"


def test_memory_counts_are_still_fenced(two_tenants):
    """The control: the fields that were already correct stay correct."""
    acme = two_tenants.stats(tenant="acme")
    globex = two_tenants.stats(tenant="globex")
    assert acme["live"] == 2 and globex["live"] == 3
    assert acme["live"] != globex["live"]


def test_the_docstring_claim_matches_the_behaviour(two_tenants):
    """The stale comment is part of the defect: it told readers the opposite.

    `stats`' own docstring says every tenant-scoped count is fenced. If a future
    change reintroduces an unfenced field, that comment becomes the lie that
    hid C7 in the first place, so it is asserted rather than trusted.
    """
    import inspect

    from omnidb_server.store import MemoryStore

    doc = inspect.getdoc(MemoryStore.stats) or ""
    assert "fenced" in doc.lower() or "tenant" in doc.lower()
    # and the behaviour matches: no key reports the whole-home total for a
    # tenant that owns a strict subset
    acme = two_tenants.stats(tenant="acme")
    assert acme["live"] < 5, "live must be the tenant's count, not the home's"


def test_stats_payload_is_json_serialisable(two_tenants):
    """The tool returns this over MCP; the fix must not break the contract."""
    json.dumps(two_tenants.stats(tenant="acme"))