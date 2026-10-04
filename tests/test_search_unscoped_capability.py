"""The raw-SQL `search` escape hatch must be gated on a *capability*, not a
tenant *name* (ADR-08, audit finding C2).

The defect this file exists to prevent: `MemoryStore.search` used to allow raw
SQL whenever `tenant == "default"`, and `default` is a legal tenant id in a key
map. So an operator who registered a credential under the most natural name —
the same id every single-tenant stdio deployment already uses — handed that
credential an unscoped read of every tenant's rows. That is precisely the
cross-tenant leak ADR-08 exists to prevent.

The property under test: **raw SQL is available to the single-tenant transport
and to nobody else.** The single-tenant path is identified by the resolver
that produced the tenant, not by what the tenant happens to be called, because
the name is operator-controlled and the capability is not.

Deliberately behavioural: each test drives the real `MemoryStore` and asserts
what comes back, not which branch of the source was taken.
"""

from __future__ import annotations

import pytest
from conftest import make_store

from omnidb_server.store import StoreError
from omnidb_server.tenancy import (
    DEFAULT_TENANT,
    TenantError,
    TenantKey,
    TenantRegistry,
    hash_key,
    load_registry,
)

_SELECT_ALL = "SELECT tenant_id, content FROM memories"


@pytest.fixture()
def two_tenants(tmp_path):
    """One home holding one acme row and one globex row, plus a default row.

    A home with more than one tenant is the precondition for a cross-tenant
    read to mean anything; on a single-tenant home there is nothing to cross
    into and the test would pass for the wrong reason.
    """
    st = make_store(tmp_path)
    try:
        st.remember("ACME confidential roadmap", tenant="acme")
        st.remember("GLOBEX confidential pricing", tenant="globex")
        st.remember("a plain local note", tenant=DEFAULT_TENANT)
        yield st
    finally:
        st.close()


def _tenants_seen(rows) -> set[str]:
    return {r[0] for r in rows}


# --- the defect: a credential whose tenant is named "default" ----------------

def test_search_refuses_a_tenant_named_default(two_tenants):
    """A tenant id alone must never unlock unscoped SQL.

    This is the regression that failed before: the caller is an ordinary scoped
    tenant whose id happens to equal the reserved single-tenant name. It must be
    refused for the same reason any other scoped tenant is refused.
    """
    with pytest.raises(StoreError, match="search is unavailable"):
        two_tenants.search(_SELECT_ALL, tenant=DEFAULT_TENANT)


def test_search_refuses_every_scoped_tenant_including_default(two_tenants):
    """`default` is refused identically to acme and globex — no special case."""
    for tenant in ("acme", "globex", DEFAULT_TENANT, "anything-else"):
        with pytest.raises(StoreError, match="search is unavailable"):
            two_tenants.search(_SELECT_ALL, tenant=tenant)


def test_a_credential_mapped_to_default_cannot_read_other_tenants(two_tenants):
    """The original exploit, end to end, with both defences in place.

    The exploit needed two independent weaknesses: a credential could resolve
    to the reserved name, and a name-based store check granted that name raw
    SQL. Fixing either closes it. Now the registry refuses to build at all, so
    the exploit cannot reach the store — but the store's gate is checked here
    too, because the store must be safe on its own if it is ever called
    directly (its own API, a test, a future transport).
    """
    # Defence 1 — the registry refuses to authenticate anyone as `default`.
    with pytest.raises(TenantError, match="default"):
        TenantRegistry(keys=(TenantKey(digest=hash_key("sk_live"), tenant=DEFAULT_TENANT),))

    # Defence 2 — the store gates on the capability, not the name, so even a
    # hand-passed `default` tenant gets nothing.
    with pytest.raises(StoreError, match="search is unavailable"):
        two_tenants.search(_SELECT_ALL, tenant=DEFAULT_TENANT)


def test_registry_still_refuses_to_register_default_as_a_key_tenant():
    """The credential layer must reject the reserved name as a tenant.

    Defence 1 of the two above, pinned on its own: the id is legal in the data
    model, but it must not be *grantable to a credential*, because that is what
    turns a name into an escalation.
    """
    with pytest.raises(TenantError, match="default"):
        TenantRegistry(keys=(TenantKey(digest=hash_key("sk_live"), tenant=DEFAULT_TENANT),))


def test_load_registry_rejects_a_default_tenant_in_the_key_map(tmp_path):
    """The startup path must refuse the same map the registry refuses.

    An operator editing `tenants.json` should learn about this at startup, not
    have the server come up in a state where one credential outranks the rest.
    """
    (tmp_path / "tenants.json").write_text(
        '{"tenants": {"sk_live": "default"}}', encoding="utf-8")
    with pytest.raises(TenantError, match="default"):
        load_registry(tmp_path)


def test_the_key_error_names_the_fix(tmp_path):
    """The rejection has to be actionable, or operators route around it."""
    (tmp_path / "tenants.json").write_text(
        '{"tenants": {"sk_live": "default"}}', encoding="utf-8")
    with pytest.raises(TenantError) as excinfo:
        load_registry(tmp_path)
    message = str(excinfo.value)
    assert "tenants.json" in message or "tenant" in message
    assert DEFAULT_TENANT in message


# --- the capability: the single-tenant path keeps its escape hatch -----------

def test_search_works_for_a_capability_granted_caller(two_tenants):
    """The fix must not brick the feature it protects.

    `search` exists for single-tenant CLI and stdio use, where the whole home
    really is one trust domain. A caller presenting that capability gets SQL.
    """
    two_tenants.search(_SELECT_ALL, tenant=DEFAULT_TENANT, unscoped=True)
    assert _tenants_seen(two_tenants.search(_SELECT_ALL, tenant=DEFAULT_TENANT,
                                            unscoped=True)["rows"]) == {
        "acme", "globex", DEFAULT_TENANT}


def test_the_capability_is_not_a_name_the_caller_can_assert_for_itself(two_tenants):
    """`unscoped` is a keyword-only capability, not part of the tenant contract.

    A caller cannot begin granting it by accident or by argument position: the
    keyword-only marker makes the old positional call a TypeError. The real
    guarantee is upstream anyway — only the transport sets this flag, and the
    HTTP transport never does — but the signature should not be the weak link.
    """
    with pytest.raises(TypeError):
        two_tenants.search(_SELECT_ALL, DEFAULT_TENANT, True)  # type: ignore[misc]