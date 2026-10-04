"""The store is the last line of defence for tenant ids, and it enforces none.

Audit finding C3. `validate_tenant_id` is called only by the auth layer,
`tenancy.index_name` and `tenant_schema` — and the latter two have no production
caller at all, so the *only* reachable validation in a running deployment is the
credential resolver. Everything below that reaches the store with whatever
string it was handed:

    st.remember("x", tenant="../../evil")     # persisted verbatim
    omnidb-server skill import --tenant ../../evil

The comment at the intake (`store.py:510`) asserts "tenant is a validated id
from the auth layer" — this module is that assumption's counterexample. A store
that trusts every caller above it is one refactor away from persisting an
unvalidated identifier into a column that other code is entitled to treat as a
tenant id.

The property: **any tenant id that reaches the store is validated there**, so
the guarantee does not depend on which transport happened to call it.
"""

from __future__ import annotations

import pytest
from conftest import make_store

from omnidb_server.tenancy import DEFAULT_TENANT, TenantError

# Ids the validator must refuse: traversal, separators, control characters,
# over-length, and the empty string. `.` and `..` matter on their own. Note that
# `-leading-dash` is deliberately NOT here — the tenant charset includes `-`, so
# that id is legal and a test asserting otherwise would be asserting a rule the
# codebase does not have.
BAD_TENANTS = [
    "../../evil",
    "..",
    ".",
    "a/b",
    "a\\b",
    "with space",
    "with\nnewline",
    "null\x00byte",
    "x" * 65,
    "  ",
    "!bang",
    "emoji\N{THUMBS UP SIGN}",
]


@pytest.fixture()
def st(tmp_path):
    store = make_store(tmp_path)
    yield store
    store.close()


@pytest.mark.parametrize("tenant", BAD_TENANTS)
def test_remember_refuses_an_invalid_tenant_id(st, tenant):
    with pytest.raises(TenantError):
        st.remember("payload", tenant=tenant)


@pytest.mark.parametrize("tenant", BAD_TENANTS)
def test_remember_many_refuses_an_invalid_tenant_id(st, tenant):
    """The batch path is the one that actually reaches the INSERT."""
    with pytest.raises(TenantError):
        st.remember_many([{"content": "payload", "tenant": tenant}])


def test_a_missing_tenant_in_a_batch_item_defaults_rather_than_failing(st):
    """`or "default"` is a documented defaulting rule, not a validation hole.

    Every pre-tenant caller omits the field entirely, so an absent tenant has to
    keep working; an *invalid* one has to fail. Those are different rules and
    conflating them would break every caller written before ADR-08.
    """
    assert len(st.remember_many([{"content": "no tenant field"}])) == 1
    with pytest.raises(TenantError):
        st.remember_many([{"content": "bad tenant", "tenant": "../../evil"}])


def test_a_refused_tenant_persists_nothing(st):
    """The rejection must happen before the INSERT, not be rolled back after.

    A row written and then deleted still leaks through the FTS index and the
    vector sidecar, so "raise afterwards" would not be a fix.
    """
    before = st.stats()["live"]
    with pytest.raises(TenantError):
        st.remember("payload", tenant="../../evil")
    assert st.stats()["live"] == before
    assert st.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE tenant_id LIKE '%evil%'"
    ).fetchone()[0] == 0


@pytest.mark.parametrize("tenant", ["acme", "globex", "ACME-prod_1", DEFAULT_TENANT])
def test_legal_tenant_ids_still_work(st, tenant):
    """Validation must not narrow what the store already accepts."""
    result = st.remember("payload", tenant=tenant)
    assert st.get(result["id"], tenant=tenant) is not None


def test_the_default_tenant_remains_legal(st):
    """DEFAULT_TENANT is reserved for *credentials*, not for data.

    Every pre-tenant row and every `tenant="default"` default argument writes
    under it, so the store must keep accepting it — only
    validate_credential_tenant refuses it.
    """
    assert st.remember("payload", tenant=DEFAULT_TENANT)["id"] > 0


def test_read_paths_validate_too(st):
    """Reads are the mirror image and get the same guarantee.

    An unvalidated tenant on a read is a fence that compares against nothing, so
    it is worth pinning that the store refuses rather than silently returning
    the empty set.
    """
    for call in (
        lambda: st.recall("anything", tenant="../../evil"),
        lambda: st.get(1, tenant="../../evil"),
        lambda: st.forget(1, tenant="../../evil"),
        lambda: st.stats(tenant="../../evil"),
    ):
        with pytest.raises(TenantError):
            call()


def test_error_is_a_tenant_error_not_a_generic_one(st):
    """Callers already catch TenantError specifically; keep that contract.

    StoreError would be laundered into the CLI's exit-2 taxonomy and read as a
    data fault rather than the credential/identifier fault it is.
    """
    with pytest.raises(TenantError) as excinfo:
        st.remember("payload", tenant="../../evil")
    assert "characters outside" in str(excinfo.value) or "non-empty" in str(excinfo.value)


def test_the_error_names_the_offending_character(st):
    """The message has to say which character was wrong, or it is not actionable."""
    with pytest.raises(TenantError) as excinfo:
        st.remember("payload", tenant="../../evil")
    message = str(excinfo.value)
    assert "'.'" in message and "'/'" in message, message

class TestCLITenantFlag:
    """`--tenant` is argv: exactly as untrusted as any other caller.

    `_tenant_of` used to be `args.tenant or DEFAULT_TENANT` — a pass-through.
    The store validates now, so this is defence in depth, but it matters that
    the operator sees the failure at the command line rather than as a failure
    partway through a skill import.
    """

    def _run(self, tmp_path, *argv):
        import subprocess
        import sys
        return subprocess.run(
            [sys.executable, "-m", "omnidb_server.cli",
             "--home", str(tmp_path), *argv],
            capture_output=True, text=True, timeout=120,
        )

    def test_skill_list_refuses_a_traversing_tenant(self, tmp_path):
        result = self._run(tmp_path, "skill", "list", "--tenant", "../../evil")
        assert result.returncode != 0
        assert "characters outside" in (result.stderr + result.stdout), \
            f"expected an actionable message, got: {result.stderr}"

    def test_skill_list_still_works_without_a_tenant_flag(self, tmp_path):
        result = self._run(tmp_path, "skill", "list")
        assert result.returncode == 0, result.stderr
