"""The per-tenant FTS migration has landed, and these assertions keep it honest.

This module was originally the opposite: it asserted that
`tenant_schema.py` was NOT wired in, and that the lexical leg still hit one
global `memory_fts` table. That guard existed because the module was built,
documented, and reachable from nowhere — its docstring explained per-tenant FTS
in detail without saying whether it happened (audit finding C14).

Schema v6 dropped the shared table, so those assertions became false and the
guard did its job: the migration had to come back here and say so. What
replaces them is the same shape — the migration's own claims, pinned so a
regression that quietly reintroduces a shared index fails loudly.

`tenant_schema.py` still carries a "NOT WIRED IN" warning from before the
migration; `test_the_warning_is_gone_now_that_it_is_wired_in` asserts that,
which will fail again if a future change re-orphans the module.
"""

from __future__ import annotations

import inspect

import pytest
from conftest import make_store

from omnidb_server import tenant_schema


def _real_fts_tables(conn) -> set[str]:
    """Per-tenant FTS tables only.

    FTS5 creates shadow tables (`fts_acme_content`, `_data`, `_idx`, `_docsize`,
    `_config`) alongside each index, so a name filter counts five per tenant.
    Only a real index carries CREATE VIRTUAL TABLE.
    """
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND sql LIKE '%CREATE VIRTUAL TABLE%'")}


def test_the_production_package_imports_tenant_schema():
    """Wired in: the write path resolves per-tenant tables through this module."""
    import ast
    import pathlib

    package = pathlib.Path(tenant_schema.__file__).parent
    importers = set()
    for path in sorted(package.rglob("*.py")):
        if path.name == "tenant_schema.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "tenant_schema"):
                importers.add(path.name)
    assert importers, (
        "no module imports tenant_schema again; the per-tenant FTS tables are "
        "now created and queried, so this is the state that should not return"
    )


def test_the_shared_table_is_gone(tmp_path):
    st = make_store(tmp_path)
    try:
        names = {r[0] for r in st.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "memory_fts" not in names
    finally:
        st.close()


def test_the_shipped_lexical_leg_queries_the_tenants_own_table(tmp_path):
    """The fence the migration replaced — still asserted, now redundantly.

    It is redundant against a per-tenant table, and that is the point: it also
    excludes soft-deleted rows via `_SELECTIVE`, which table separation does not.
    """
    st = make_store(tmp_path)
    try:
        acme_id = st.remember("acme quixotic zinnia", tenant="acme")["id"]
        globex_id = st.remember("globex unrelated pipelines", tenant="globex")["id"]
        assert acme_id in {r["id"] for r in st.recall("quixotic", k=5, tenant="acme")}
        globex_hits = {r["id"] for r in st.recall("quixotic", k=5, tenant="globex")}
        assert acme_id not in globex_hits
        assert globex_hits <= {globex_id}

        tables = _real_fts_tables(st.conn)
        assert "fts_acme" in tables and "fts_globex" in tables
        assert "memory_fts" not in tables
    finally:
        st.close()


def test_the_warning_is_gone_now_that_it_is_wired_in():
    """A "NOT WIRED IN" banner on a wired module is a lie — and worse than no doc."""
    doc = inspect.getdoc(tenant_schema) or ""
    assert "NOT WIRED IN" not in doc, (
        "tenant_schema.py still says it is not wired in, but the store resolves "
        "per-tenant FTS tables through it. Update the docstring to describe "
        "what it now does."
    )


@pytest.mark.parametrize("fn", [
    "register_and_create", "drop_tenant", "fts_table_name", "is_registered",
])
def test_the_lifecycle_api_is_still_present(fn):
    """The store registers on demand; these are the pieces it depends on."""
    assert callable(getattr(tenant_schema, fn))


def test_the_fts_table_name_is_derived_from_a_validated_tenant():
    """Identifier safety remains the load-bearing property of this module."""
    from omnidb_server.tenancy import TenantError

    assert tenant_schema.fts_table_name("acme") == '"fts_acme"'
    with pytest.raises(TenantError):
        tenant_schema.fts_table_name("../../evil")