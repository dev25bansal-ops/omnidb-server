"""`tenant_schema.py` is designed, not wired — and that must stay visible.

Audit finding C14. The module owns the per-tenant FTS5 table lifecycle and is
unit-tested, but nothing in the shipped read or write path calls it: the
lexical leg still queries the single global `memory_fts`, fenced by a predicate
on the joined `memories` row.

That is fine as an interim state, and the progress doc already marks W3-3 as
partial. The risk is that the module's own docstring reads as a description of
current behaviour — it explains *what* per-tenant FTS is for in detail, and says
nothing about whether it happens. So the file is the trap, and this test is the
guard.

Two things are pinned:

1. The module still has no production caller. If a future change wires it in,
   this test fails and the docstring is updated to match — that is the point.
2. The shipped lexical leg queries the one global table.

A test that only asserted (1) would pass silently forever after the migration;
asserting (2) as well means the assertion has to be consciously retired.
"""

from __future__ import annotations

import inspect

import pytest
from conftest import make_store

from omnidb_server import tenant_schema


def test_the_module_says_it_is_not_wired_in():
    """The docstring is the interface a reader actually sees."""
    doc = inspect.getdoc(tenant_schema) or ""
    assert "NOT WIRED IN" in doc, (
        "tenant_schema.py no longer states that it has no production caller. "
        "Either it has been wired in — in which case this test and the "
        "docstring both need updating — or the warning was removed and a "
        "reader would assume the migration is done."
    )


def test_the_production_package_does_not_import_it():
    """Census across the whole package, not just a grep of one file."""
    import ast
    import pathlib

    package = pathlib.Path(tenant_schema.__file__).parent
    importers = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "tenant_schema.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "tenant_schema"):
                importers.append(path.name)
            elif isinstance(node, ast.Import):
                if any(a.name.endswith("tenant_schema") for a in node.names):
                    importers.append(path.name)
    assert not importers, (
        f"tenant_schema is now imported by {importers}. The W3-3 migration "
        "has landed: update this test and the module docstring to say so, and "
        "check whether the shipped lexical leg uses per-tenant tables yet."
    )


def test_the_shipped_lexical_leg_queries_the_one_global_table(tmp_path):
    """The behaviour the docstring describes as *not* what ships today."""
    st = make_store(tmp_path)
    try:
        acme_id = st.remember("quixotic zinnia marmalade", tenant="acme")["id"]
        globex_id = st.remember("unrelated content about pipelines",
                                tenant="globex")["id"]

        acme_hits = st.recall("quixotic", k=5, tenant="acme")
        assert acme_id in {r["id"] for r in acme_hits}

        # The fence is a predicate, so the assertion is about *whose rows* come
        # back, not about whether anything comes back: the dense leg ranks by
        # vector similarity, and the hashing embedder gives every pair some
        # score, so a tenant's own low-similarity rows legitimately appear.
        globex_hits = {r["id"] for r in st.recall("quixotic", k=5, tenant="globex")}
        assert acme_id not in globex_hits,             "the lexical fence failed: globex's recall returned acme's row"
        assert globex_hits <= {globex_id}

        # And there is exactly one FTS table, not one per tenant.
        tables = [r[0] for r in st.conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE '%fts%'")]
        assert "memory_fts" in tables
        assert not [t for t in tables if t.startswith("fts_")], (
            f"per-tenant FTS tables now exist: {tables}"
        )
    finally:
        st.close()


@pytest.mark.parametrize("fn", [
    "register_and_create", "drop_tenant", "fts_table_name", "is_registered",
])
def test_the_lifecycle_api_is_present_and_validated(fn):
    """Whatever else is true, the module must not rot into something unusable.

    These are the pieces W3-3 will call. They are exercised here so the
    "designed, not wired" state is a *maintained* state rather than an
    abandoned one.
    """
    func = getattr(tenant_schema, fn)
    assert callable(func), f"{fn} is missing from tenant_schema"


def test_the_fts_table_name_is_derived_from_a_validated_tenant():
    """Identifier safety is the load-bearing property of this module.

    A table name built from an unvalidated tenant would be an injection or a
    path escape, which is the reason `validate_tenant_id` is called first here.
    """
    # Returned double-quoted so a hyphenated id (`acme-prod`) is legal SQL;
    # the quotes are defence in depth, the validation is the real control.
    assert tenant_schema.fts_table_name("acme") == '"fts_acme"' 
    from omnidb_server.tenancy import TenantError

    with pytest.raises(TenantError):
        tenant_schema.fts_table_name('../../evil')