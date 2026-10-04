"""The query log must record which tenant ran the query.

`_record_query` inserts into `queries` without listing `tenant_id`, even though
migration 4 added the column (`ALTER TABLE queries ADD COLUMN tenant_id TEXT NOT
NULL DEFAULT 'default'`). Every logged row therefore silently took the default,
so a recall issued by `acme` was recorded as belonging to `default` — verified
before this change.

Three consequences, all from one missing column in one INSERT:

- `query_stats` aggregates with no tenant predicate, so a multi-tenant home's
  per-tenant activity cannot be separated at all.
- `omnidb-server query-log` dumps every tenant's verbatim query text, including
  whatever those tenants searched for.
- `forget`/`purge` cannot remove a tenant's query history, because nothing links
  a logged query back to the tenant that wrote it.

This file pins the attribution, and pins that the *default* stays the fallback
rather than becoming an error — pre-tenant callers must keep working.
"""

from __future__ import annotations

import pytest
from conftest import make_store


@pytest.fixture()
def logging_store(tmp_path):
    """A store with the query log on (ARCH-03 makes it opt-in)."""
    st = make_store(tmp_path, query_log=True)
    yield st
    st.close()


def _logged_tenants(st) -> list[str]:
    return [r["tenant_id"] for r in st.conn.execute(
        "SELECT tenant_id FROM queries ORDER BY id")]


def test_a_recall_is_attributed_to_the_calling_tenant(logging_store):
    logging_store.remember("acme payroll note", tenant="acme")
    logging_store.recall("payroll", tenant="acme")
    assert _logged_tenants(logging_store) == ["acme"], (
        "the query log records tenant_id='default' for a query issued by "
        "'acme'; the column exists and the INSERT simply omits it"
    )


def test_two_tenants_are_distinguished(logging_store):
    logging_store.remember("acme payroll note", tenant="acme")
    logging_store.remember("globex payroll note", tenant="globex")
    logging_store.recall("payroll", tenant="acme")
    logging_store.recall("payroll", tenant="globex")
    assert _logged_tenants(logging_store) == ["acme", "globex"]


def test_the_default_tenant_is_still_recorded_as_default(logging_store):
    logging_store.remember("a plain local note")
    logging_store.recall("plain")
    assert _logged_tenants(logging_store) == ["default"]


def test_a_query_log_row_can_be_removed_with_its_tenant(logging_store):
    """The point of attribution: erasure must be able to find the rows.

    Before this change nothing linked a logged query back to the tenant that
    wrote it, so no amount of purging could remove one tenant's query history.
    """
    logging_store.remember("acme payroll note", tenant="acme")
    logging_store.recall("payroll", tenant="acme")
    conn = logging_store.conn
    assert conn.execute(
        "DELETE FROM queries WHERE tenant_id = ?", ("acme",)).rowcount == 1


def test_query_stats_can_be_scoped_to_a_tenant(logging_store):
    logging_store.remember("acme payroll note", tenant="acme")
    logging_store.remember("globex payroll note", tenant="globex")
    logging_store.recall("payroll", tenant="acme")
    logging_store.recall("payroll", tenant="globex")

    scoped = logging_store.query_stats(hours=24.0, tenant="acme")
    assert scoped["total"] == 1, scoped
    assert scoped["top_queries"][0]["query"] == "payroll"


def test_query_stats_without_a_tenant_still_aggregates_everything(logging_store):
    """The unscoped view is the single-tenant operator's; it must not regress."""
    logging_store.remember("acme payroll note", tenant="acme")
    logging_store.remember("globex payroll note", tenant="globex")
    logging_store.recall("payroll", tenant="acme")
    logging_store.recall("payroll", tenant="globex")

    assert logging_store.query_stats(hours=24.0)["total"] == 2


def test_the_search_query_is_attributed_too(logging_store):
    """`search` is only reachable single-tenant, but it logs like any other."""
    logging_store.remember("a note about invoices")
    logging_store.search("SELECT COUNT(*) FROM memories", unscoped=True)
    assert _logged_tenants(logging_store) == ["default"]


def test_a_disabled_log_still_writes_nothing(tmp_path):
    st = make_store(tmp_path, query_log=False)
    try:
        st.remember("a note")
        st.recall("note")
        assert st.conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 0
    finally:
        st.close()