"""F11/F23 - the tenant a tool runs under must be the authenticated one.

Two defects, one shared root cause: the tenant value came from a helper that defaulted
instead of failing, and the export/import path never threaded a tenant at all.

F11  `server.py:_request_tenant` returned the default tenant on ImportError, on a
     missing token, and on a token carrying no tenant claim. The correct
     implementation, `http_server.current_tenant`, already existed and raised - it was
     simply not the one `attach_tools` called. The tool surface now takes an injected
     resolver, so the transport decides whether "no credential" is legal. stdio has no
     token and passes the single-tenant default; streamable HTTP passes the strict one.

F23  `portability.export_jsonl` built its WHERE clause with no tenant predicate, so
     exporting a multi-tenant home wrote every tenant's rows to one file.
     `import_jsonl` omitted tenant_id from its INSERT and took the column default, so
     every imported row silently collapsed into `default`. Both are tenant-scoped now.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3

import pytest
from conftest import make_store

from omnidb_server.portability import (
    _aliased_where,
    _build_memory_where,
    export_jsonl,
    import_jsonl,
)
from omnidb_server.server import _stdio_tenant, attach_tools
from omnidb_server.tenancy import TenantError

_TOOL_ARGS = {
    "remember": {"content": "x"},
    "recall": {"query": "x"},
    "search": {"sql": "SELECT 1"},
    "stats": {},
}


class _FakeMCP:
    """Captures the tool surface without needing the real mcp package."""

    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_a, **_k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


# --- F11: the injected resolver is what the tool body actually calls ----------


def test_stdio_is_the_default_resolver():
    """stdio carries no token, so its single tenant is the correct answer."""
    assert _stdio_tenant() == "default"


def test_attach_tools_registers_the_whole_surface(store):
    mcp = _FakeMCP()
    attach_tools(mcp, store)

    # A named set rather than a bare count: the old `len(...) == 10` stopped
    # detecting an accidental removal the moment the surface legitimately grew
    # (the unified `query` tool did exactly that). A named set catches both an
    # addition and a removal, and the eleven names are legible.
    assert {t for t in mcp.tools.keys()} == {
        "remember", "recall", "search", "schema", "related", "update_memory",
        "forget", "forget_many", "consolidate", "stats", "query",
    }, "the shared surface drifted"


@pytest.mark.parametrize("tool", ["stats", "remember", "recall", "search"])
def test_a_raising_resolver_stops_the_tool_running_unscoped(store, tool):
    """The F11 regression: a strict resolver must not be downgraded to a default."""
    mcp = _FakeMCP()

    def strict():
        raise TenantError("access token carries no tenant claim")

    attach_tools(mcp, store, tenant_resolver=strict)

    with pytest.raises(TenantError):
        asyncio.run(mcp.tools[tool](**_TOOL_ARGS[tool]))


def test_the_fail_open_helper_is_gone():
    """The defaulting helper must not survive, or it becomes the next regression."""
    from omnidb_server import server

    assert not hasattr(server, "_request_tenant")
    assert "resolve_tenant()" in inspect.getsource(server.attach_tools)


def test_the_strict_http_resolver_is_wired_at_the_http_transport():
    """run_http must pass current_tenant rather than inherit the stdio default."""
    from omnidb_server import http_server

    assert "tenant_resolver=current_tenant" in inspect.getsource(
        http_server._register_tools
    )
    assert http_server.current_tenant.__defaults__ == (None,), (
        "current_tenant must default to raising, not to a tenant"
    )


# --- F23: export and import stay inside one tenant ---------------------------


@pytest.fixture()
def two_tenants(tmp_path):
    """Two tenants with a real edge, so the aliased edge JOIN is exercised."""
    store = make_store(tmp_path)
    a = store.remember("acme secret roadmap", tenant="acme")
    b = store.remember("acme secret budget", tenant="acme")
    store.remember("globex secret pricing", tenant="globex")
    store.conn.execute(
        "INSERT OR IGNORE INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, ?)",
        (a["id"], b["id"], "relates", 1),
    )
    store.conn.commit()
    yield store
    try:
        store.close()
    except sqlite3.ProgrammingError:
        pass


def test_export_does_not_leak_another_tenant(two_tenants, tmp_path):
    """No tenant argument means the default tenant, which holds nothing here, so
    the export must be empty rather than quietly containing every tenant."""
    out = tmp_path / "default.jsonl"
    report = export_jsonl(two_tenants, out)
    body = out.read_text(encoding="utf-8")

    assert report["memories"] == 0
    assert report["edges"] == 0
    assert body.strip() == ""
    assert "acme secret" not in body
    assert "globex secret" not in body


def test_export_can_target_a_specific_tenant(two_tenants, tmp_path):
    out = tmp_path / "acme.jsonl"
    report = export_jsonl(two_tenants, out, tenant="acme")
    body = out.read_text(encoding="utf-8")

    assert report["memories"] == 2
    assert "acme secret" in body
    assert "globex secret" not in body


def test_edges_are_fenced_too(two_tenants, tmp_path):
    """The edge JOIN reuses the memory clause under two aliases, so tenant_id
    must be among the columns _aliased_where prefixes - otherwise the JOIN
    raises "ambiguous column name" or, worse, goes unfenced."""
    clause, params = _build_memory_where({"tenant": "acme"})
    assert "tenant_id = ?" in clause
    assert params[0] == "acme"
    for alias in ("s", "d"):
        aliased = _aliased_where(clause, alias)
        assert f"{alias}.tenant_id = ?" in aliased
        assert aliased.count("?") == clause.count("?") == 1, (
            "the bound param must survive aliasing"
        )

    out = tmp_path / "e.jsonl"
    report = export_jsonl(two_tenants, out, tenant="acme")
    kinds = [json.loads(line)["type"] for line in out.read_text().splitlines() if line]
    assert report["edges"] == 1
    assert kinds.count("edge") == 1


def test_import_stamps_the_tenant_instead_of_inheriting_the_default(
    two_tenants, tmp_path
):
    dump = tmp_path / "acme.jsonl"
    export_jsonl(two_tenants, dump, tenant="acme")
    two_tenants.close()

    dst = make_store(tmp_path / "restored")
    try:
        import_jsonl(dst, dump, tenant="acme")
        rows = dst.conn.execute(
            "SELECT tenant_id, count(*) FROM memories GROUP BY tenant_id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [("acme", 2)]
    finally:
        dst.close()


def test_single_tenant_round_trip_is_unaffected(tmp_path):
    """The only deployment shape that shipped must behave exactly as before."""
    src = make_store(tmp_path / "src")
    src.remember("plain default row")
    dump = tmp_path / "d.jsonl"
    export_jsonl(src, dump)
    src.close()

    dst = make_store(tmp_path / "dst")
    try:
        report = import_jsonl(dst, dump)
        assert report["memories"] == 1
        stored = [r[0] for r in dst.conn.execute("SELECT content FROM memories")]
        assert stored == ["plain default row"]
    finally:
        dst.close()
