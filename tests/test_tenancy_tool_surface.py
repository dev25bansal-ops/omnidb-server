"""F9 - the enforcement test that the docs cite but which did not exist.

tests/test_tenancy_auth.py states that this file structurally enforces that no
tool in the MCP surface accepts a tenant argument. That is the load-bearing half
of the tenancy property: a client must not be able to name its own tenant. If a
tool grows a tenant parameter, every other test in the suite keeps passing -
behavioural tests call tools through the resolver, never with a client-supplied
tenant - so without this the property is asserted nowhere.

Named for the citation in test_tenancy_auth.py:9-10.
"""

from __future__ import annotations

import inspect

import pytest

from conftest import make_store

from omnidb_server.server import attach_tools


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_a, **_k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def surface(tmp_path):
    st = make_store(tmp_path)
    mcp = _FakeMCP()
    attach_tools(mcp, st)
    yield mcp.tools
    st.close()


def test_the_surface_is_not_empty(surface):
    """A vacuous structural test is worse than none: it would pass against a
    registration that had silently stopped registering anything."""
    assert len(surface) >= 10, f"only {len(surface)} tools registered"
    assert {"remember", "recall", "consolidate"} <= set(surface)


def test_no_tool_accepts_a_tenant_argument(surface):
    """THE F9 PROPERTY. A tenant parameter on any tool is a client choosing
    its own identity, which is exactly what ADR-08 exists to prevent."""
    offenders: dict[str, list[str]] = {}
    for name, fn in surface.items():
        params = inspect.signature(fn).parameters
        for banned in ("tenant", "tenant_id", "acting_tenant", "as_tenant"):
            if banned in params:
                offenders.setdefault(name, []).append(banned)
    assert not offenders, (
        "tool surface exposes a client-controlled tenant: %s" % offenders
    )
