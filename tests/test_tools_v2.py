"""MCP tool surface v2: new agent_id / trigger_patterns / reranker parameters
must appear in the advertised tool schemas."""

import asyncio

import pytest

from omnidb_server.server import create_mcp

pytest.importorskip("mcp")


class _NoopStore:
    def remember(self, *args, **kwargs):
        return {}

    def recall(self, *args, **kwargs):
        return []


def test_tool_schemas_expose_v2_params():
    mcp = create_mcp(_NoopStore())
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}

    remember_props = set(tools["remember"].inputSchema["properties"])
    assert {"agent_id", "trigger_patterns"} <= remember_props

    recall_props = set(tools["recall"].inputSchema["properties"])
    assert {"agent_id", "reranker"} <= recall_props


def test_tool_docstrings_mention_scoping_and_rerank():
    mcp = create_mcp(_NoopStore())
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}

    assert "agent" in tools["remember"].description.lower()
    assert "trigger_patterns" in tools["remember"].description
    assert "agent" in tools["recall"].description.lower()
    assert "REGISTRY" in tools["recall"].description
