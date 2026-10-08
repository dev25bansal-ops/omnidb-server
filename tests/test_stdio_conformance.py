"""End-to-end MCP conformance smoke: spawn `python -m omnidb_server serve` as a
subprocess and drive it with the official MCP client over stdio — initialize
handshake, tools/list, successful tool calls, and error semantics."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

REPO = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "remember", "recall", "search", "related",
    "forget", "consolidate", "stats",
}


def _payload(result):
    structured = getattr(result, "structuredContent", None)
    data = structured if structured is not None else json.loads(result.content[0].text)
    # FastMCP wraps non-object return schemas under a single key
    if isinstance(data, dict) and set(data) == {"result"}:
        data = data["result"]
    return data


def _run_session(home: Path) -> bool:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "omnidb_server", "serve", "--home", str(home)],
        env=env,
    )

    async def inner() -> bool:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await asyncio.wait_for(session.initialize(), 30)
                assert init.serverInfo.name == "omnidb"
                # The version is ours, not the mcp SDK's. FastMCP forwards no
                # version to the low-level server, whose fallback is the
                # *SDK* version -- so this advertised e.g. "1.29.1". Asserting
                # only the name is what let that stand.
                from omnidb_server import __version__ as ours
                assert init.serverInfo.version == ours, (
                    f"serverInfo.version {init.serverInfo.version!r} "
                    f"is not OmniDB's {ours!r}")

                names = {t.name for t in (await session.list_tools()).tools}
                assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"

                rem = await asyncio.wait_for(
                    session.call_tool("remember", {"content": "stdio conformance probe"}), 30)
                assert rem.isError is not True, rem.content
                mid = _payload(rem)["id"]

                rec = await asyncio.wait_for(
                    session.call_tool("recall", {"query": "stdio conformance probe", "k": 3}), 30)
                hits = _payload(rec)
                assert hits and hits[0]["id"] == mid

                denied = await asyncio.wait_for(
                    session.call_tool("search", {"sql": "DELETE FROM memories"}), 30)
                assert denied.isError is True

                stats = await asyncio.wait_for(session.call_tool("stats", {}), 30)
                assert _payload(stats)["live"] >= 1
                return True

    return asyncio.run(asyncio.wait_for(inner(), 90))


def test_stdio_end_to_end(tmp_path):
    assert _run_session(tmp_path / "home")
