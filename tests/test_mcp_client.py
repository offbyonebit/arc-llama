"""Tests for MCP client manager."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import arc_llama.agent.mcp_client as mcp_mod
from arc_llama.agent.mcp_client import MCPClientManager, MCPServerConfig
from arc_llama.agent.tools import TOOLS


@pytest.mark.asyncio
async def test_mcp_manager_raises_without_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr(mcp_mod, "_MCP_AVAILABLE", False)
    monkeypatch.setattr(mcp_mod, "_MCP_IMPORT_ERROR", ImportError("missing mcp"), raising=False)
    manager = MCPClientManager([MCPServerConfig(name="test", command="echo")])
    with pytest.raises(RuntimeError, match="mcp"):
        await manager.start()


@pytest.mark.asyncio
async def test_failed_mcp_initialization_unwinds_session_and_stdio(monkeypatch) -> None:
    exits = []

    class Stdio:
        async def __aenter__(self):
            return "read", "write"

        async def __aexit__(self, *args):
            exits.append("stdio")

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def initialize(self):
            raise RuntimeError("handshake failed")

        async def __aexit__(self, *args):
            exits.append("session")

    monkeypatch.setattr(mcp_mod, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_mod, "StdioServerParameters", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(mcp_mod, "stdio_client", lambda _params: Stdio(), raising=False)
    monkeypatch.setattr(mcp_mod, "ClientSession", Session, raising=False)
    manager = MCPClientManager([MCPServerConfig(name="test", command="unused")])
    with pytest.raises(RuntimeError, match="handshake failed"):
        await manager.start()
    assert exits == ["session", "stdio"]
    assert manager._sessions == []
    assert manager._clients == []


# ---------------------------------------------------------------------------
# Discovery and results against a scripted session (cases from a public MCP
# conformance study: pagination, tool errors, non-text content, names).
# ---------------------------------------------------------------------------

def _tool(name):
    return SimpleNamespace(name=name, description=f"synthetic {name}", inputSchema={"type": "object", "properties": {}})


def _install_session(monkeypatch, pages, call_result=None, calls=None):
    class Stdio:
        async def __aenter__(self):
            return "read", "write"

        async def __aexit__(self, *args):
            pass

    class Session:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            assert self.kwargs.get("read_timeout_seconds"), "requests must be bounded"

        async def list_tools(self, cursor=None):
            return pages[cursor]

        async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
            if calls is not None:
                calls.append((name, read_timeout_seconds))
            return call_result

    monkeypatch.setattr(mcp_mod, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_mod, "StdioServerParameters", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(mcp_mod, "stdio_client", lambda _params: Stdio(), raising=False)
    monkeypatch.setattr(mcp_mod, "ClientSession", Session, raising=False)


@pytest.mark.asyncio
async def test_all_pages_of_tools_are_registered_and_cursor_loops_stop(monkeypatch) -> None:
    pages = {
        None: SimpleNamespace(tools=[_tool("a"), _tool("b")], nextCursor="c1"),
        "c1": SimpleNamespace(tools=[_tool("c"), _tool("d")], nextCursor="c2"),
        "c2": SimpleNamespace(tools=[_tool("e")], nextCursor="c1"),  # repeats: stop instead of looping forever
    }
    _install_session(monkeypatch, pages)
    manager = MCPClientManager([MCPServerConfig(name="pg", command="unused")])
    await manager.start()
    try:
        assert sorted(manager._registered_tools) == [f"mcp_pg_{n}" for n in "abcde"]
    finally:
        await manager.stop()
    assert TOOLS.get("mcp_pg_a") is None


@pytest.mark.asyncio
async def test_duplicate_and_empty_tool_names_are_skipped(monkeypatch) -> None:
    pages = {None: SimpleNamespace(tools=[_tool("same"), _tool("same"), _tool("")], nextCursor=None)}
    _install_session(monkeypatch, pages)
    manager = MCPClientManager([MCPServerConfig(name="dup", command="unused")])
    await manager.start()
    try:
        assert manager._registered_tools == ["mcp_dup_same"]
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_tool_errors_stay_errors_and_binary_content_is_summarised(monkeypatch) -> None:
    calls: list = []
    result = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=[
            SimpleNamespace(type="text", text="disk full"),
            SimpleNamespace(type="image", data="iVBORw0KGgo=", mimeType="image/png", text=None),
            SimpleNamespace(type="audio", data="UklGRgAAAAA=", mimeType="audio/wav", text=None),
        ],
    )
    _install_session(monkeypatch, {None: SimpleNamespace(tools=[_tool("t")], nextCursor=None)}, result, calls)
    manager = MCPClientManager([MCPServerConfig(name="res", command="unused")])
    await manager.start()
    try:
        out = await TOOLS.get("mcp_res_t").handler({}, None)
    finally:
        await manager.stop()
    assert out.error is True
    assert out.content.splitlines()[0] == "disk full"
    assert "[audio audio/wav" in out.content and "UklGR" not in out.content
    assert out.images == ("data:image/png;base64,iVBORw0KGgo=",)
    assert calls and calls[0][1] == mcp_mod.CALL_TIMEOUT


@pytest.mark.asyncio
async def test_structured_only_result_is_returned_as_json(monkeypatch) -> None:
    result = SimpleNamespace(isError=False, structuredContent={"total": 3}, content=[])
    _install_session(monkeypatch, {None: SimpleNamespace(tools=[_tool("s")], nextCursor=None)}, result)
    manager = MCPClientManager([MCPServerConfig(name="st", command="unused")])
    await manager.start()
    try:
        out = await TOOLS.get("mcp_st_s").handler({}, None)
    finally:
        await manager.stop()
    assert out.error is False and '"total": 3' in out.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,structured,duplicated",
    [
        ("echo:hi", {"result": "echo:hi"}, False),  # FastMCP wraps a str return value
        ("3", {"result": 3}, False),  # ... and a number
        ('{"a": 1}', {"a": 1}, False),  # object returned as JSON text
        ("summary only", {"total": 3}, True),  # different data: keep both
        ("echo:hi", {"result": "other"}, True),
    ],
)
async def test_structured_content_already_in_text_is_not_sent_twice(monkeypatch, text, structured, duplicated) -> None:
    result = SimpleNamespace(isError=False, structuredContent=structured,
                             content=[SimpleNamespace(type="text", text=text)])
    _install_session(monkeypatch, {None: SimpleNamespace(tools=[_tool("w")], nextCursor=None)}, result)
    manager = MCPClientManager([MCPServerConfig(name="wrap", command="unused")])
    await manager.start()
    try:
        out = await TOOLS.get("mcp_wrap_w").handler({}, None)
    finally:
        await manager.stop()
    assert out.content.startswith(text)
    assert (out.content != text) is duplicated
