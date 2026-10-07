"""Tests for MCP client manager."""
from __future__ import annotations

import pytest

import arc_llama.agent.mcp_client as mcp_mod
from arc_llama.agent.mcp_client import MCPClientManager, MCPServerConfig


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
        def __init__(self, *args):
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
