"""Tests for the dynamic agent tool registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from arc_llama.agent.tools import (
    TOOLS,
    ToolContext,
    ToolRegistry,
    ToolResult,
    execute_tool,
    tool,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore the global registry so tests don't leak tools."""
    original = dict(TOOLS._tools)
    yield
    TOOLS._tools.clear()
    TOOLS._tools.update(original)


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    (tmp_path / "file.txt").write_text("hello")
    return tmp_path


def test_registry_definitions_include_core_tools() -> None:
    names = {d["function"]["name"] for d in TOOLS.definitions}
    assert names >= {"read_file", "write_file", "list_directory", "run_command", "search_files"}


def test_registry_requires_confirmation() -> None:
    assert TOOLS.requires_confirmation("write_file") is True
    assert TOOLS.requires_confirmation("run_command") is True
    assert TOOLS.requires_confirmation("read_file") is False


@pytest.mark.asyncio
async def test_registry_execute_read_file(tmp_root: Path) -> None:
    ctx = ToolContext(root=tmp_root, client=AsyncMock(spec=httpx.AsyncClient))
    result = await TOOLS.execute("read_file", {"path": "file.txt"}, ctx)
    assert result.error is False
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_registry_execute_unknown_tool() -> None:
    ctx = ToolContext(root=Path("."), client=AsyncMock(spec=httpx.AsyncClient))
    result = await TOOLS.execute("does_not_exist", {}, ctx)
    assert result.error is True
    assert "Unknown tool" in result.content


def test_tool_decorator_registers_sync_tool() -> None:
    @tool(
        name="ping",
        description="Returns pong.",
        parameters={"type": "object", "properties": {}},
    )
    def ping(arguments: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult("pong")

    assert "ping" in TOOLS.list_tools()


@pytest.mark.asyncio
async def test_tool_decorator_registers_async_tool() -> None:
    @tool(
        name="async_ping",
        description="Returns pong asynchronously.",
        parameters={"type": "object", "properties": {}},
    )
    async def async_ping(arguments: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult("pong")

    ctx = ToolContext(root=Path("."), client=AsyncMock(spec=httpx.AsyncClient))
    result = await TOOLS.execute("async_ping", {}, ctx)
    assert result.content == "pong"


def test_custom_registry_is_independent() -> None:
    reg = ToolRegistry()

    def handler(arguments: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult("ok")

    reg.register_function(
        name="custom",
        description="A custom tool.",
        parameters={"type": "object", "properties": {}},
    )(handler)

    assert "custom" in reg.list_tools()
    assert "custom" not in TOOLS.list_tools()


@pytest.mark.asyncio
async def test_execute_tool_backwards_compatible(tmp_root: Path) -> None:
    result = await execute_tool(
        "read_file",
        {"path": "file.txt"},
        tmp_root,
        AsyncMock(spec=httpx.AsyncClient),
    )
    assert result.error is False
    assert result.content == "hello"
