"""Tests for the user skill loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.agent.tools import TOOLS, ToolContext
from arc_llama.skills import load_skills


@pytest.mark.asyncio
async def test_load_skills_registers_and_executes_tool(tmp_path: Path) -> None:
    skill_file = tmp_path / "timestamp_skill.py"
    skill_file.write_text(
        """
from arc_llama.agent.tools import tool, ToolResult

@tool(
    name="test_timestamp",
    description="Return a fixed timestamp for testing.",
    parameters={"type": "object", "properties": {}},
)
def _test_timestamp(arguments, ctx):
    return ToolResult("2026-06-12T00:00:00+00:00")
"""
    )

    assert TOOLS.get("test_timestamp") is None
    try:
        load_skills(tmp_path)
        tool_def = TOOLS.get("test_timestamp")
        assert tool_def is not None
        assert tool_def.name == "test_timestamp"

        ctx = ToolContext(root=tmp_path, client=None)  # type: ignore[arg-type]
        result = await TOOLS.execute("test_timestamp", {}, ctx)
        assert not result.error
        assert result.content == "2026-06-12T00:00:00+00:00"
    finally:
        TOOLS.unregister("test_timestamp")


def test_load_skills_skips_broken_skill_and_continues(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('simulated import error')\n")

    good = tmp_path / "good.py"
    good.write_text(
        """
from arc_llama.agent.tools import tool

@tool(
    name="test_hello",
    description="Say hello.",
    parameters={"type": "object", "properties": {}},
)
def _test_hello(arguments, ctx):
    return "hello"
"""
    )

    try:
        load_skills(tmp_path)
        assert TOOLS.get("test_hello") is not None
    finally:
        TOOLS.unregister("test_hello")
