"""Tests for the interactive agent REPL loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from arc_llama.agent import interactive as interactive_mod
from arc_llama.agent.interactive import InteractiveAgent


class FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class FakeAsyncClient:
    """Drop-in replacement for httpx.AsyncClient in tests."""

    def __init__(self, *, timeout: float | None = None, base_url: str = "") -> None:
        self.timeout = timeout
        self.base_url = base_url
        self.responses: list[dict[str, Any]] = []
        self.closed = False

    def set_responses(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)

    async def post(self, path: str, **kwargs: Any) -> FakeResponse:
        if not self.responses:
            raise RuntimeError("No more fake responses")
        return FakeResponse(self.responses.pop(0))

    async def aclose(self) -> None:
        self.closed = True

    @property
    def is_closed(self) -> bool:
        return self.closed


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InteractiveAgent:
    fake_client = FakeAsyncClient(base_url="http://test")
    monkeypatch.setattr(
        interactive_mod.httpx,
        "AsyncClient",
        lambda **kwargs: fake_client,
    )

    async def fake_execute(name: str, args: dict[str, Any], ctx: Any) -> Any:
        from arc_llama.agent.tools import ToolResult

        return ToolResult("ok", error=False)

    monkeypatch.setattr(interactive_mod.TOOLS, "execute", fake_execute)
    return InteractiveAgent(
        model="test-model",
        base_url="http://test",
        root=tmp_path,
        max_turns=5,
    )


async def collect_events(
    agent: InteractiveAgent,
    text: str,
    *,
    confirm_callback: Any = None,
    plan_callback: Any = None,
) -> list[dict[str, Any]]:
    return [
        e
        async for e in agent.chat(
            text, confirm_callback=confirm_callback, plan_callback=plan_callback
        )
    ]


def test_interactive_agent_conversation_persisted(agent: InteractiveAgent) -> None:
    agent._client = FakeAsyncClient(base_url="http://test")
    agent._client.set_responses(
        [
            {"choices": [{"message": {"content": "Hello!"}}]},
        ]
    )

    events = asyncio.run(collect_events(agent, "hi"))
    assert any(e["type"] == "assistant" and e["content"] == "Hello!" for e in events)
    assert any(e["type"] == "done" for e in events)
    assert len(agent.conversation) > 2


def test_interactive_agent_tool_loop(agent: InteractiveAgent) -> None:
    agent._client = FakeAsyncClient(base_url="http://test")
    agent._client.set_responses(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "readme.md"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {"choices": [{"message": {"content": "Done reading."}}]},
        ]
    )

    events = asyncio.run(collect_events(agent, "read the readme"))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "assistant" in types
    assert "done" in types


def test_interactive_agent_plan_mode_denied(agent: InteractiveAgent) -> None:
    agent.plan_mode = True
    agent._client = FakeAsyncClient(base_url="http://test")
    agent._client.set_responses(
        [
            {"choices": [{"message": {"content": "1. Do thing"}}]},
        ]
    )

    async def deny(_plan: str) -> bool:
        return False

    events = asyncio.run(collect_events(agent, "do thing", plan_callback=deny))
    assert any(e["type"] == "plan" for e in events)
    assert any(e["type"] == "done" for e in events)
