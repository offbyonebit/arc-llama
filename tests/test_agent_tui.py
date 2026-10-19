"""Smoke tests for the arcllama agent TUI."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent_tui import (
    AgentApp,
    AgentEvent,
    ApprovePlanScreen,
    ConfirmToolScreen,
    run_agent_tui,
)
from arc_llama.chat_store import ChatStore


@pytest.fixture
def app(tmp_path: Path) -> AgentApp:
    chats = ChatStore(tmp_path / "chats")
    checkpoints = CheckpointStore(tmp_path / "checkpoints")
    return AgentApp(
        base_url="http://test",
        model="test-model",
        root=tmp_path,
        folder="work",
        chat_store=chats,
        checkpoint_store=checkpoints,
    )


async def test_agent_app_mounts_widgets(app: AgentApp) -> None:
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#chat-log") is not None
        assert pilot.app.query_one("#chat-input") is not None
        assert pilot.app.query_one("#send-btn") is not None


def test_confirm_tool_screen_dismisses_bool() -> None:
    screen = ConfirmToolScreen("write_file", {"path": "foo.txt"})
    assert screen.tool == "write_file"


def test_approve_plan_screen_dismisses_bool() -> None:
    screen = ApprovePlanScreen("1. Do thing\n2. Done")
    assert "Do thing" in screen.plan_text


def test_agent_event_message() -> None:
    msg = AgentEvent({"type": "assistant", "content": "hello"})
    assert msg.event["type"] == "assistant"


def test_run_agent_tui_accepts_profile(monkeypatch, tmp_path: Path) -> None:
    from arc_llama import agent_tui as tui_mod
    from arc_llama.config import Config, MCPServerConfig, ProfileConfig

    cfg = Config()
    cfg.agent.profile = "work"
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", command="npx"),
        MCPServerConfig(name="gh", command="npx"),
    ]
    cfg.profiles = [ProfileConfig(name="work", mcp_servers=["fs"])]

    started_servers: list[str] = []
    stopped = False

    class FakeMCPManager:
        def __init__(self, servers):
            self.servers = servers

        async def start(self):
            started_servers.extend(s.name for s in self.servers)

        async def stop(self):
            nonlocal stopped
            stopped = True

    class FakeApp:
        def __init__(self, **kwargs):
            pass

        async def run_async(self):
            pass

    def fake_get(url, timeout=None):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "test-model"}]}

        return Resp()

    monkeypatch.setattr(tui_mod.httpx, "get", fake_get)
    monkeypatch.setattr(tui_mod, "MCPClientManager", FakeMCPManager)
    monkeypatch.setattr(tui_mod, "AgentApp", FakeApp)
    monkeypatch.setattr(tui_mod, "load_skills", lambda _dir: None)

    run_agent_tui(base_url="http://test", profile="work", config=cfg)

    assert started_servers == ["fs"]
    assert stopped is True
