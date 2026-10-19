from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from arc_llama.agent.harness import hermes as adapter

FAKE = Path(__file__).parent / "fixtures" / "fake_hermes.py"


@pytest.fixture(autouse=True)
def stop_fake_without_host_signals(monkeypatch):
    async def stop(proc):
        if proc.stdin:
            proc.stdin.close()
        await asyncio.wait_for(proc.wait(), 3)

    monkeypatch.setattr(adapter, "_stop", stop)


@pytest.fixture
def binary(tmp_path):
    target = tmp_path / "hermes-test"
    target.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    target.chmod(0o755)
    return str(target)


async def run(tmp_path, binary, scenario="success", approved=False):
    (tmp_path / ".scenario").write_text(scenario)
    seen = []

    async def confirm(*args):
        seen.append(args)
        return approved

    events = [
        e
        async for e in adapter.run_hermes(
            task="Make a change",
            model="arc-test",
            base_url="http://127.0.0.1:1",
            root=tmp_path,
            binary=binary,
            confirm_callback=confirm,
        )
    ]
    return events, seen


@pytest.mark.asyncio
async def test_hermes_streams_reasoning_and_text_and_cleans_up(tmp_path, binary):
    events, _ = await run(tmp_path, binary)
    assert (
        "".join(e["content"] for e in events if e["type"] == "assistant_delta")
        == "Hello from Hermes."
    )
    assert {"type": "activity", "phase": "thinking"} in events
    assert not any(e["type"] == "error" for e in events)
    assert events[-1] == {"type": "done"}
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert not Path(observed["home"]).exists()
    assert not Path(f"/proc/{observed['pid']}").exists()
    endpoint = urlparse(observed["config"]["model"]["base_url"])
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", endpoint.port)) != 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("approved", "scenario"), [(True, "approve"), (False, "deny")])
async def test_hermes_edits_wait_for_single_use_approval(tmp_path, binary, approved, scenario):
    events, seen = await run(tmp_path, binary, scenario, approved)
    assert len(seen) == 1 and seen[0][:2] == ("write-1", "write_file")
    assert (tmp_path / "answer.txt").exists() is approved
    permission = json.loads((tmp_path / "permission.json").read_text())
    assert permission["result"]["outcome"] == {
        "outcome": "selected",
        "optionId": "one" if approved else "no",
    }
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["id"] == "write-1" and result["error"] is not approved


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["outside", "persistent_only"])
async def test_hermes_does_not_grant_outside_edits_or_persistent_permissions(
    tmp_path, binary, scenario
):
    events, seen = await run(tmp_path, binary, scenario, True)
    assert seen == [] and not (tmp_path.parent / "outside.txt").exists()
    assert not (tmp_path / "answer.txt").exists()
    assert not any(e["type"] == "confirm_required" for e in events)


@pytest.mark.asyncio
async def test_hermes_rejects_unsupported_client_rpc(tmp_path, binary):
    await run(tmp_path, binary, "unsupported")
    reply = json.loads((tmp_path / "unsupported.json").read_text())
    assert reply["error"]["code"] == -32601
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["malformed", "error", "early_exit", "incomplete"])
async def test_hermes_protocol_failure_is_visible(tmp_path, binary, scenario):
    events, _ = await run(tmp_path, binary, scenario)
    assert any(e["type"] == "error" and "Hermes" in e["message"] for e in events)
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_hermes_cancel_reaps_process_and_removes_home(tmp_path, binary):
    (tmp_path / ".scenario").write_text("hang")

    async def confirm(*args):
        return False

    stream = adapter.run_hermes(
        task="Work",
        model="arc-test",
        base_url="http://127.0.0.1:1",
        root=tmp_path,
        binary=binary,
        confirm_callback=confirm,
    )
    async for event in stream:
        if event["type"] == "activity":
            break
    await stream.aclose()
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert not Path(f"/proc/{observed['pid']}").exists()
    assert not Path(observed["home"]).exists()


@pytest.mark.asyncio
async def test_hermes_startup_timeout_cleans_up(tmp_path, binary, monkeypatch):
    monkeypatch.setattr(adapter, "STARTUP_TIMEOUT", 0.15)
    events, _ = await run(tmp_path, binary, "startup_timeout")
    assert any(e["type"] == "error" and "time" in e["message"] for e in events)
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert not Path(observed["home"]).exists()


@pytest.mark.asyncio
async def test_hermes_child_does_not_inherit_credentials_or_approval_bypasses(
    tmp_path, binary, monkeypatch
):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HTTP_PROXY",
        "PYTHONPATH",
        "HERMES_YOLO_MODE",
    ):
        monkeypatch.setenv(key, "must-not-inherit")
    events, _ = await run(tmp_path, binary)
    assert not any(e["type"] == "error" for e in events)
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert "must-not-inherit" not in json.dumps(observed)
    config = observed["config"]
    assert config["model"]["provider"] == "custom" and config["model"]["default"] == "arc-test"
    assert config["fallback_model"] == [] and config["approvals"]["mode"] == "manual"
    assert all(
        value["base_url"] == config["model"]["base_url"] for value in config["auxiliary"].values()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{"auto_confirm": True}, {"local_only": False}, {"base_url": "https://cloud.example"}],
)
async def test_hermes_rejects_unsupported_execution_modes(tmp_path, binary, options):
    async def confirm(*args):
        return False

    kwargs = {
        "task": "x",
        "model": "m",
        "base_url": "http://127.0.0.1:1",
        "root": tmp_path,
        "confirm_callback": confirm,
        "binary": binary,
        **options,
    }
    events = [e async for e in adapter.run_hermes(**kwargs)]
    assert events[0]["type"] == "error" and not (tmp_path / "observed.json").exists()


@pytest.mark.asyncio
async def test_missing_hermes_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.shutil, "which", lambda _: None)

    async def confirm(*args):
        return False

    events = [
        e
        async for e in adapter.run_hermes(
            task="x",
            model="m",
            base_url="http://localhost:1",
            root=tmp_path,
            confirm_callback=confirm,
        )
    ]
    assert "not installed" in events[0]["message"]


def test_hermes_unknown_session_updates_are_ignored():
    rpc = adapter._ACP(None, asyncio.Queue())
    rpc.session = "mine"
    assert (
        rpc.updates(
            {
                "sessionId": "other",
                "update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "wrong"}},
            }
        )
        == []
    )
