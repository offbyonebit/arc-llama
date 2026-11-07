from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
from helpers import python_executable

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
    return python_executable(tmp_path / "hermes-test", FAKE.read_text(encoding="utf-8"))


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
    assert {"type": "harness_session", "session_id": "hermes-session-1"} in events
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


def test_hermes_windows_environment_sets_home_and_keeps_secrets_out(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.sys, "platform", "win32")
    monkeypatch.setenv("TEMP", r"C:\Temp")
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-hermes")
    monkeypatch.setenv("APPDATA", r"C:\Users\someone\AppData\Roaming")

    env = adapter.hermes_environment(tmp_path)

    # Python on Windows reads USERPROFILE, not HOME, for the home folder.
    assert env["USERPROFILE"] == str(tmp_path)
    assert env["APPDATA"] == str(tmp_path / "appdata")
    assert env["LOCALAPPDATA"] == str(tmp_path / "localappdata")
    assert env["TEMP"] == r"C:\Temp"
    assert env["COMSPEC"] == r"C:\Windows\System32\cmd.exe"
    assert "OPENAI_API_KEY" not in env


def test_hermes_posix_environment_has_no_windows_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.sys, "platform", "linux")

    env = adapter.hermes_environment(tmp_path)

    assert env["HOME"] == str(tmp_path)
    assert "USERPROFILE" not in env and "APPDATA" not in env


@pytest.mark.skipif(adapter.sys.platform != "win32", reason="Windows home-folder lookup")
def test_hermes_environment_lets_a_windows_child_find_its_home(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-I", "-c", "import pathlib, tempfile; print(pathlib.Path.home()); tempfile.mkdtemp()"],
        env=adapter.hermes_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path)


def _gateway_record(home, **overrides):
    return json.dumps({"pid": 4242, "kind": "hermes-gateway", "argv": ["main.py", "gateway", "run"],
                       "hermes_home": str(home), **overrides})


def test_recorded_gateway_is_found_in_the_private_home(tmp_path):
    (tmp_path / "gateway.pid").write_text(_gateway_record(tmp_path))
    (tmp_path / "gateway.lock").write_text(_gateway_record(str(tmp_path).swapcase() if adapter.sys.platform == "win32" else tmp_path))

    assert adapter._recorded_gateway_pids(tmp_path) == {4242}


@pytest.mark.parametrize(
    "record",
    [
        "not json",
        json.dumps([4242]),
        json.dumps({"pid": "4242", "kind": "hermes-gateway"}),
        json.dumps({"pid": 4242, "kind": "something-else"}),
        json.dumps({"pid": 4242, "kind": "hermes-gateway"}),
    ],
)
def test_unusable_gateway_records_name_no_process(tmp_path, record):
    (tmp_path / "gateway.pid").write_text(record)

    assert adapter._recorded_gateway_pids(tmp_path) == set()


def test_gateway_serving_another_home_is_left_alone(tmp_path):
    (tmp_path / "gateway.pid").write_text(_gateway_record(tmp_path / "someone-else"))

    assert adapter._recorded_gateway_pids(tmp_path) == set()


def test_no_gateway_record_names_no_process(tmp_path):
    assert adapter._recorded_gateway_pids(tmp_path) == set()
