"""OpenCode harness adapter: event translation, config, and an end-to-end run
against a fake `opencode serve` (no real OpenCode or model needed)."""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import pytest
from helpers import python_executable

from arc_llama.agent.harness import opencode as oc
from arc_llama.agent.harness.opencode import (
    OpenCodeTranslator,
    build_config,
    find_binary,
    run_opencode,
)

FIXTURES = Path(__file__).parent / "fixtures"
FAKE = FIXTURES / "fake_opencode.py"


@pytest.fixture(autouse=True)
def _native_fake(tmp_path, monkeypatch):
    script = FIXTURES / "fake_opencode.py"
    binary = python_executable(
        tmp_path / "fake-opencode",
        f"import runpy; runpy.run_path({str(script)!r}, run_name='__main__')\n",
    )
    monkeypatch.setattr(f"{__name__}.FAKE", Path(binary))


def _events() -> list[dict]:
    return json.loads((FIXTURES / "opencode_events_bash.json").read_text())


def test_translator_maps_real_session_to_agent_events():
    sid = next(e["properties"]["sessionID"] for e in _events() if e["type"] == "session.created")
    tr = OpenCodeTranslator(sid)
    out = [o for e in _events() for o in tr.feed(e)]
    kinds = [o["type"] for o in out]
    assert kinds[0] == "status" and kinds[-1] == "done"
    call = next(o for o in out if o["type"] == "tool_call")
    result = next(o for o in out if o["type"] == "tool_result")
    assert call["name"] == "bash" and call["arguments"] == {"command": "echo arc-probe-ok"}
    assert result["id"] == call["id"] and result["content"] == "arc-probe-ok\n" and result["error"] is False
    assistant = [o["content"] for o in out if o["type"] == "assistant"]
    assert assistant == ["It printed: `arc-probe-ok`"]       # the user's own prompt is not echoed back
    assert kinds.count("tool_call") == 1 and kinds.count("tool_result") == 1   # no duplicates from updates
    assert tr.done


def test_translator_ignores_other_sessions_and_early_idle():
    tr = OpenCodeTranslator("ses_a")
    assert tr.feed({"type": "session.idle", "properties": {"sessionID": "ses_a"}}) == []      # before any work
    assert tr.feed({"type": "session.status", "properties": {"sessionID": "ses_b", "status": {"type": "busy"}}}) == []
    assert not tr.done


def test_translator_reports_tool_error_and_session_error():
    tr = OpenCodeTranslator("s")
    tr.feed({"type": "message.updated", "properties": {"info": {"id": "m", "role": "assistant", "sessionID": "s"}}})
    part = {"type": "tool", "tool": "bash", "callID": "c1", "messageID": "m", "id": "p",
            "state": {"status": "error", "input": {"command": "false"}, "error": "exit 1"}}
    out = tr.feed({"type": "message.part.updated", "properties": {"sessionID": "s", "part": part}})
    assert [o["type"] for o in out] == ["tool_call", "tool_result"]
    assert out[1]["error"] is True and out[1]["content"] == "exit 1"
    err = tr.feed({"type": "session.error", "properties": {"sessionID": "s", "error": {"name": "X", "data": {"message": "boom"}}}})
    assert err == [{"type": "error", "message": "boom"}] and tr.done


def test_build_config_pins_local_provider_and_guards_mutations():
    cfg = build_config("http://127.0.0.1:11435/", "my-model")
    assert cfg["model"] == "arc-llama/my-model"
    assert cfg["provider"]["arc-llama"]["options"]["baseURL"] == "http://127.0.0.1:11435/v1"
    assert cfg["enabled_providers"] == ["arc-llama"]                      # no silent fallback to cloud providers
    assert cfg["permission"]["edit"] == "ask" and cfg["permission"]["bash"] == "ask"
    assert {cfg["permission"][tool] for tool in ("webfetch", "websearch", "codesearch")} == {"ask"}     # web tools ask in Ask mode
    auto = build_config("http://x", "m", auto_confirm=True)["permission"]
    assert auto["edit"] == "allow"
    rules = list(auto["bash"].items())
    assert rules[0] == ("*", "allow")                               # catch-all first: the last matching rule wins
    asked = {pattern for pattern, action in rules if action == "ask"}
    assert {"sudo *", "git reset --hard*", "rm -r*"} <= asked      # destructive commands still ask
    assert "enabled_providers" not in build_config("http://x", "m", local_only=False)


def test_find_binary(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_LLAMA_OPENCODE_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_binary() is None
    monkeypatch.setenv("ARC_LLAMA_OPENCODE_BIN", str(FAKE))
    assert find_binary() == str(FAKE)


@pytest.fixture(autouse=True)
def _stop_fake_without_signals(monkeypatch):
    """The suite forbids real signals; the fake exits on request instead."""

    async def stop(self):
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(self.url + "/__exit", timeout=2)
        await asyncio.wait_for(proc.wait(), 5)

    terminated = []

    def terminate_now(self):
        # the suite forbids real signals; the fake exits on request, and we record that it was asked
        terminated.append(self.url)
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(self.url + "/__exit", timeout=2)

    monkeypatch.setattr(oc._Server, "stop", stop)
    monkeypatch.setattr(oc._Server, "terminate_now", terminate_now)
    monkeypatch.terminated = terminated


async def _run(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(tmp_path / "log.jsonl"))
    events = []
    async for e in run_opencode("run it", "m", "http://127.0.0.1:11435", tmp_path, binary=str(FAKE), timeout=30, **kwargs):
        events.append(e)
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return events, log


async def test_run_end_to_end_auto_confirm(tmp_path, monkeypatch):
    events, log = await _run(tmp_path, monkeypatch, auto_confirm=True)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status" and kinds[-1] == "done" and "error" not in kinds
    assert [e["name"] for e in events if e["type"] == "tool_call"] == ["bash"]
    boot = log[0]
    assert boot["argv"][:1] == ["serve"] and "127.0.0.1" in boot["argv"]
    assert Path(boot["cwd"]).resolve() == tmp_path.resolve()
    cfg = json.loads(boot["env"]["OPENCODE_CONFIG_CONTENT"])
    assert cfg["enabled_providers"] == ["arc-llama"] and boot["env"]["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    prompt = next(e for e in log if "prompt_async" in e.get("path", ""))
    assert prompt["body"]["model"] == {"providerID": "arc-llama", "modelID": "m"} and prompt["auth"]
    assert any(e.get("path", "").endswith("/abort") for e in log)         # always cleans the session up


@pytest.mark.parametrize("approve", [True, False])
async def test_run_routes_permission_through_confirm_callback(tmp_path, monkeypatch, approve):
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", "1")
    asked = []

    async def confirm(call_id, tool, arguments):
        asked.append((call_id, tool, arguments))
        return approve

    events, log = await _run(tmp_path, monkeypatch, confirm_callback=confirm)
    need = [e for e in events if e["type"] == "confirm_required"]
    assert len(need) == 1 and need[0]["tool"] == "bash" and need[0]["arguments"]["patterns"] == ["rm -rf build"]
    assert asked and asked[0][1] == "bash"
    reply = next(e for e in log if "/permission/" in e.get("path", ""))
    assert reply["body"] == {"reply": "once" if approve else "reject"}
    assert events[-1]["type"] == "done"


async def test_missing_binary_and_missing_callback_are_clean_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_OPENCODE_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    out = [e async for e in run_opencode("t", "m", "http://x", tmp_path)]
    assert out[0]["type"] == "error" and "opencode" in out[0]["message"]
    out = [e async for e in run_opencode("t", "m", "http://x", tmp_path, binary=str(FAKE))]
    assert out == [{"type": "error", "message": "confirmation callback required when auto_confirm is false"}]


async def test_missing_project_folder_is_reported_not_raised(tmp_path, monkeypatch):
    events = [e async for e in run_opencode("t", "m", "http://x", tmp_path / "does-not-exist", binary=str(FAKE), auto_confirm=True)]
    assert events[-1]["type"] == "error" and "project folder does not exist" in events[-1]["message"]
    assert "does-not-exist" in events[-1]["message"]


async def test_auto_confirm_never_answers_a_question_the_harness_raised(tmp_path, monkeypatch):
    """Auto-approve relaxes what the harness asks about; anything it still asks reaches the user."""
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", "1")
    asked = []

    async def confirm(call_id, tool, arguments):
        asked.append(arguments["patterns"])
        return False

    events, log = await _run(tmp_path, monkeypatch, auto_confirm=True, confirm_callback=confirm)
    assert asked == [["rm -rf build"]]
    reply = next(e for e in log if "/permission/" in e.get("path", ""))
    assert reply["body"] == {"reply": "reject"}


async def test_a_cancelled_run_terminates_the_harness_even_though_its_awaits_are_cancelled(tmp_path, monkeypatch):
    """A disconnecting browser cancels the run; the OpenCode process must not be left behind."""
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", "1")
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(tmp_path / "log.jsonl"))
    reached = asyncio.Event()

    async def never_answers(call_id, tool, arguments):
        reached.set()
        await asyncio.Event().wait()

    async def consume():
        async for _ in run_opencode("t", "m", "http://x", tmp_path, binary=str(FAKE), confirm_callback=never_answers, timeout=60):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(reached.wait(), 20)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert len(monkeypatch.terminated) == 1


async def test_permission_is_seen_even_when_opencode_list_endpoint_fails(tmp_path, monkeypatch):
    """OpenCode 1.19.0 answers 400 on GET /permission for a pending web fetch; the stream still carries it."""
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", "event")
    asked = []

    async def confirm(call_id, tool, arguments):
        asked.append((tool, arguments["patterns"]))
        return False

    events, log = await _run(tmp_path, monkeypatch, confirm_callback=confirm)
    assert asked == [("bash", ["rm -rf build"])]                       # asked exactly once, not hung and not duplicated
    assert [e["type"] for e in events].count("confirm_required") == 1
    reply = next(e for e in log if "/permission/" in e.get("path", ""))
    assert reply["body"] == {"reply": "reject"} and events[-1]["type"] == "done"


def test_inherited_tools_ask_in_ask_mode_and_run_in_auto_mode(tmp_path):
    ask = build_config("http://x", "m", auto_confirm=False, mcp_servers=["github", "memory"])["permission"]
    keys = list(ask)
    assert ask["github_*"] == "ask" and ask["memory_*"] == "ask"
    # the blanket ask comes first; the read-only allowances after it (the last matching rule wins)
    assert keys.index("github_*") < keys.index("github_list*") < len(keys)
    assert ask["github_get*"] == "allow" and ask["memory_search*"] == "allow"
    assert "github_create*" not in ask and "github_delete*" not in ask          # mutating tools are not allowed through
    auto = build_config("http://x", "m", auto_confirm=True, mcp_servers=["github", "memory"])["permission"]
    assert set(auto) == {"edit", "bash"}                                        # Auto-approve leaves inherited tools alone


def test_mcp_server_names_reads_only_the_keys(tmp_path, monkeypatch):
    from arc_llama.agent.harness.opencode import mcp_server_names

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "opencode.json").write_text('{"mcp": {"docs": {"type": "remote", "url": "https://x", "headers": {"Authorization": "secret"}}}}')
    names = mcp_server_names(project)
    assert names == ["docs"] and "secret" not in str(names)
    assert mcp_server_names(tmp_path / "nowhere") == []


async def test_a_slash_command_runs_through_the_harness_instead_of_a_prompt(tmp_path, monkeypatch):
    events, log = await _run(tmp_path, monkeypatch, auto_confirm=True, command="review", arguments="main")
    assert [e["type"] for e in events][-1] == "done" and "error" not in [e["type"] for e in events]
    sent = next(e for e in log if e.get("path", "").endswith("/command"))
    assert sent["body"] == {"command": "review", "arguments": "main", "model": "arc-llama/m"} and sent["auth"]
    assert not any("prompt_async" in e.get("path", "") for e in log)


async def test_a_command_the_harness_rejects_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OPENCODE_COMMAND_FAILS", "1")
    events, _ = await _run(tmp_path, monkeypatch, auto_confirm=True, command="nope")
    assert events[-1]["type"] == "error" and "/nope" in events[-1]["message"]


async def test_list_commands_reads_the_harness_and_caches(tmp_path, monkeypatch):
    oc._COMMAND_CACHE.clear()
    found = await oc.list_commands(tmp_path, binary=str(FAKE))
    assert [c["name"] for c in found] == ["review", "quota", "pdf"]
    assert found[0] == {"name": "review", "description": "review changes", "source": "command", "takes_arguments": True}
    assert found[1]["takes_arguments"] is False and found[2]["source"] == "skill"
    assert await oc.list_commands(tmp_path, binary=str(FAKE)) is found        # cached, no second server


def test_translator_reports_latest_prompt_usage_including_cached_tokens():
    tr = OpenCodeTranslator("s")
    def message(mid, tokens, sid="s", role="assistant"):
        return {"type": "message.updated", "properties": {
            "sessionID": sid, "info": {"id": mid, "role": role, "tokens": tokens}}}
    first = message("m1", {"input": 100, "output": 20, "reasoning": 8,
                           "cache": {"read": 500, "write": 200}})
    assert tr.feed(first) == [{"type": "usage", "prompt_tokens": 800, "completion_tokens": 20}]
    assert tr.feed(first) == []
    # Per-call context, not an accumulated bill across the run.
    assert tr.feed(message("m2", {"input": 300, "output": 10})) == [
        {"type": "usage", "prompt_tokens": 300, "completion_tokens": 10}]
    assert tr.feed(message("other", {"input": 900, "output": 1}, sid="different")) == []
    for bad in ({"input": -1, "output": 2}, {"input": True, "output": 2}, {"input": "42", "output": 2}, {}):
        assert tr.feed(message("bad", bad)) == []


def test_translator_reports_thinking_and_answering_without_reasoning_text():
    tr = OpenCodeTranslator("s")
    tr.feed({"type":"message.updated", "properties":{"sessionID":"s", "info":{"id":"m","role":"assistant"}}})
    reasoning = {"type":"message.part.updated", "properties":{"sessionID":"s", "part":{"id":"r", "messageID":"m", "type":"reasoning", "text":"private reasoning", "time":{"start":1}}}}
    assert tr.feed(reasoning) == [{"type":"activity", "phase":"thinking"}]
    assert tr.feed(reasoning) == []
    delta = {"type":"message.part.delta", "properties":{"sessionID":"s", "partID":"r", "delta":"private text"}}
    assert tr.feed(delta) == []
    tr._phase("generating")
    assert tr.feed(delta) == [{"type":"activity", "phase":"thinking"}]
    text = {"type":"message.part.updated", "properties":{"sessionID":"s", "part":{"id":"a", "messageID":"m", "type":"text", "text":"Public answer", "time":{"end":2}}}}
    assert tr.feed(text) == [{"type":"activity", "phase":"answering"}, {"type":"assistant", "content":"Public answer"}]


async def test_generated_session_reports_id_and_uses_task_title(tmp_path, monkeypatch):
    events, log = await _run(tmp_path, monkeypatch, auto_confirm=True, session_title="Synthetic user title")
    created = next(e for e in log if e.get("path", "").split("?")[0] == "/session")
    assert created["body"]["title"] == "Synthetic user title"
    linked = next(e for e in events if e["type"] == "harness_session")
    assert linked["session_id"].startswith("oc-ses_")
