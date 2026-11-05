"""OpenCode 2 support: the /api surface, its event stream and its question forms, against a fake
`opencode serve` that speaks OpenCode 2.2.5's protocol (no real OpenCode or model needed)."""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import pytest
from helpers import python_executable

from arc_llama.agent.harness import opencode as oc
from arc_llama.agent.harness.opencode import OpenCodeV2Translator, build_config, run_opencode

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake(tmp_path):
    return python_executable(
        tmp_path / "fake-opencode2",
        f"import runpy; runpy.run_path({str(FIXTURES / 'fake_opencode2.py')!r}, run_name='__main__')\n",
    )


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
        terminated.append(self.url)
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(self.url + "/__exit", timeout=2)

    monkeypatch.setattr(oc._Server, "stop", stop)
    monkeypatch.setattr(oc._Server, "terminate_now", terminate_now)
    monkeypatch.terminated = terminated


async def _run(tmp_path, monkeypatch, fake, **kwargs):
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(tmp_path / "log.jsonl"))
    kwargs.setdefault("timeout", 30)
    events = [e async for e in run_opencode("run it", "m", "http://127.0.0.1:11435", tmp_path, binary=str(fake), **kwargs)]
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return events, log


async def test_a_run_speaks_the_api_and_translates_the_stream(tmp_path, monkeypatch, fake):
    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True)
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and "error" not in kinds
    created = next(e for e in log if e.get("path") == "/api/session")
    assert created["body"] == {"model": {"id": "m", "providerID": "arc-llama"}, "location": {"directory": str(tmp_path.resolve())}}
    prompt = next(e for e in log if e.get("path", "").endswith("/prompt"))
    assert prompt["body"] == {"text": "run it"} and prompt["auth"] and prompt["location"] == str(tmp_path.resolve())
    assert [(e["name"], e["arguments"]) for e in events if e["type"] == "tool_call"] == [("shell", {"command": "rm -rf build"})]
    assert [e["content"] for e in events if e["type"] == "tool_result"] == ["(no output)"]
    assert "".join(e["content"] for e in events if e["type"] == "assistant_delta") == "Done. All good."
    assert "not ours" not in json.dumps(events)
    assert [(e["prompt_tokens"], e["completion_tokens"]) for e in events if e["type"] == "usage"] == [(120, 7), (130, 4)]
    assert any(e["type"] == "status" and "retrying (attempt 2): ECONNRESET" in e["message"] for e in events)
    assert [e["phase"] for e in events if e["type"] == "activity"][:2] == ["thinking", "answering"]
    assert next(e for e in events if e["type"] == "harness_session")["session_id"] == "oc-ses_fake2"
    assert any(e.get("path") == "/api/session/ses_fake2/interrupt" for e in log)     # always cleans the session up


@pytest.mark.parametrize("mode", ["1", "event"])
@pytest.mark.parametrize("approve", [True, False])
async def test_a_permission_reaches_the_user_in_the_1x_vocabulary(tmp_path, monkeypatch, fake, mode, approve):
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", mode)
    asked = []

    async def confirm(call_id, tool, arguments):
        asked.append((call_id, tool, arguments))
        return approve

    events, log = await _run(tmp_path, monkeypatch, fake, confirm_callback=confirm)
    # OpenCode 2 calls it "shell"; the Work view's approval sheet speaks of "bash".
    assert [(a[0], a[1], a[2]["patterns"]) for a in asked] == [("per_1", "bash", ["rm -rf build"])]
    replies = [e for e in log if "/permission/per_" in e.get("path", "")]
    assert [(r["path"], r["body"]) for r in replies] == [
        ("/api/session/ses_fake2/permission/per_1/reply", {"decision": "once" if approve else "reject"})]
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("auto", [False, True])
async def test_saved_always_allows_are_reported_before_the_run(tmp_path, monkeypatch, fake, auto):
    """A saved "Always allow" in the shared opencode.db beats Work's "ask", so the user is told."""
    monkeypatch.setenv("FAKE_OPENCODE_SAVED", json.dumps([
        {"id": "prm_1", "projectID": "p", "action": "shell", "resource": "rm *"},
        {"id": "prm_2", "projectID": "p", "action": "edit", "resource": "*"}]))

    async def confirm(call_id, tool, arguments):
        return True

    events, _ = await _run(tmp_path, monkeypatch, fake, auto_confirm=auto, confirm_callback=None if auto else confirm)
    warnings = [e["message"] for e in events if e["type"] == "warning"]
    assert len(warnings) == 1 and "bash rm *" in warnings[0] and ("edit *" in warnings[0]) is not auto
    assert events.index(next(e for e in events if e["type"] == "warning")) < events.index(next(e for e in events if e["type"] == "tool_call"))
    assert events[-1]["type"] == "done"


async def test_a_provider_blocked_by_the_users_config_is_explained(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_PROVIDER_BLOCKED", "1")
    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True)
    assert events[-1]["type"] == "error" and "enabled_providers" in events[-1]["message"]
    assert not any(e.get("path") == "/api/session" for e in log)       # nothing is started for a run that can't work


async def test_no_saved_permissions_means_no_warning(tmp_path, monkeypatch, fake):
    events, _ = await _run(tmp_path, monkeypatch, fake, auto_confirm=True)
    assert not [e for e in events if e["type"] == "warning"]


@pytest.mark.parametrize("mode", ["1", "event"])
async def test_a_question_form_goes_to_the_user_and_the_answer_back(tmp_path, monkeypatch, fake, mode):
    monkeypatch.setenv("FAKE_OPENCODE_QUESTION", mode)
    monkeypatch.setenv("FAKE_OPENCODE_QUESTION_MULTI", "1")
    asked = []

    async def ask(question_id, questions):
        asked.append((question_id, questions))
        return [["Postgres"], ["Cache", "Search"]]

    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True, question_callback=ask)
    need = [e for e in events if e["type"] == "question_required"]
    assert len(need) == 1 and need[0]["id"] == "frm_1" and need[0]["call_id"] == "call_q"
    assert need[0]["questions"][0] == {"question": "Which database?", "header": "Database", "multiple": False, "custom": True,
                                       "options": [{"label": "SQLite", "description": "One file"},
                                                   {"label": "Postgres", "description": "A server"}]}
    assert need[0]["questions"][1]["multiple"] is True
    reply = [e for e in log if "/form/" in e.get("path", "")]
    assert [(r["path"], r["body"]) for r in reply] == [
        ("/api/session/ses_fake2/form/frm_1/reply", {"answer": {"q0": "Postgres", "q1": ["Cache", "Search"]}})]
    assert log[0]["env"]["OPENCODE_ENABLE_QUESTION_TOOL"] == "1" and events[-1]["type"] == "done"


async def test_a_dismissed_question_cancels_the_form(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_QUESTION", "event")

    async def dismiss(question_id, questions):
        return None

    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True, question_callback=dismiss)
    assert [(e["method"], e["path"]) for e in log if "/form/" in e.get("path", "")] == [("DELETE", "/api/session/ses_fake2/form/frm_1")]
    assert events[-1]["type"] == "done"


async def test_a_failed_run_reports_opencodes_reason(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_FAIL", "1")
    events, _ = await _run(tmp_path, monkeypatch, fake, auto_confirm=True)
    assert events[-1] == {"type": "error", "message": "context length exceeded"}


async def test_a_slash_command_and_a_skill_run_through_the_harness(tmp_path, monkeypatch, fake):
    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True, command="review", arguments="main")
    assert events[-1]["type"] == "done"
    assert next(e for e in log if e.get("path", "").endswith("/command"))["body"] == {"name": "review", "text": "main"}
    (tmp_path / "log.jsonl").unlink()
    events, log = await _run(tmp_path, monkeypatch, fake, auto_confirm=True, command="pdf", arguments="report.pdf")
    # OpenCode 2 lists skills apart from commands and runs one as a prompt that names it.
    assert events[-1]["type"] == "done"
    assert next(e for e in log if e.get("path", "").endswith("/prompt"))["body"] == {"text": "report.pdf", "skills": [{"id": "pdf"}]}


async def test_a_command_the_harness_rejects_is_reported(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_COMMAND_FAILS", "1")
    events, _ = await _run(tmp_path, monkeypatch, fake, auto_confirm=True, command="nope")
    assert events[-1]["type"] == "error" and "/nope" in events[-1]["message"] and "HTTP 500" in events[-1]["message"]


async def test_list_commands_waits_for_the_list_and_adds_skills(tmp_path, monkeypatch, fake):
    oc._COMMAND_CACHE.clear()
    found = await oc.list_commands(tmp_path, binary=str(fake))
    assert found == [
        {"name": "review", "description": "review changes", "source": "command", "takes_arguments": True},
        {"name": "pdf", "description": "PDF skill", "source": "skill", "takes_arguments": False},
    ]


async def test_probe_tools_scans_through_the_api_and_deletes_the_session(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_SEND_TOOLS", "shell,read,execute")
    monkeypatch.setenv("FAKE_OPENCODE_LOG", str(tmp_path / "log.jsonl"))
    names = await oc.probe_tools(tmp_path, build_config("http://127.0.0.1:1", "m"), binary=str(fake), timeout=20)
    assert names == ["execute", "read", "shell"]
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(e.get("method") == "DELETE" and e["path"] == "/api/session/ses_fake2" for e in log)
    assert {t["id"]: t["group"] for t in oc.group_tools(names, [])} == {"execute": "builtin", "read": "builtin", "shell": "builtin"}


async def test_a_cancelled_run_terminates_the_harness(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("FAKE_OPENCODE_PERMISSION", "event")
    reached = asyncio.Event()

    async def never_answers(call_id, tool, arguments):
        reached.set()
        await asyncio.Event().wait()

    async def consume():
        async for _ in run_opencode("t", "m", "http://x", tmp_path, binary=str(fake), confirm_callback=never_answers, timeout=60):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(reached.wait(), 20)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert len(monkeypatch.terminated) == 1


def test_a_profile_also_denies_left_out_servers_by_policy():
    """OpenCode 2 drops `mcp: {name: {enabled: false}}`; only a policy keeps the server from starting."""
    config = build_config("http://x", "m", mcp_servers=["docs", "github"], profile={"mcp": ["docs"], "skills": [], "web": False})
    assert config["policies"] == [{"action": "mcp.use", "resource": "github", "effect": "deny"}]
    assert config["mcp"]["github"] == {"enabled": False} and config["permission"]["github_*"] == "deny"
    assert "policies" not in build_config("http://x", "m", mcp_servers=["docs"])


def test_mcp_server_names_reads_opencode_2_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "opencode.json").write_text(json.dumps({"mcp": {"timeout": {"startup": 5}, "servers": {
        "docs": {"type": "remote", "url": "https://x", "headers": {"Authorization": "secret"}}}}}))
    assert oc.mcp_server_names(project) == ["docs"]


def test_translator_settles_final_text_and_reports_tool_failures():
    t = OpenCodeV2Translator("s")
    out = []
    for delta in ("Hello ", "there"):
        out += t.feed({"type": "session.text.delta", "data": {"sessionID": "s", "assistantMessageID": "m", "ordinal": 0, "delta": delta}})
    out += t.feed({"type": "session.text.ended", "data": {"sessionID": "s", "assistantMessageID": "m", "ordinal": 0, "text": "Hello there!"}})
    assert [e["content"] for e in out if e["type"] == "assistant_delta"] == ["Hello ", "there", "!"]
    out = t.feed({"type": "session.tool.failed", "data": {"sessionID": "s", "id": "c9",
                                                         "error": {"type": "aborted", "message": "The user dismissed this question"}}})
    assert out[0]["type"] == "tool_call" and out[1] == {"type": "tool_result", "id": "c9", "name": "tool",
                                                         "content": "The user dismissed this question", "error": True}
    assert t.feed({"type": "session.execution.interrupted", "data": {"sessionID": "s"}}) == [{"type": "done"}]
    assert t.feed({"type": "session.execution.succeeded", "data": {"sessionID": "s"}}) == []
