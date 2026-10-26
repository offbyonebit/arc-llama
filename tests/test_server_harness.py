"""POST /v1/agent with an external harness: gating, auth and event streaming."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from helpers import ConfigRouter as FakeRouter

import arc_llama.server as server_mod
from arc_llama.config import Config, ServerConfig
from arc_llama.server import create_app


class FakeUpstreamManager:
    def __init__(self, upstreams=None):
        pass

    async def models(self):
        return []

    def find_model(self, model_id):
        return None

    def upstreams_status(self):
        return []


def _app(tmp_path, monkeypatch, token=""):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(server=ServerConfig(admin_token=token))
    cfg.paths.state_dir = str(tmp_path / "state")
    return create_app(cfg)


def _events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


BODY = {"model": "qwen", "task": "list files", "harness": "opencode"}


@pytest.fixture(autouse=True)
def _no_flag(monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_EXPERIMENTAL_AGENT", raising=False)


def test_unknown_harness_is_rejected(tmp_path, monkeypatch):
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json={**BODY, "harness": "nope"})
        assert r.status_code == 400 and "harness" in r.json()["detail"]


def test_external_harness_requires_experimental_flag(tmp_path, monkeypatch):
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json=BODY)
        assert r.status_code == 403 and "ARC_LLAMA_EXPERIMENTAL_AGENT" in r.json()["detail"]


def test_external_harness_requires_admin_when_token_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch, token="secret")) as client:
        assert client.post("/v1/agent", json=BODY).status_code == 401
        assert client.post("/v1/agent", json=BODY, headers={"Authorization": "Bearer wrong"}).status_code == 403


def test_opencode_run_streams_events_and_passes_options(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        yield {"type": "status", "message": "Working…"}
        yield {"type": "tool_call", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}
        yield {"type": "tool_result", "id": "c1", "name": "bash", "content": "a.txt\n", "error": False}
        yield {"type": "assistant", "content": "There is one file."}
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    app = _app(tmp_path, monkeypatch, token="secret")
    with TestClient(app) as client:
        r = client.post("/v1/agent", json={**BODY, "local_only": True, "root": str(tmp_path)},
                        headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
        assert [e["type"] for e in _events(r)] == ["status", "tool_call", "tool_result", "assistant", "done"]
        assert seen["model"] == "qwen" and seen["local_only"] is True and seen["auto_confirm"] is False
        assert seen["root"] == tmp_path.resolve() and callable(seen["confirm_callback"])
        # the run is saved as a chat transcript like any other agent run
        chats = client.get("/v1/chats").json()["data"]
        assert len(chats) == 1


def test_builtin_harness_is_unaffected_by_flag(tmp_path, monkeypatch):
    # No flag set: default behaviour must be untouched (validation only; run_agent is covered elsewhere).
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json={"model": "qwen", "task": "x", "harness": "builtin", "max_turns": 0})
        assert r.status_code == 400 and "max_turns" in r.json()["detail"]


def test_agent_info_reports_flag_and_harnesses(tmp_path, monkeypatch):
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        info = client.get("/v1/agent/info").json()
        assert info["experimental"] is False and info["root"]
        assert info["auto_approve"] is True            # the UI only offers auto-approve when this is present
        ids = {h["id"]: h for h in info["harnesses"]}
        assert ids["builtin"]["available"] is True and "opencode" in ids
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        assert client.get("/v1/agent/info").json()["experimental"] is True


def test_agent_info_requires_admin_token(tmp_path, monkeypatch):
    with TestClient(_app(tmp_path, monkeypatch, token="secret")) as client:
        assert client.get("/v1/agent/info").status_code == 401
        assert client.get("/v1/agent/info", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_tracked_run_reports_changes_then_can_be_reviewed_and_undone(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "keep.txt").write_text("original\n")

    async def fake_run(**kwargs):
        (project / "hello.txt").write_text("arc-llama work mode\n")
        (project / "keep.txt").write_text("edited by agent\n")
        yield {"type": "assistant", "content": "done"}
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    body = {**BODY, "root": str(project), "track_changes": True}
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        events = _events(client.post("/v1/agent", json=body))
        changes = next(e for e in events if e["type"] == "changes")
        assert {f["path"]: f["status"] for f in changes["files"]} == {"hello.txt": "added", "keep.txt": "modified"}
        assert changes["undoable"] is True and changes["run_id"]
        run_id = changes["run_id"]

        review = client.get(f"/v1/agent/{run_id}/changes/keep.txt").json()
        assert "-original" in review["diff"] and "+edited by agent" in review["diff"]
        assert client.get(f"/v1/agent/{run_id}/changes/nope.txt").status_code == 404

        undone = client.post(f"/v1/agent/{run_id}/undo").json()
        assert {r["path"] for r in undone["restored"]} == {"hello.txt", "keep.txt"} and undone["skipped"] == []
        assert not (project / "hello.txt").exists()
        assert (project / "keep.txt").read_text() == "original\n"
        assert client.post(f"/v1/agent/{run_id}/undo").status_code == 404      # already undone


def test_untracked_run_emits_no_changes_event(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")

    async def fake_run(**kwargs):
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        events = _events(client.post("/v1/agent", json={**BODY, "root": str(tmp_path)}))
        assert [e["type"] for e in events] == ["done"]


def test_missing_project_folder_is_a_clear_400_before_anything_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json={**BODY, "root": str(tmp_path / "nope"), "track_changes": True})
        assert r.status_code == 400 and "Project folder not found" in r.json()["detail"]
        assert client.get("/v1/agent/info").json()["home"]


def test_missing_folder_suggests_the_closest_existing_one_and_relative_means_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json={**BODY, "root": str(home / "projects" / "typo" / "deeper"), "track_changes": True})
        assert r.status_code == 400
        assert r.json()["closest"] == str(home / "projects") and "typo" in r.json()["detail"]
        # a relative path is read as "inside the home folder", not relative to the server process
        r = client.post("/v1/agent", json={**BODY, "root": "projects/nope", "track_changes": True})
        assert r.status_code == 400 and r.json()["detail"].endswith(str(home / "projects" / "nope"))


def test_earlier_turns_reach_the_agent_but_the_saved_transcript_stays_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        yield {"type": "assistant", "content": "done"}
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    history = [{"role": "user", "content": "write calc.py"}, {"role": "assistant", "content": "Created calc.py"}]
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        r = client.post("/v1/agent", json={**BODY, "task": "now add tests", "root": str(tmp_path), "history": history})
        assert r.status_code == 200
        assert "write calc.py" in seen["task"] and "Created calc.py" in seen["task"]
        assert f"The project folder is {tmp_path.resolve()}" in seen["task"]     # the agent is told where it works
        assert seen["task"].rstrip().endswith("now add tests")
        chat_id = client.get("/v1/chats").json()["data"][0]["id"]
        saved = client.get(f"/v1/chats/{chat_id}").json()
        assert saved["messages"][0]["content"] == "now add tests"          # the user's own words, not the composed prompt


def test_bad_history_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        for bad in ("nope", [{"role": "system", "content": "x"}], [{"role": "user", "content": 3}], [1]):
            r = client.post("/v1/agent", json={**BODY, "root": str(tmp_path), "history": bad})
            assert r.status_code == 400 and "history" in r.json()["detail"]


def test_compose_prompt_keeps_the_newest_turns_when_it_has_to_trim():
    from arc_llama.agent.harness import compose_prompt

    assert compose_prompt("hi", []) == "hi"
    long = [{"role": "user", "content": f"turn {i} " + "x" * 2900} for i in range(10)]
    out = compose_prompt("final question", long)
    assert "turn 9 " in out and "turn 0 " not in out                      # oldest dropped first
    assert out.rstrip().endswith("final question")


def test_slash_command_is_passed_to_the_harness_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        ok = client.post("/v1/agent", json={**BODY, "root": str(tmp_path), "command": "review", "arguments": "main"})
        assert ok.status_code == 200 and seen["command"] == "review" and seen["arguments"] == "main"
        assert client.post("/v1/agent", json={**BODY, "root": str(tmp_path), "command": "../etc"}).status_code == 400
        assert client.post("/v1/agent", json={**BODY, "root": str(tmp_path), "command": "x", "arguments": 3}).status_code == 400
        builtin = client.post("/v1/agent", json={"model": "qwen", "task": "x", "harness": "builtin", "command": "review"})
        assert builtin.status_code == 400


def test_agent_commands_lists_the_harness_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    calls = []

    async def fake_list(folder, model, **_kw):
        calls.append(str(folder))
        return [{"name": "review", "description": "d", "source": "command", "takes_arguments": True}]

    monkeypatch.setattr(server_mod, "list_commands", fake_list)
    with TestClient(_app(tmp_path, monkeypatch, token="secret")) as client:
        auth = {"Authorization": "Bearer secret"}
        assert client.get("/v1/agent/commands").status_code == 401
        r = client.get("/v1/agent/commands", params={"root": str(tmp_path)}, headers=auth)
        assert r.status_code == 200 and r.json()["commands"][0]["name"] == "review" and calls == [str(tmp_path.resolve())]
        assert client.get("/v1/agent/commands", params={"harness": "builtin"}, headers=auth).json() == {"commands": []}
        assert client.get("/v1/agent/commands", params={"root": str(tmp_path / "missing")}, headers=auth).status_code == 400


@pytest.mark.parametrize("harness", ["opencode", "hermes"])
def test_external_adapters_save_streamed_text_and_track_changes(tmp_path, monkeypatch, harness):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        yield {"type": "activity", "phase": "thinking"}
        yield {"type": "assistant_delta", "content": "Before "}
        yield {"type": "assistant_delta", "content": "the edit."}
        yield {"type": "tool_call", "id": "edit-1", "name": "write_file", "arguments": {"path": "answer.txt"}}
        (tmp_path / "answer.txt").write_text("Hello from Hermes\n")
        yield {"type": "tool_result", "id": "edit-1", "name": "write_file", "content": "written", "error": False}
        yield {"type": "assistant_delta", "content": "All "}
        yield {"type": "assistant_delta", "content": "done."}
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_" + harness, fake_run)
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        events = _events(client.post("/v1/agent", json={**BODY, "harness": harness, "root": str(tmp_path), "track_changes": True, "history": [{"role": "user", "content": "Earlier task"}]}))
        assert "Earlier task" in seen["task"]
        assert seen["local_only"] is True and seen["auto_confirm"] is False
        assert next(e for e in events if e["type"] == "changes")["files"][0]["path"] == "answer.txt"
        chats = client.get("/v1/chats").json()["data"]
        chat = client.get("/v1/chats/" + chats[0]["id"]).json()
        assert [m["content"] for m in chat["messages"] if m["role"] == "assistant"] == ["Before the edit.", "All done."]


@pytest.mark.parametrize("options", [{"auto_confirm": True}, {"local_only": False}, {"tool_profile_id": "p"}, {"command": "model"}])
def test_hermes_rejects_unsupported_features(tmp_path, monkeypatch, options):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        response = client.post("/v1/agent", json={**BODY, "harness": "hermes", "root": str(tmp_path), **options})
        assert response.status_code == 400


def test_hermes_requires_flag_and_auth_and_reports_capabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "find_hermes_binary", lambda *_: "/bin/hermes")
    with TestClient(_app(tmp_path, monkeypatch)) as client:
        assert client.post("/v1/agent", json={**BODY, "harness": "hermes"}).status_code == 403
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    with TestClient(_app(tmp_path, monkeypatch, token="secret")) as client:
        assert client.post("/v1/agent", json={**BODY, "harness": "hermes"}).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        info = client.get("/v1/agent/info", headers=headers).json()
        hermes = next(h for h in info["harnesses"] if h["id"] == "hermes")
        assert hermes["available"] is True and hermes["auto_approve"] is False
        assert "flagged" in hermes["approval_note"]
        assert client.get("/v1/agent/commands?harness=hermes", headers=headers).json() == {"commands": []}


def _install_app(tmp_path, monkeypatch, token="secret"):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(server=ServerConfig(admin_token=token))
    cfg.paths.state_dir = str(tmp_path / "state")
    return create_app(cfg, config_path=tmp_path / "config.toml"), cfg


def test_agent_info_reports_harness_source_and_install_offer(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    monkeypatch.delenv("ARC_LLAMA_OPENCODE_BIN", raising=False)
    monkeypatch.setattr(server_mod, "find_binary", lambda *_: None)
    monkeypatch.setattr(server_mod, "find_hermes_binary", lambda *_: "/usr/bin/hermes")
    app, _cfg = _install_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        info = client.get("/v1/agent/info", headers={"Authorization": "Bearer secret"}).json()
    by_id = {h["id"]: h for h in info["harnesses"]}
    assert by_id["opencode"]["available"] is False and by_id["opencode"]["source"] == "missing"
    offer = by_id["opencode"]["install"]
    assert offer["license"] == "MIT" and offer["source"].startswith("https://github.com/")
    assert offer["location"].startswith(str(tmp_path / "state"))
    assert by_id["hermes"]["source"] == "path" and by_id["hermes"]["path"] == "/usr/bin/hermes"


def test_harness_install_endpoints_gate_auth_and_run_jobs(tmp_path, monkeypatch):
    import arc_llama.harness_install as hi

    app, cfg = _install_app(tmp_path, monkeypatch)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.post("/admin/harnesses/opencode/install").status_code == 401
        assert client.post("/admin/harnesses/opencode/install", headers=headers).status_code == 403
        monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
        seen = {}

        def fake_install(cfg_, name, **kw):
            seen.update(name=name, version=kw["version"], config_path=kw["config_path"])
            kw["progress"]("Downloading OpenCode", 1, 2)
            return hi.InstallResult(name, "v1", tmp_path / "opencode", tmp_path, True)

        app.state.harness_installs.installer = fake_install
        assert client.post("/admin/harnesses/nope/install", headers=headers).status_code == 400
        assert client.post("/admin/harnesses/opencode/install", json={"version": "v0"}, headers=headers).status_code == 400
        job = client.post("/admin/harnesses/opencode/install", json={"version": "latest"}, headers=headers).json()
        for _ in range(100):
            jobs = client.get("/admin/harnesses/jobs", headers=headers).json()["jobs"]
            if jobs[0]["status"] == "done":
                break
            import time

            time.sleep(0.02)
        assert jobs[0]["id"] == job["id"] and jobs[0]["status"] == "done" and jobs[0]["installed_version"] == "v1"
        assert seen == {"name": "opencode", "version": "latest", "config_path": tmp_path / "config.toml"}
        assert client.post("/admin/harnesses/jobs/nope/cancel", headers=headers).status_code == 404
        assert client.delete("/admin/harnesses/opencode/v1", headers=headers).status_code == 404
        app.state.active_harness_runs["opencode"] = 1
        assert client.delete("/admin/harnesses/opencode/v1", headers=headers).status_code == 409


def test_work_run_uses_the_configured_harness_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    seen = {}

    async def fake_run(**kwargs):
        seen["binary"] = kwargs.get("binary")
        yield {"type": "done"}

    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    app, cfg = _install_app(tmp_path, monkeypatch, token="")
    cfg.agent.opencode_bin = "/managed/opencode"
    with TestClient(app) as client:
        client.post("/v1/agent", json={**BODY, "root": str(tmp_path)})
        assert app.state.active_harness_runs["opencode"] == 0
    assert seen["binary"] == "/managed/opencode"
