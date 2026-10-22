"""Saved Work sessions: storage and the admin-only endpoints that reopen them."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from helpers import ConfigRouter as FakeRouter

import arc_llama.server as server_mod
from arc_llama.agent.sessions import MAX_SESSIONS, SessionStore
from arc_llama.config import Config, ServerConfig


def test_store_round_trips_lists_newest_first_and_deletes(tmp_path):
    store = SessionStore(tmp_path)
    store.save("session-aaaa", {"title": "First", "folder": "/p", "runs": [{"task": "a"}]})
    store.save("session-bbbb", {"title": "Second", "runs": []})
    assert [s["title"] for s in store.list()] == ["Second", "First"]
    assert store.get("session-aaaa")["runs"] == [{"task": "a"}]
    assert store.list()[1]["tasks"] == 1
    store.save("session-aaaa", {"title": "First again", "runs": [{}, {}]})          # saving again updates in place
    assert [s["title"] for s in store.list()] == ["First again", "Second"]
    assert store.delete("session-aaaa") and store.get("session-aaaa") is None
    assert store.delete("session-aaaa") is False


def test_store_refuses_bad_ids_and_oversize(tmp_path):
    store = SessionStore(tmp_path)
    for bad in ("../x", "short", "a/b/c/d/e/f/g/h", "x" * 100):
        with pytest.raises(ValueError):
            store.save(bad, {})
    with pytest.raises(ValueError, match="too large"):
        store.save("session-big1", {"runs": ["x" * (5 * 1024 * 1024)]})
    assert store.list() == []


def test_store_keeps_only_the_newest_sessions(tmp_path):
    store = SessionStore(tmp_path)
    for i in range(MAX_SESSIONS + 3):
        store.save(f"session-{i:04d}", {"title": str(i)})
    assert len(store.list()) == MAX_SESSIONS


@pytest.fixture
def client(tmp_path, monkeypatch):
    class Upstreams:
        def __init__(self, *_a, **_k):
            pass

        async def models(self):
            return []

        def find_model(self, _id):
            return None

        def upstreams_status(self):
            return []

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", Upstreams)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    cfg.paths.state_dir = str(tmp_path / "state")
    with TestClient(server_mod.create_app(cfg)) as c:
        yield c


AUTH = {"Authorization": "Bearer secret"}


def test_endpoints_need_the_admin_token(client):
    assert client.get("/v1/agent/sessions").status_code == 401
    assert client.put("/v1/agent/sessions/session-aaaa", json={}).status_code == 401
    assert client.get("/v1/agent/sessions/session-aaaa").status_code == 401
    assert client.delete("/v1/agent/sessions/session-aaaa").status_code == 401


def test_save_list_reopen_and_delete_a_session(client):
    body = {"title": "Fix the build", "folder": "/p/app", "model": "m", "harness": "opencode",
            "compacted": {"summary": "s", "upTo": 1}, "runs": [{"task": "Fix the build", "answer": ["Done"]}]}
    saved = client.put("/v1/agent/sessions/session-aaaa", json=body, headers=AUTH)
    assert saved.status_code == 200 and saved.json()["tasks"] == 1
    listed = client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"]
    assert [s["id"] for s in listed] == ["session-aaaa"] and listed[0]["title"] == "Fix the build"
    again = client.get("/v1/agent/sessions/session-aaaa", headers=AUTH).json()
    assert again["runs"][0]["answer"] == ["Done"] and again["compacted"] == {"summary": "s", "upTo": 1}
    assert client.delete("/v1/agent/sessions/session-aaaa", headers=AUTH).json() == {"deleted": True}
    assert client.get("/v1/agent/sessions/session-aaaa", headers=AUTH).status_code == 404


def test_endpoints_reject_bad_input(client):
    assert client.put("/v1/agent/sessions/bad", json={}, headers=AUTH).status_code == 400
    assert client.put("/v1/agent/sessions/session-aaaa", content=b"not json", headers=AUTH).status_code == 400
    assert client.put("/v1/agent/sessions/session-aaaa", json=[1], headers=AUTH).status_code == 400
    assert client.get("/v1/agent/sessions/bad", headers=AUTH).status_code == 404


def test_native_history_is_admin_only_and_requires_experimental_flag(client, monkeypatch):
    calls = []
    monkeypatch.setattr(server_mod, "list_native_sessions", lambda: (calls.append("list") or [], []))
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    assert client.get("/v1/agent/sessions").status_code == 401
    assert client.get("/v1/agent/sessions/oc-ses_fixture").status_code == 401
    assert calls == []
    monkeypatch.delenv("ARC_LLAMA_EXPERIMENTAL_AGENT")
    assert client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"] == []
    assert calls == []


def test_native_history_merges_opens_saves_copy_and_never_deletes_original(client, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    native = {"id": "oc-ses_fixture", "title": "Synthetic native chat", "folder": "/fixture", "updated": 1,
              "tasks": 1, "harness": "opencode", "native": True, "origin": "opencode",
              "runs": [{"task": "Synthetic question", "answer": ["Synthetic answer"]}]}
    monkeypatch.setattr(server_mod, "list_native_sessions", lambda: ([native], ["Synthetic history warning"]))
    monkeypatch.setattr(server_mod, "get_native_session", lambda sid: dict(native) if sid == native["id"] else None)
    client.put("/v1/agent/sessions/session-work", json={"title": "Work chat"}, headers=AUTH)
    listed = client.get("/v1/agent/sessions", headers=AUTH).json()
    assert [s["id"] for s in listed["sessions"]] == ["session-work", "oc-ses_fixture"]
    assert listed["warnings"] == ["Synthetic history warning"]
    assert client.get("/v1/agent/sessions/oc-ses_fixture", headers=AUTH).json()["runs"] == native["runs"]
    # Viewing a session is read-only; deleting an unsaved native entry does nothing.
    assert client.delete("/v1/agent/sessions/oc-ses_fixture", headers=AUTH).json() == {"deleted": False}
    copy = {**native, "runs": native["runs"] + [{"task": "Follow-up", "answer": []}]}
    assert client.put("/v1/agent/sessions/oc-ses_fixture", json=copy, headers=AUTH).status_code == 200
    listed = client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"]
    assert len(listed) == 2
    saved = next(s for s in listed if s["id"] == "oc-ses_fixture")
    assert saved["origin"] == "opencode" and not saved.get("native")
    assert len(client.get("/v1/agent/sessions/oc-ses_fixture", headers=AUTH).json()["runs"]) == 2
    assert client.delete("/v1/agent/sessions/oc-ses_fixture", headers=AUTH).json() == {"deleted": True}
    assert client.get("/v1/agent/sessions/oc-ses_fixture", headers=AUTH).json()["runs"] == native["runs"]
    assert len(native["runs"]) == 1


@pytest.mark.parametrize("message,status", [("Session is too large", 413), ("History could not be read", 503)])
def test_native_import_errors_are_actionable(client, monkeypatch, message, status):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")

    def unavailable(_sid):
        raise server_mod.NativeHistoryError(message)

    monkeypatch.setattr(server_mod, "get_native_session", unavailable)
    result = client.get("/v1/agent/sessions/oc-ses_fixture", headers=AUTH)
    assert result.status_code == status and result.json()["detail"] == message


def test_harness_links_survive_browser_saves_without_reading_or_changing_runs(tmp_path):
    store = SessionStore(tmp_path)
    store.save("session-work", {"title": "Synthetic conversation", "runs": [{"task": "Synthetic task"}]})
    before = store.get("session-work")
    store.link_harness_session("session-work", "oc-ses_first")
    store.link_harness_session("session-work", "oc-ses_first")
    store.link_harness_session("session-work", "hermes-second")
    assert store.get("session-work") == before
    # A stale browser save cannot erase separately persisted associations.
    store.save("session-work", before)
    assert store.list()[0]["harness_sessions"] == ["oc-ses_first", "hermes-second"]
    assert SessionStore(tmp_path).list()[0]["harness_sessions"] == ["oc-ses_first", "hermes-second"]
    for bad in ("../unsafe", "session-other", None):
        with pytest.raises(ValueError):
            store.link_harness_session("session-work", bad)
    assert store.delete("session-work")
    assert not (tmp_path / "harness-links" / "session-work.json").exists()


def test_generated_harness_sessions_are_represented_once_by_work(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    native = [{"id": f"oc-ses_turn{i}", "title": "Synthetic generated", "updated": 1,
               "harness": "opencode", "native": True, "folder": str(tmp_path), "tasks": 1} for i in range(6)]
    monkeypatch.setattr(server_mod, "list_native_sessions", lambda: (native, []))
    def no_transcript_access(_sid):
        raise AssertionError("Listing or linking must never load native messages")
    monkeypatch.setattr(server_mod, "get_native_session", no_transcript_access)
    seen = []
    async def fake_run(**kwargs):
        seen.append(kwargs)
        yield {"type": "harness_session", "session_id": native[len(seen)-1]["id"]}
        yield {"type": "done"}
    monkeypatch.setattr(server_mod, "run_opencode", fake_run)
    client.put("/v1/agent/sessions/session-work", json={"title": "Synthetic Work", "runs": [{"task": "Synthetic task"}]}, headers=AUTH)
    body = {"task": "Actual synthetic title", "model": "fixture", "harness": "opencode", "root": str(tmp_path), "work_session_id": "session-work",
            "history": [{"role": "user", "content": "Synthetic earlier context"}]}
    for _ in native:
        response = client.post("/v1/agent", json=body, headers=AUTH)
        assert response.status_code == 200 and '"type": "harness_session"' in response.text
    assert seen[0]["session_title"] == body["task"]
    assert seen[0]["task"] != body["task"]  # composed context is never used as the title
    # Save again after generation, just as the browser does after each turn.
    client.put("/v1/agent/sessions/session-work", json={"title": "Synthetic Work"}, headers=AUTH)
    listed = client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"]
    assert [s["id"] for s in listed] == ["session-work"]
    assert len(listed[0]["harness_sessions"]) == 6
    assert {s["id"] for s in listed[0]["harness_history"]} == {s["id"] for s in native}
    assert client.post("/v1/agent", json={**body, "work_session_id": "../bad"}, headers=AUTH).status_code == 400
    # Deleting Work never deletes originals; their native entries become visible again.
    client.delete("/v1/agent/sessions/session-work", headers=AUTH)
    assert len(client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"]) == 6


def test_unsaved_work_link_never_hides_native_history(client, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
    native = {"id": "oc-ses_original", "title": "Synthetic original", "updated": 1}
    monkeypatch.setattr(server_mod, "list_native_sessions", lambda: ([native], []))
    client.app.state.work_sessions.link_harness_session("session-missing", native["id"])
    assert client.get("/v1/agent/sessions", headers=AUTH).json()["sessions"] == [native]
