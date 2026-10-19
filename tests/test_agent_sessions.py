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
