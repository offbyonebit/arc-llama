"""Tests for the /v1/chats server endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient
from helpers import ConfigRouter as FakeRouter

import arc_llama.server as server_mod
from arc_llama.config import Config
from arc_llama.server import create_app


class FakeUpstreamManager:
    def __init__(self, upstreams=None):
        pass

    async def models(self):
        return []

    def find_model(self, model_id):
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        raise RuntimeError("should not be called")

    def upstreams_status(self):
        return []


def _app(tmp_path):
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    app = create_app(cfg)
    return app


def test_create_list_get_delete_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        create_resp = client.post("/v1/chats", json={"id": "chat-1", "title": "Planning"})
        assert create_resp.status_code == 200
        assert create_resp.json()["id"] == "chat-1"
        assert create_resp.json()["title"] == "Planning"

        list_resp = client.get("/v1/chats")
        assert list_resp.status_code == 200
        data = list_resp.json()["data"]
        assert len(data) == 1
        assert data[0]["id"] == "chat-1"
        assert data[0]["message_count"] == 0

        get_resp = client.get("/v1/chats/chat-1")
        assert get_resp.status_code == 200
        assert get_resp.json()["title"] == "Planning"

        delete_resp = client.delete("/v1/chats/chat-1")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted"] is True

        assert client.get("/v1/chats/chat-1").status_code == 404


def test_patch_appends_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "T"})
        patch_resp = client.patch(
            "/v1/chats/chat-1",
            json={
                "title": "Renamed",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            },
        )
        assert patch_resp.status_code == 200
        body = patch_resp.json()
        assert body["title"] == "Renamed"
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == "world"


def test_search_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "Rust ideas"})
        client.patch(
            "/v1/chats/chat-1",
            json={"messages": [{"role": "user", "content": "I want to learn rust"}]},
        )

        resp = client.post("/v1/chats/search", json={"query": "rust"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["chat"]["id"] == "chat-1"
        assert data[0]["matching_message_indices"] == [-1, 0]


def test_export_and_import_chats(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "Plan"})
        client.patch(
            "/v1/chats/chat-1",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        export_resp = client.get("/v1/chats/export")
        assert export_resp.status_code == 200
        payload = export_resp.json()
        assert payload["version"] == 1
        assert len(payload["chats"]) == 1
        assert payload["chats"][0]["id"] == "chat-1"

        import_resp = client.post(
            "/v1/chats/import",
            json={"chats": payload["chats"], "overwrite": False},
        )
        assert import_resp.status_code == 200
        summary = import_resp.json()
        assert summary["imported"] == 0
        assert summary["skipped"] == 1

        payload["chats"][0]["id"] = "chat-2"
        import_resp = client.post(
            "/v1/chats/import",
            json={"chats": payload["chats"], "overwrite": False},
        )
        assert import_resp.status_code == 200
        summary = import_resp.json()
        assert summary["imported"] == 1
        assert client.get("/v1/chats/chat-2").status_code == 200


def test_create_and_filter_by_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "root-chat", "title": "Root"})
        client.post("/v1/chats", json={"id": "work-chat", "title": "Work", "folder": "work"})

        all_resp = client.get("/v1/chats")
        assert all_resp.status_code == 200
        assert {c["id"] for c in all_resp.json()["data"]} == {"root-chat", "work-chat"}

        work_resp = client.get("/v1/chats?folder=work")
        assert work_resp.status_code == 200
        assert [c["id"] for c in work_resp.json()["data"]] == ["work-chat"]


def test_folders_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "a", "title": "A"})
        client.post("/v1/chats", json={"id": "b", "title": "B", "folder": "work"})
        client.post("/v1/chats", json={"id": "c", "title": "C", "folder": "work"})

        resp = client.get("/v1/chats/folders")
        assert resp.status_code == 200
        data = {f["name"]: f["count"] for f in resp.json()["data"]}
        assert data.get("") == 1
        assert data.get("work") == 2


def test_move_chat_via_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = _app(tmp_path)

    with TestClient(app) as client:
        client.post("/v1/chats", json={"id": "chat-1", "title": "A", "folder": "work"})
        patch_resp = client.patch("/v1/chats/chat-1", json={"folder": "personal"})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["folder"] == "personal"

        work_resp = client.get("/v1/chats?folder=work")
        assert work_resp.json()["data"] == []

        personal_resp = client.get("/v1/chats?folder=personal")
        assert [c["id"] for c in personal_resp.json()["data"]] == ["chat-1"]


def test_invalid_chat_requests_report_400_and_preserve_saved_history(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, 'Router', FakeRouter)
    monkeypatch.setattr(server_mod, 'UpstreamManager', FakeUpstreamManager)
    with TestClient(_app(tmp_path)) as client:
        assert client.post('/v1/chats', json={'id': 'safe', 'title': 'Original'}).status_code == 200
        for route in ['/v1/chats', '/v1/chats/search', '/v1/chats/import']:
            assert client.post(route, json=[]).status_code == 400
        for method in [client.put, client.patch]:
            for body in [[], {'messages': [None]}, {'messages': 'bad'}, {'messages': [{'content': 123}]}]:
                assert method('/v1/chats/safe', json=body).status_code == 400
        for body in [{'query': 123}, {'query': 'x', 'limit': 'bad'}, {'query': 'x', 'limit': 0}]:
            assert client.post('/v1/chats/search', json=body).status_code == 400
        assert client.post('/v1/chats', json={'id': 'bad', 'folder': '..'}).status_code == 400
        assert client.patch('/v1/chats/safe', json={'folder': '..'}).status_code == 400
        assert client.post('/v1/chats/import', json={'chats': [], 'overwrite': 'false'}).status_code == 400
        saved = client.get('/v1/chats/safe').json()
        assert saved['title'] == 'Original'
        assert saved['messages'] == []
        assert saved['folder'] == ''


def test_chat_storage_failure_reports_retryable_error_and_keeps_history(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(server_mod, 'Router', FakeRouter)
    monkeypatch.setattr(server_mod, 'UpstreamManager', FakeUpstreamManager)
    with TestClient(_app(tmp_path)) as client:
        client.post('/v1/chats', json={'id': 'safe', 'title': 'Original'})
        original_replace = Path.replace
        def failed_replace(path, target):
            if Path(target).name == 'safe.json':
                raise OSError('disk full')
            return original_replace(path, target)
        monkeypatch.setattr(Path, 'replace', failed_replace)
        response = client.patch('/v1/chats/safe', json={'title': 'Changed'})
        assert response.status_code == 503
        assert 'disk space' in response.json()['detail']
        assert client.get('/v1/chats/safe').json()['title'] == 'Original'


def test_invalid_folder_query_returns_client_error(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, 'Router', FakeRouter)
    monkeypatch.setattr(server_mod, 'UpstreamManager', FakeUpstreamManager)
    with TestClient(_app(tmp_path)) as client:
        assert client.get('/v1/chats', params={'folder': '..'}).status_code == 400
