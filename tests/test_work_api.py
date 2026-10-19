from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from helpers import ConfigRouter
from test_server_chats import FakeUpstreamManager

import arc_llama.server as server_mod
from arc_llama.config import Config, ModelConfig
from arc_llama.workbench.runner import WorkManager

AUTH = {"Authorization": "Bearer work-test-token"}


@pytest.fixture
def work_app(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.server.admin_token = "work-test-token"
    cfg.tune.auto = False
    cfg.models = [ModelConfig(name="local", path="unused.gguf", gpu_pci_slot="unused", port=18089)]
    monkeypatch.setattr(server_mod, "Router", ConfigRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)

    async def generate(self, job):
        if job["turns"] == 0:
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"note.txt","content":"work copy"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Finished file edit; no tests run"}

    monkeypatch.setattr(WorkManager, "generate", generate)
    source = tmp_path / "project"
    source.mkdir()
    (source / "note.txt").write_text("original")
    return cfg, source


def test_work_api_requires_admin_and_applies_only_reviewed_changes(work_app):
    cfg, source = work_app
    with TestClient(server_mod.create_app(cfg, plugins=[])) as client:
        assert client.get("/work").status_code == 200
        assert client.get("/work.js").status_code == 200
        assert client.get("/admin/work").status_code == 401
        assert (
            client.post(
                "/admin/work/workspaces", json={"root": str(source), "model": "local"}
            ).status_code
            == 401
        )
        body = {"name": "My project", "root": str(source), "model": "local", "verify": False}
        assert (
            client.post(
                "/admin/work/workspaces", json={**body, "model": "external/model"}, headers=AUTH
            ).status_code
            == 400
        )
        project = client.post("/admin/work/workspaces", json=body, headers=AUTH).json()
        job = client.post(
            "/admin/work/jobs",
            json={"workspace_id": project["id"], "task": "Change note.txt"},
            headers=AUTH,
        ).json()
        identity = job["id"]
        for _ in range(100):
            status = client.get(f"/admin/work/jobs/{identity}", headers=AUTH).json()
            if status["state"] == "completed":
                break
            time.sleep(0.01)
        assert status["state"] == "completed", status
        assert (source / "note.txt").read_text() == "original"
        preview = client.get(f"/admin/work/jobs/{identity}/changes", headers=AUTH).json()
        assert preview["files"] == ["note.txt"]
        assert "+work copy" in preview["diff"]
        assert (
            client.post(f"/admin/work/jobs/{identity}/apply", json={}, headers=AUTH).status_code
            == 409
        )
        assert (
            client.post(
                f"/admin/work/jobs/{identity}/apply",
                json={"review_revision": preview["revision"]},
                headers=AUTH,
            ).status_code
            == 200
        )
        assert (source / "note.txt").read_text() == "work copy"
        assert (
            client.post(f"/admin/work/jobs/{identity}/undo", json={}, headers=AUTH).status_code
            == 200
        )
        assert (source / "note.txt").read_text() == "original"
        assert client.get("/admin/work/jobs/../../outside", headers=AUTH).status_code in (404, 405)
    with TestClient(server_mod.create_app(cfg, plugins=[])) as client:
        status = client.get(f"/admin/work/jobs/{identity}", headers=AUTH).json()
        assert status["state"] == "completed"
        assert any(event["type"] == "undo" for event in status["events"])
        assert client.get("/admin/work", headers=AUTH).json()["workspaces"][0]["root"] == str(
            source
        )


def test_review_cannot_describe_a_copy_mid_replacement(work_app):
    cfg, source = work_app
    with TestClient(server_mod.create_app(cfg, plugins=[])) as client:
        project = client.post(
            "/admin/work/workspaces",
            json={"root": str(source), "model": "local", "verify": False},
            headers=AUTH,
        ).json()
        job = client.post(
            "/admin/work/jobs",
            json={"workspace_id": project["id"], "task": "Change note.txt"},
            headers=AUTH,
        ).json()
        for _ in range(100):
            job = client.get(f"/admin/work/jobs/{job['id']}", headers=AUTH).json()
            if job["state"] == "completed":
                break
            time.sleep(0.01)
        assert job["state"] == "completed"
        manager = client.app.state.work
        saved = manager.store.get("job", job["id"])
        saved["state"] = "settling"
        manager.store.save("job", saved)
        response = client.get(f"/admin/work/jobs/{job['id']}/changes", headers=AUTH)
        assert response.status_code == 409
        assert "being replaced" in response.json()["detail"]
        for action in ("apply", "discard"):
            assert (
                client.post(
                    f"/admin/work/jobs/{job['id']}/{action}", json={}, headers=AUTH
                ).status_code
                == 409
            )
        assert (source / "note.txt").read_text() == "original"
