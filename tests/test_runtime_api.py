from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from helpers import FakeServer
from httpx import ASGITransport, AsyncClient

from arc_llama import runtime_update
from arc_llama.config import Config, GPUConfig, ModelConfig, load_config

AUTH = {"Authorization": "Bearer runtime-test-token"}


@pytest.fixture
def installed(tmp_path, monkeypatch):
    import arc_llama.router as router_mod
    import arc_llama.server as server

    root = tmp_path / "state" / "runtime"
    paths = {}
    for tag in ("b100", "b200", "b300"):
        binary = root / tag / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"controlled runtime")
        (binary.parent / ".arc-llama-runtime.json").write_text(json.dumps({"tag": tag}))
        paths[tag] = str(binary)
    (root / "history.json").write_text(json.dumps([
        {"previous": paths["b100"], "installed": paths["b200"], "tag": "b200", "at": 1700000000,
         "previous_backend": "vulkan"},
        {"previous": paths["b200"], "installed": paths["b300"], "tag": "b300", "at": 1700001000,
         "previous_backend": "vulkan"},
    ]))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"controlled model")
    cfg = Config(gpus=[GPUConfig("gpu", 0, "battlemage", backend="vulkan", vram_mb=24576)],
                 models=[ModelConfig("test", str(model), 18080, "gpu")])
    cfg.paths.llama_server = paths["b300"]
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.server.admin_token = "runtime-test-token"
    cfg.tune.auto = False
    config_path = tmp_path / "config.toml"
    cfg.save(config_path)
    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    app = server.create_app(cfg, config_path=config_path, plugins=[])
    return SimpleNamespace(app=app, cfg=cfg, paths=paths, config_path=config_path, root=root)


def test_history_lists_retained_files_and_dates_without_executing(installed):
    setup = installed
    with TestClient(setup.app) as client:
        assert client.get("/admin/runtime/history").status_code == 401
        assert client.post("/admin/runtime/rollback", json={}).status_code == 401
        response = client.get("/admin/runtime/history", headers=AUTH)
        assert response.status_code == 200
        data = response.json()
        assert data["current"]["build"] == "b300"
        assert data["previous"]["build"] == "b200"
        assert data["current"]["installed_at"] == 1700001000
        assert data["current"]["date_source"] == "update history"
        assert {row["build"] for row in data["builds"]} == {"b100", "b200", "b300"}
        oldest = next(row for row in data["builds"] if row["build"] == "b100")
        assert oldest["date_source"] == "install marker timestamp"
        assert oldest["installed_at"] > 0
        assert FakeServer.starts == FakeServer.stops == []


def test_rollback_uses_existing_function_persists_and_rebuilds_loaded_models(installed, monkeypatch):
    setup = installed
    original = runtime_update.rollback_runtime
    calls = []
    def restore(cfg, path):
        calls.append((cfg.paths.llama_server, path))
        return original(cfg, path)
    monkeypatch.setattr(runtime_update, "rollback_runtime", restore)
    with TestClient(setup.app) as client:
        rt = setup.app.state.router
        old_server = rt._servers["test"]
        old_server.running = old_server.ready = True
        response = client.post("/admin/runtime/rollback", headers=AUTH,
                               json={"expected_previous": setup.paths["b200"]})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["to"] == setup.paths["b200"]
        assert data["restarted"] == ["test"] and data["restart_errors"] == []
        assert calls == [(setup.paths["b300"], setup.config_path)]
        assert setup.cfg.paths.llama_server == setup.paths["b200"]
        assert load_config(setup.config_path).paths.llama_server == setup.paths["b200"]
        assert len(runtime_update.read_history(setup.cfg)) == 1
        assert not old_server.is_running
        assert rt._servers["test"] is not old_server
        assert rt._servers["test"].plan.argv[0] == setup.paths["b200"]
        assert rt._servers["test"].is_running and rt._servers["test"].ready
        assert FakeServer.stops == FakeServer.starts == ["test"]
        assert not setup.app.state.runtime_rollback_running
        assert not setup.app.state.resources.exclusive_active


@pytest.mark.parametrize("busy,reason", [("loading", "loading"), ("download", "download"),
                                       ("tune", "tune"), ("setup", "setup"), ("answer", "answering")])
def test_rollback_refuses_busy_before_stopping_anything(installed, busy, reason):
    setup = installed
    with TestClient(setup.app) as client:
        state = setup.app.state
        state.router._servers["test"].running = state.router._servers["test"].ready = True
        if busy == "loading":
            state.router._loading_futures["test"] = object()
        elif busy == "download":
            state.downloads = SimpleNamespace(jobs={"job": SimpleNamespace(status="downloading")}, shutdown=AsyncMock())
        elif busy == "tune":
            state.tuner = SimpleNamespace(is_sweep_running=True)
        elif busy == "setup":
            state.first_answer = SimpleNamespace(phase="runtime", shutdown=AsyncMock())
        else:
            state.router.inflight = 1
        response = client.post("/admin/runtime/rollback", headers=AUTH, json={})
        assert response.status_code == 409 and reason in response.json()["detail"].lower()
        assert FakeServer.stops == []
        assert setup.cfg.paths.llama_server == setup.paths["b300"]


def test_history_handles_missing_old_files_and_rejects_stale_confirmation(installed):
    from pathlib import Path

    setup = installed
    Path(setup.paths["b200"]).unlink()
    with TestClient(setup.app) as client:
        info = client.get("/admin/runtime/history", headers=AUTH).json()
        assert info["previous"]["build"] == "b100"
        assert "b200" not in {row["build"] for row in info["builds"]}
        response = client.post("/admin/runtime/rollback", headers=AUTH,
                               json={"expected_previous": setup.paths["b200"]})
        assert response.status_code == 409 and "changed" in response.json()["detail"]
        assert FakeServer.stops == []
        Path(setup.paths["b100"]).unlink()
        assert client.get("/admin/runtime/history", headers=AUTH).json()["previous"] is None
        assert client.post("/admin/runtime/rollback", headers=AUTH, json={}).status_code == 409


def test_switch_success_reports_model_restart_failure_separately(installed, monkeypatch):
    setup = installed
    with TestClient(setup.app) as client:
        rt = setup.app.state.router
        rt._servers["test"].running = rt._servers["test"].ready = True
        monkeypatch.setattr(rt, "_ensure_active", AsyncMock(side_effect=ValueError("controlled reload failure")))
        response = client.post("/admin/runtime/rollback", headers=AUTH, json={})
        assert response.status_code == 200
        assert response.json()["restart_errors"] == [{"model": "test", "message": "controlled reload failure"}]
        assert setup.cfg.paths.llama_server == setup.paths["b200"]
        assert not setup.app.state.resources.exclusive_active


async def test_rollback_blocks_new_downloads_setup_and_a_second_rollback(installed, monkeypatch):
    setup = installed
    entered, release = asyncio.Event(), asyncio.Event()
    async with setup.app.router.lifespan_context(setup.app):
        await setup.app.state.ready.wait()
        rt = setup.app.state.router
        server = rt._servers["test"]
        server.running = server.ready = True
        original = server.astop
        async def stop(*args, **kwargs):
            entered.set()
            await release.wait()
            await original(*args, **kwargs)
        monkeypatch.setattr(server, "astop", stop)
        async with AsyncClient(transport=ASGITransport(app=setup.app), base_url="http://test") as client:
            operation = asyncio.create_task(client.post("/admin/runtime/rollback", headers=AUTH, json={}))
            try:
                await asyncio.wait_for(entered.wait(), 3)
                assert (await client.post("/admin/library/download", headers=AUTH,
                                         json={"repo": "owner/model", "file": "model.gguf"})).status_code == 409
                assert (await client.post("/admin/onboarding", headers=AUTH,
                                         json={"action": "start", "starter": "test"})).status_code == 409
                assert (await client.post("/admin/runtime/rollback", headers=AUTH, json={})).status_code == 409
            finally:
                release.set()
            assert (await asyncio.wait_for(operation, 3)).status_code == 200
