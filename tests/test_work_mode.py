"""Saved Work opt-in, authenticated writes and CLI visibility without re-imports."""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from helpers import ConfigRouter

import arc_llama.server as server_mod
from arc_llama.agent.harness import experimental_enabled
from arc_llama.cli import cli
from arc_llama.config import Config, ServerConfig, load_config


@pytest.mark.parametrize("setting,env,expected", [
    (False, "", False), (True, "", True), (False, "1", True),
    (False, "TRUE", True), (False, "yes", True), (False, "on", False),
])
def test_work_opt_in(setting, env, expected, monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", env)
    assert experimental_enabled(Config(server=ServerConfig(work_mode=setting))) is expected


def test_work_setting_persists_and_updates_flags_without_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_EXPERIMENTAL_AGENT", raising=False)
    monkeypatch.setattr(server_mod, "Router", ConfigRouter)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    cfg.paths.state_dir = str(tmp_path / "state")
    path = tmp_path / "config.toml"
    app = server_mod.create_app(cfg, config_path=path)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert '"work": false' in client.get("/ui-flags.js").text
        assert client.put("/admin/settings", json={"work_mode": True}).status_code == 401
        assert client.put("/admin/settings", json={"work_mode": True}, headers={"Authorization": "Bearer wrong"}).status_code == 403
        assert not path.exists()
        assert client.put("/admin/settings", json={"work_mode": "yes"}, headers=headers).status_code == 400
        response = client.put("/admin/settings", json={"work_mode": True}, headers=headers)
        assert response.json() == {"work_mode": True, "experimental": True}
        assert load_config(path).server.work_mode is True
        assert '"work": true' in client.get("/ui-flags.js").text
        assert client.get("/v1/agent/info", headers=headers).json()["experimental"] is True
        monkeypatch.setenv("ARC_LLAMA_EXPERIMENTAL_AGENT", "1")
        response = client.put("/admin/settings", json={"work_mode": False}, headers=headers)
        assert response.json() == {"work_mode": False, "experimental": True}
        assert load_config(path).server.work_mode is False


def test_cli_rechecks_saved_setting_and_explicit_config(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_EXPERIMENTAL_AGENT", raising=False)
    path = tmp_path / "config.toml"
    cfg = Config()
    cfg.save(path)
    runner = CliRunner()
    assert "agent-tui" not in runner.invoke(cli, ["--config", str(path), "--help"]).output
    assert runner.invoke(cli, ["--config", str(path), "agent", "--help"]).exit_code != 0
    cfg.server.work_mode = True
    cfg.save(path)
    assert "agent-tui" in runner.invoke(cli, ["--config", str(path), "--help"]).output
    assert runner.invoke(cli, ["--config", str(path), "agent", "--help"]).exit_code == 0
