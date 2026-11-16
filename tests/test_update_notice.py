from __future__ import annotations

import asyncio

import httpx
import pytest

import arc_llama.update_notice as updates
from arc_llama.config import Config, ServerConfig, load_config


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.tune.auto = False
    monkeypatch.setattr(updates, "__version__", "1.2.0")
    wire = {"payload": {"releases": {"1.3.0": [{}]}}, "error": None, "calls": [], "clients": []}
    original = httpx.AsyncClient
    def handler(request):
        wire["calls"].append(request)
        if wire["error"]:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json=wire["payload"])
    def client(**kwargs):
        wire["clients"].append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(updates.httpx, "AsyncClient", client)
    return cfg, wire


@pytest.mark.parametrize("version,newer", [("1.3.0", True), ("1.2.0", False), ("1.1.9", False), ("1.2.0.post1", True)])
async def test_version_comparison_uses_stable_pep440_versions(setup, version, newer):
    cfg, wire = setup
    wire["payload"] = {"releases": {version: [{}]}}
    notice = await updates.check_for_update(cfg)
    assert bool(notice) is newer
    if newer:
        assert notice == {"current": "1.2.0", "latest": version, "command": "pip install -U arc-llama"}
    assert wire["calls"][0].url == updates.PYPI_URL
    assert wire["clients"] == [{"timeout": 2.0, "follow_redirects": False}]
    assert "authorization" not in wire["calls"][0].headers


def test_latest_release_ignores_prereleases_dev_yanked_and_invalid_versions():
    payload = {"info": {"version": "9.0.0rc1"}, "releases": {
        "1.2.0": [{}], "1.10.0": [{}], "2.0.0rc1": [{}], "3.0.0.dev1": [{}],
        "9.0.0": [{"yanked": True}], "10.0.0": [], "bad-version": [{}],
    }}
    assert updates.latest_stable(payload) == "1.10.0"
    assert updates.latest_stable({"releases": {"2.0.0rc1": [{}]}}) is None
    assert updates.latest_stable({"releases": []}) is None
    assert updates.latest_stable(None) is None


async def test_daily_cache_survives_new_config_instances_and_rolls_after_24h(setup, monkeypatch):
    cfg, wire = setup
    now = [1_000_000.0]
    monkeypatch.setattr(updates, "time", lambda: now[0])
    first = await updates.check_for_update(cfg)
    again = Config()
    again.paths.state_dir = cfg.paths.state_dir
    now[0] += updates.CHECK_INTERVAL - 1
    assert await updates.check_for_update(again) == first
    assert len(wire["calls"]) == 1
    now[0] += 1
    wire["payload"] = {"releases": {"1.4.0": [{}]}}
    assert (await updates.check_for_update(cfg))["latest"] == "1.4.0"
    assert len(wire["calls"]) == 2
    assert updates.cached_notice(again)["latest"] == "1.4.0"


async def test_concurrent_checks_claim_one_request(setup):
    cfg, wire = setup
    notices = await asyncio.gather(*(updates.check_for_update(cfg) for _ in range(8)))
    assert len(wire["calls"]) == 1
    assert any(notice and notice["latest"] == "1.3.0" for notice in notices)


async def test_offline_is_silent_and_cached_for_a_day(setup, monkeypatch, capsys):
    cfg, wire = setup
    wire["error"] = True
    now = [1_000_000.0]
    monkeypatch.setattr(updates, "time", lambda: now[0])
    assert await updates.check_for_update(cfg) is None
    assert await updates.check_for_update(cfg) is None
    assert len(wire["calls"]) == 1
    assert capsys.readouterr() == ("", "")
    now[0] += updates.CHECK_INTERVAL
    wire["error"] = None
    assert (await updates.check_for_update(cfg))["latest"] == "1.3.0"
    assert len(wire["calls"]) == 2


async def test_disabled_and_unwritable_cache_never_request(setup, tmp_path):
    cfg, wire = setup
    cfg.server.check_updates = False
    assert await updates.check_for_update(cfg) is None
    assert not updates._path(cfg).exists()
    cfg.server.check_updates = True
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("file")
    cfg.paths.state_dir = str(blocked)
    assert await updates.check_for_update(cfg) is None
    assert wire["calls"] == []


def test_check_switch_defaults_on_and_round_trips(tmp_path):
    assert ServerConfig().check_updates is True
    cfg = Config()
    cfg.server.check_updates = False
    path = tmp_path / "config.toml"
    cfg.save(path)
    assert load_config(path).server.check_updates is False
    with pytest.raises(ValueError, match="check_updates"):
        ServerConfig(check_updates="false")


def test_endpoint_requires_admin_and_honors_disabled_switch(setup, monkeypatch):
    from fastapi.testclient import TestClient
    from helpers import ConfigRouter

    import arc_llama.server as server

    cfg, wire = setup
    cfg.server.admin_token = "version-test-token"
    cfg.server.check_updates = False
    monkeypatch.setattr(server, "Router", ConfigRouter)
    with TestClient(server.create_app(cfg, plugins=[])) as client:
        assert client.get("/admin/update-notice").status_code == 401
        auth = {"Authorization": "Bearer version-test-token"}
        assert client.get("/admin/update-notice", headers=auth).json() == {"update": None}
        assert wire["calls"] == []
        cfg.server.check_updates = True
        assert client.get("/admin/update-notice", headers=auth).json()["update"]["latest"] == "1.3.0"
        assert client.get("/admin/update-notice", headers=auth).json()["update"]["latest"] == "1.3.0"
        assert len(wire["calls"]) == 1


def test_doctor_prints_cached_notice_without_network(setup, tmp_path, monkeypatch):
    from click.testing import CliRunner

    import arc_llama.cli as cli_mod
    import arc_llama.cli_diagnostics as diagnostics
    import arc_llama.plugins as plugins

    cfg, wire = setup
    path = tmp_path / "config.toml"
    path.touch()
    stamp = updates.time()
    updates._claim(updates._path(cfg), stamp)
    updates._save(updates._path(cfg), stamp, "1.3.0")
    monkeypatch.setattr(cli_mod, "load_config", lambda _: cfg)
    monkeypatch.setattr(cli_mod, "detect_gpus", lambda **_: [])
    monkeypatch.setattr(cli_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(diagnostics, "lspci_intel_gpus", lambda: "")
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: None)
    monkeypatch.setattr(diagnostics, "level_zero_loader_present", lambda: (False, ""))
    monkeypatch.setattr(diagnostics, "oneapi_setvars_path", lambda: None)
    monkeypatch.setattr(plugins, "plugin_health", lambda: [])
    args = ["-c", str(path), "doctor"]
    result = CliRunner().invoke(cli_mod.cli, args)
    assert result.exit_code == 2  # no synthetic GPU
    assert "Arc Llama 1.3.0 available" in result.output
    assert "pip install -U arc-llama" in result.output
    cfg.server.check_updates = False
    assert "update:" not in CliRunner().invoke(cli_mod.cli, args).output
    assert wire["calls"] == []
