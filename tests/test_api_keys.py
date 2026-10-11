from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner
from httpx import ASGITransport, AsyncClient

from arc_llama.api_keys import ApiKeyStore
from arc_llama.cli import _prepare_lan, cli
from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig


def test_store_create_verify_revoke(tmp_path) -> None:
    store = ApiKeyStore.for_state_dir(tmp_path)
    assert not store
    key, plaintext = store.create("phone")
    assert plaintext.startswith("arc_")
    raw = (tmp_path / "api-keys.json").read_text()
    assert plaintext not in raw
    if os.name != "nt":
        assert (tmp_path / "api-keys.json").stat().st_mode & 0o777 == 0o600
    assert store.verify(plaintext) is not None
    assert store.verify("arc_wrong") is None
    assert store.verify("") is None
    [listed] = store.list()
    assert listed["requests"] == 1 and "sha256" not in listed
    store.flush()
    reloaded = ApiKeyStore.for_state_dir(tmp_path)
    assert reloaded.list()[0]["requests"] == 1
    assert store.revoke(key.id)
    assert not store.revoke(key.id)
    assert store.verify(plaintext) is None


def test_store_sees_keys_written_by_another_process(tmp_path) -> None:
    server_side = ApiKeyStore.for_state_dir(tmp_path)
    assert not server_side
    _key, plaintext = ApiKeyStore.for_state_dir(tmp_path).create("cli")
    assert server_side.verify(plaintext) is not None
    ApiKeyStore.for_state_dir(tmp_path).revoke(_key.id)
    assert server_side.verify(plaintext) is None


def test_store_rejects_bad_names_and_survives_corrupt_file(tmp_path) -> None:
    store = ApiKeyStore.for_state_dir(tmp_path)
    for bad in ("", "   ", "x" * 65):
        with pytest.raises(ValueError):
            store.create(bad)
    (tmp_path / "api-keys.json").write_text("{not json")
    assert not ApiKeyStore.for_state_dir(tmp_path)
    (tmp_path / "api-keys.json").write_text(json.dumps({"keys": [{"id": "x"}, "junk"]}))
    assert not ApiKeyStore.for_state_dir(tmp_path)


def _app(monkeypatch, tmp_path):
    import arc_llama.server as server_mod

    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(server=ServerConfig(admin_token="admin-tok"))
    cfg.paths.state_dir = str(tmp_path)
    return server_mod.create_app(cfg, plugins=[])


def _client(app, peer: str) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app, client=(peer, 5555)), base_url="http://t")


async def test_remote_clients_need_a_key_once_one_exists(monkeypatch, tmp_path) -> None:
    app = _app(monkeypatch, tmp_path)
    async with app.router.lifespan_context(app):
        async with _client(app, "192.168.1.20") as remote, _client(app, "127.0.0.1") as local:
            # No keys yet: unchanged, open behaviour.
            assert (await remote.get("/v1/models")).status_code == 200
            created = await local.post(
                "/admin/api-keys",
                json={"name": "phone"},
                headers={"Authorization": "Bearer admin-tok"},
            )
            assert created.status_code == 200
            plaintext = created.json()["key"]
            denied = await remote.get("/v1/models")
            assert denied.status_code == 401
            assert denied.json()["error"]["code"] == "invalid_api_key"
            assert (await remote.get("/api/tags")).status_code == 401
            bad = await remote.get("/v1/models", headers={"Authorization": "Bearer nope"})
            assert bad.status_code == 401
            ok = await remote.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
            assert ok.status_code == 200
            admin = await remote.get(
                "/v1/models", headers={"Authorization": "Bearer admin-tok"}
            )
            assert admin.status_code == 200
            # Loopback, health and the UI stay open.
            assert (await local.get("/v1/models")).status_code == 200
            assert (await remote.get("/health")).status_code == 200
            listed = await local.get(
                "/admin/api-keys", headers={"Authorization": "Bearer admin-tok"}
            )
            [entry] = listed.json()["keys"]
            assert entry["requests"] == 1 and "key" not in entry
            revoked = await local.delete(
                f"/admin/api-keys/{entry['id']}", headers={"Authorization": "Bearer admin-tok"}
            )
            assert revoked.status_code == 200
            missing = await local.delete(
                "/admin/api-keys/nope", headers={"Authorization": "Bearer admin-tok"}
            )
            assert missing.status_code == 404
            # Last key revoked: remote access is open again, as documented.
            assert (await remote.get("/v1/models")).status_code == 200


async def test_key_admin_routes_need_admin_token(monkeypatch, tmp_path) -> None:
    app = _app(monkeypatch, tmp_path)
    async with app.router.lifespan_context(app):
        async with _client(app, "127.0.0.1") as local:
            assert (await local.get("/admin/api-keys")).status_code == 401
            assert (await local.post("/admin/api-keys", json={"name": "x"})).status_code == 401
            bad = await local.post(
                "/admin/api-keys", json={"name": 5}, headers={"Authorization": "Bearer admin-tok"}
            )
            assert bad.status_code == 400


def test_keys_cli(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.save(config_path)
    runner = CliRunner()
    base = ["--config", str(config_path), "keys"]
    empty = runner.invoke(cli, [*base, "list"], obj={})
    assert "No API keys" in empty.output
    created = runner.invoke(cli, [*base, "create", "laptop"], obj={})
    assert created.exit_code == 0, created.output
    plaintext = next(line for line in created.output.splitlines() if line.startswith("arc_"))
    store = ApiKeyStore.for_state_dir(tmp_path / "state")
    assert store.verify(plaintext) is not None
    listed = runner.invoke(cli, [*base, "list"], obj={})
    assert "laptop" in listed.output
    key_id = store.list()[0]["id"]
    assert runner.invoke(cli, [*base, "revoke", str(key_id)], obj={}).exit_code == 0
    assert runner.invoke(cli, [*base, "revoke", str(key_id)], obj={}).exit_code != 0


def test_prepare_lan_creates_a_key_once(tmp_path, monkeypatch) -> None:
    import arc_llama.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_lan_addresses", lambda: ["192.168.1.5"])
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path)
    _prepare_lan(cfg, 11437, assume_yes=True)
    _prepare_lan(cfg, 11437, assume_yes=True)
    keys = ApiKeyStore.for_state_dir(tmp_path).list()
    assert [k["name"] for k in keys] == ["lan"]


def test_serve_lan_refuses_without_confirmation(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    cfg = Config()
    cfg.paths.state_dir = str(tmp_path / "state")
    cfg.save(config_path)
    result = CliRunner().invoke(
        cli, ["--config", str(config_path), "serve", "--lan", "--no-scan"], input="n\n", obj={}
    )
    assert result.exit_code != 0
    assert not (tmp_path / "state" / "api-keys.json").exists()
    conflict = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "serve", "--lan", "--host", "10.0.0.2"],
        obj={},
    )
    assert conflict.exit_code != 0


def test_backends_always_bind_loopback(tmp_path) -> None:
    from arc_llama.router import Router

    model_path = tmp_path / "m.gguf"
    model_path.write_bytes(b"GGUF")
    cfg = Config(
        server=ServerConfig(host="0.0.0.0"),
        gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", vram_mb=24576, backend="vulkan")],
        models=[ModelConfig("m", str(model_path), 18555, "0000:03:00.0")],
    )
    cfg.paths.llama_server = str(tmp_path / "missing-llama-server")
    plan = Router(cfg)._servers["m"].plan
    assert plan.argv[plan.argv.index("--host") + 1] == "127.0.0.1"
    assert plan.backend_url == "http://127.0.0.1:18555"
