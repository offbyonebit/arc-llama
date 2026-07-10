"""Tests for Open WebUI / LMStudio integration helpers (PR #10).

Covers CORS, /v1/models created timestamps, and serve env-var bindings.
Full docker compose + Open WebUI UI is manual (see PR test plan).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from arc_llama.cli import cli, serve as serve_cmd
from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig, UpstreamConfig
from arc_llama.server import create_app


class FakeRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self._servers = {}

    def all_models(self):
        return list(self.cfg.models)

    async def shutdown(self):
        return None


class FakeUpstreamManager:
    def __init__(self, upstreams):
        self.upstreams = upstreams

    async def models(self):
        return [
            SimpleNamespace(
                id="lmstudio-model",
                upstream_name=u.name,
                metadata={},
            )
            for u in self.upstreams
        ]


def _cfg(gguf: Path, *, upstream: bool = True) -> Config:
    return Config(
        server=ServerConfig(host="127.0.0.1", port=11437, admin_token=None),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[ModelConfig(
            name="qwen",
            path=str(gguf),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            aliases=["qwen.gguf"],
        )],
        upstreams=(
            [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
            if upstream else []
        ),
    )


@pytest.fixture
def gguf(tmp_path: Path) -> Path:
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00" * 32)
    os.utime(p, (1_700_000_123, 1_700_000_123))
    return p


def test_cors_allows_cross_origin(gguf: Path):
    import arc_llama.server as server_mod

    with patch.object(server_mod, "Router", FakeRouter), \
         patch.object(server_mod, "UpstreamManager", FakeUpstreamManager):
        app = create_app(_cfg(gguf))
        with TestClient(app) as client:
            pre = client.options(
                "/v1/models",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert pre.status_code in (200, 204)
            assert pre.headers.get("access-control-allow-origin") == "*"
            r = client.get("/v1/models", headers={"Origin": "http://localhost:3000"})
            assert r.status_code == 200
            assert r.headers.get("access-control-allow-origin") == "*"


def test_list_models_created_from_gguf_mtime(gguf: Path):
    import arc_llama.server as server_mod

    with patch.object(server_mod, "Router", FakeRouter), \
         patch.object(server_mod, "UpstreamManager", FakeUpstreamManager):
        app = create_app(_cfg(gguf))
        with TestClient(app) as client:
            data = client.get("/v1/models").json()["data"]
            local = next(m for m in data if m["id"] == "qwen")
            assert local["created"] == 1_700_000_123
            alias = next(m for m in data if m["id"] == "qwen.gguf")
            assert alias["created"] == 1_700_000_123
            assert any(m["owned_by"] == "upstream:lmstudio" for m in data)


def test_list_models_missing_file_created_zero(tmp_path: Path):
    import arc_llama.server as server_mod

    missing = tmp_path / "gone.gguf"
    with patch.object(server_mod, "Router", FakeRouter), \
         patch.object(server_mod, "UpstreamManager", FakeUpstreamManager):
        app = create_app(_cfg(missing, upstream=False))
        with TestClient(app) as client:
            local = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "qwen")
            assert local["created"] == 0


def test_serve_options_read_env_vars():
    host_opt = next(p for p in serve_cmd.params if p.name == "host")
    port_opt = next(p for p in serve_cmd.params if p.name == "port")
    assert host_opt.envvar == "ARC_LLAMA_HOST"
    assert port_opt.envvar == "ARC_LLAMA_PORT"
    result = CliRunner().invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "ARC_LLAMA_HOST" in result.output
    assert "ARC_LLAMA_PORT" in result.output


def test_upstream_add_lmstudio(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    Config().save(cfg_path)
    result = CliRunner().invoke(cli, [
        "--config", str(cfg_path),
        "upstream", "add", "lmstudio", "http://127.0.0.1:1234",
    ])
    assert result.exit_code == 0, result.output
    from arc_llama.config import load_config
    loaded = load_config(cfg_path)
    assert any(u.name == "lmstudio" and "1234" in u.url for u in loaded.upstreams)


def test_docker_compose_yml_present():
    root = Path(__file__).resolve().parents[1]
    compose = root / "docker-compose.yml"
    assert compose.is_file()
    text = compose.read_text()
    assert "open-webui" in text
    assert "ARC_LLAMA_HOST" in text
    assert "OPENAI_API_BASE_URL" in text
    assert "/dev/dri" in text
    # Prefer python healthcheck (curl not in runtime image)
    assert "python3" in text
