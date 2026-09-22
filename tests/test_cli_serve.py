from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig


def _cfg(models_dir: Path) -> Config:
    models_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        paths=PathsConfig(models_dir=str(models_dir)),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                enabled=True,
                name="Arc Pro B60",
            )
        ],
        models=[],
    )


def _fake_app() -> MagicMock:
    app = MagicMock()
    app.state.router = None  # serve's atexit shutdown no-ops on None
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_serve_auto_registers_new_gguf(runner, tmp_path, monkeypatch):
    """serve discovers a GGUF dropped into models_dir since the last run."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve"]
        )

    assert result.exit_code == 0, result.output
    assert "Auto-registered" in result.output
    assert len(cfg.models) == 1


def test_serve_no_scan_skips_discovery(runner, tmp_path, monkeypatch):
    """--no-scan leaves the registry untouched even with a GGUF present."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve", "--no-scan"]
        )

    assert result.exit_code == 0, result.output
    assert "Auto-registered" not in result.output
    assert cfg.models == []


def test_serve_scan_is_idempotent(runner, tmp_path, monkeypatch):
    """A GGUF already registered is not added again on the next serve."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    patches = (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    )
    for _ in range(2):
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = runner.invoke(
                cli, ["--config", str(tmp_path / "config.toml"), "serve"]
            )
            assert result.exit_code == 0, result.output

    assert len(cfg.models) == 1  # not duplicated on the second run


def test_serve_banner_printed_once(runner, tmp_path, monkeypatch):
    """Round 7 leftover: ``arc-llama serve`` prints its Auto-tune banner exactly once."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    cfg.tune.auto = True
    (models_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve"]
        )

    assert result.exit_code == 0, result.output
    assert result.output.count("Auto-tune") == 1


def test_serve_does_not_print_admin_token(runner, tmp_path, monkeypatch):
    """The startup banner must not leak the admin bearer token."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    cfg.server.admin_token = "super-secret-token"
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    with (
        patch("arc_llama.models.has_mtp_heads", return_value=False),
        patch("arc_llama.models.is_moe", return_value=False),
        patch("arc_llama.server.create_app", return_value=_fake_app()),
        patch("uvicorn.run"),
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = runner.invoke(
            cli, ["--config", str(tmp_path / "config.toml"), "serve"]
        )

    assert result.exit_code == 0, result.output
    assert "super-secret-token" not in result.output
    # Rich hard-wraps the banner to the terminal width, so compare against the
    # output with runs of whitespace collapsed rather than the raw string.
    unwrapped = " ".join(result.output.split())
    assert "Authorization: Bearer <token>" in unwrapped


async def test_serve_load_error_counted_once(tmp_path, monkeypatch):
    """Round 7 leftover: a failed health check increments load_errors exactly once."""
    from conftest import make_config

    import arc_llama.router as router_mod
    from arc_llama.router import Router

    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", NeverReadyServer)
    rt = Router(cfg)

    with pytest.raises(RuntimeError, match="timed out while loading"):
        await rt.ensure_active("qwen")
    assert rt.metrics["load_errors"] == 1


class NeverReadyServer:
    """Stub server that fails the health check and yields a log tail."""

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False
        self.ready = False

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.ready = False

    async def wait_ready(self):
        await asyncio.sleep(0.01)
        return False

    def tail_log(self, lines=50):
        return "boom: failed to bind port"

    def stop(self):
        self.running = False
        self.ready = False


def test_list_models_sends_admin_bearer(runner, tmp_path, monkeypatch):
    """list hits /admin/status with the configured bearer token."""
    models_dir = tmp_path / "models"
    cfg = _cfg(models_dir)
    cfg.server.admin_token = "secret-token"
    cfg.models = [
        ModelConfig(
            name="qwen",
            path=str(models_dir / "qwen.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
        ),
    ]
    cfg_path = tmp_path / "config.toml"
    cfg.save(cfg_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    calls = []

    def fake_get(url, *, timeout, headers=None):
        calls.append((url, headers))
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"models": [{"name": "qwen", "loaded": True}]}
        return Resp()

    monkeypatch.setattr("arc_llama.cli.httpx.get", fake_get)
    result = runner.invoke(cli, ["--config", str(cfg_path), "list"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][1] == {"Authorization": "Bearer secret-token"}
    assert "loaded" in result.output
