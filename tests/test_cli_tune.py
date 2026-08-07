"""Tests for `arc-llama tune` CLI — dry-run must not persist tune state."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, ServerConfig
from arc_llama.tune import TuneReport


def _make_test_config(tmp_path: Path) -> Config:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00" * 16)
    return Config(
        server=ServerConfig(host="127.0.0.1", port=11437),
        paths=PathsConfig(models_dir=str(tmp_path / "models")),
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
        models=[
            ModelConfig(
                name="m",
                path=str(gguf),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            )
        ],
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestTuneDryRun:
    """Regression tests for issue #22."""

    def test_tune_default_applies_state(self, monkeypatch, tmp_path, cli_runner):
        """`arc-llama tune m` (default --apply) must persist tuned state."""
        config_file = tmp_path / "config.toml"
        cfg = _make_test_config(tmp_path)
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.load_config", lambda path: cfg)
        async def fake_tune_model(*a, **kw):
            return TuneReport(model="m", target="balanced", applied=True)

        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.tune_model", fake_tune_model)
        save_spy = MagicMock()
        monkeypatch.setattr(cfg, "save", save_spy)
        state_spy = MagicMock()
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.set_tuned_state", state_spy)

        result = cli_runner.invoke(cli, ["--config", str(config_file), "tune", "m"])

        assert result.exit_code == 0, result.output
        state_spy.assert_called_once()
        save_spy.assert_called_once_with(config_file)

    def test_tune_dry_run_does_not_apply_state(self, monkeypatch, tmp_path, cli_runner):
        """`arc-llama tune m --dry-run` must NOT persist tuned state."""
        config_file = tmp_path / "config.toml"
        cfg = _make_test_config(tmp_path)
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.load_config", lambda path: cfg)
        async def fake_tune_model(*a, **kw):
            return TuneReport(model="m", target="balanced", applied=False)

        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.tune_model", fake_tune_model)
        save_spy = MagicMock()
        monkeypatch.setattr(cfg, "save", save_spy)
        state_spy = MagicMock()
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.set_tuned_state", state_spy)

        result = cli_runner.invoke(cli, ["--config", str(config_file), "tune", "m", "--dry-run"])

        assert result.exit_code == 0, result.output
        state_spy.assert_not_called()
        save_spy.assert_not_called()

    def test_tune_all_default_applies_state(self, monkeypatch, tmp_path, cli_runner):
        """`arc-llama tune --all` (default --apply) must persist tuned state."""
        config_file = tmp_path / "config.toml"
        cfg = _make_test_config(tmp_path)
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.load_config", lambda path: cfg)
        async def fake_tune_all(*a, **kw):
            return [TuneReport(model="m", target="balanced", applied=True)]

        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.tune_all", fake_tune_all)
        save_spy = MagicMock()
        monkeypatch.setattr(cfg, "save", save_spy)
        state_spy = MagicMock()
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.set_tuned_state", state_spy)

        result = cli_runner.invoke(cli, ["--config", str(config_file), "tune", "--all"])

        assert result.exit_code == 0, result.output
        state_spy.assert_called_once()
        save_spy.assert_called_once_with(config_file)

    def test_tune_all_dry_run_does_not_apply_state(self, monkeypatch, tmp_path, cli_runner):
        """`arc-llama tune --all --dry-run` must NOT persist tuned state."""
        config_file = tmp_path / "config.toml"
        cfg = _make_test_config(tmp_path)
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.load_config", lambda path: cfg)
        async def fake_tune_all(*a, **kw):
            return [TuneReport(model="m", target="balanced", applied=False)]

        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.tune_all", fake_tune_all)
        save_spy = MagicMock()
        monkeypatch.setattr(cfg, "save", save_spy)
        state_spy = MagicMock()
        monkeypatch.setattr("arc_llama.cli_commands.bench_tune.set_tuned_state", state_spy)

        result = cli_runner.invoke(
            cli, ["--config", str(config_file), "tune", "--all", "--dry-run"]
        )

        assert result.exit_code == 0, result.output
        state_spy.assert_not_called()
        save_spy.assert_not_called()
