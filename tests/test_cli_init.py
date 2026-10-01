from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, PathsConfig


def _make_test_config(tmp_path: Path) -> Config:
    return Config(
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
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_init_fails_when_llama_server_binary_missing(monkeypatch, tmp_path, cli_runner):
    """`arc-llama init` must exit with a clear error if the binary is missing."""
    config_file = tmp_path / "config.toml"
    cfg = _make_test_config(tmp_path)

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr("arc_llama.cli.default_config_path", lambda: config_file)
    monkeypatch.setattr("arc_llama.cli.detect_gpus", lambda: cfg.gpus)
    monkeypatch.setattr(
        "arc_llama.cli.init_config_from_detection",
        lambda gpus, llama_server_path: cfg,
    )

    result = cli_runner.invoke(
        cli,
        [
            "--config", str(config_file), "init", "--no-scan",
            "--llama-server", "/nonexistent/llama-server",
        ],
    )
    assert result.exit_code == 3
    assert "llama-server binary not found" in result.output
    assert "/nonexistent/llama-server" in result.output


def test_init_writes_config_without_binary_and_hints_install_runtime(
    monkeypatch, tmp_path, cli_runner
):
    """A new user with no llama-server yet: init still writes a GPU config and
    points them at install-runtime, instead of hard-failing."""
    config_file = tmp_path / "config.toml"
    cfg = _make_test_config(tmp_path)

    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr("arc_llama.cli.default_config_path", lambda: config_file)
    monkeypatch.setattr("arc_llama.cli.detect_gpus", lambda: cfg.gpus)
    monkeypatch.setattr(
        "arc_llama.cli.init_config_from_detection",
        lambda gpus, llama_server_path: cfg,
    )
    monkeypatch.setattr(
        "arc_llama.cli._resolve_llama_server",
        lambda explicit: str(tmp_path / "no-such-llama-server"),
    )
    # detect_gpus is mocked to return GPUConfig objects; the real gpu-table
    # renderer expects detected-GPU objects (with vram_gb), so stub it out.
    monkeypatch.setattr("arc_llama.cli._print_gpu_table", lambda gpus: None)

    result = cli_runner.invoke(
        cli, ["--config", str(config_file), "init", "--no-scan"]
    )

    assert result.exit_code == 0, result.output
    assert "install-runtime" in result.output
    assert config_file.exists()


def test_windows_no_gpu_guidance_does_not_claim_detection_is_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr('arc_llama.cli._IS_WINDOWS', True)
    monkeypatch.setattr('arc_llama.cli.detect_gpus', lambda: [])
    result = CliRunner().invoke(cli, ['-c', str(tmp_path/'config.toml'), 'init', '--no-scan'])
    assert result.exit_code == 2
    assert 'Device Manager' in ' '.join(result.output.split())
    assert 'doctor' in result.output
    assert 'not supported' not in result.output
    assert not (tmp_path/'config.toml').exists()
