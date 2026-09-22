from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from arc_llama.arch import Arch, Backend
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig
from arc_llama.detect import DetectedGPU


def _config(tmp_path: Path, *, models: int = 1) -> Config:
    runtime = tmp_path / "llama-server"
    runtime.write_bytes(b"runtime")
    model_entries: list[ModelConfig] = []
    for index in range(models):
        gguf = tmp_path / f"model-{index}.gguf"
        gguf.write_bytes(b"GGUF")
        model_entries.append(
            ModelConfig(
                name=f"model-{index}",
                display_name=f"Model {index}",
                path=str(gguf),
                port=18080 + index,
                gpu_pci_slot="0000:03:00.0",
                recipe={
                    "ctx": 32768,
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                },
            )
        )
    return Config(
        paths=PathsConfig(
            llama_server=str(runtime),
            models_dir=str(tmp_path / "models"),
        ),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24576,
                enabled=True,
                name="Intel Arc Pro B60",
                backend="vulkan",
            )
        ],
        models=model_entries,
    )


def test_run_existing_model_prints_launch_contract(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    backend_probe = MagicMock(return_value={Backend.VULKAN})
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli.detect_backends", backend_probe)
    monkeypatch.setattr("arc_llama.router.estimate_model_vram_quick_mb", lambda _model: 12288)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(tmp_path / "config.toml"),
            "run",
            "model-0",
            "--setup-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Arc inference is ready" in result.output
    assert "comfortable" in result.output
    assert "http://127.0.0.1:11437/v1" in result.output
    assert "Setup-only complete" in result.output
    backend_probe.assert_called_once()


def test_run_new_machine_bootstraps_runtime_and_local_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    runtime = tmp_path / "runtime" / "llama-server"
    runtime.parent.mkdir()
    runtime.write_bytes(b"runtime")
    gguf = tmp_path / "Qwen3-8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    detected = DetectedGPU(
        pci_slot="0000:03:00.0",
        device_id=0xE211,
        arch=Arch.BATTLEMAGE,
        name="Intel Arc Pro B60",
        driver="xe",
        vram_mb=24576,
        drm_card="card1",
        drm_render="renderD129",
        sysfs_path="/fake",
    )
    calls: list[str] = []

    monkeypatch.setattr("arc_llama.cli.detect_gpus", lambda: [detected])
    monkeypatch.setattr("arc_llama.cli._configured_runtime", lambda cfg: None)

    def fake_install_runtime(**kwargs):
        calls.append(kwargs["backend"])
        cfg = kwargs["cfg"]
        cfg.paths.llama_server = str(runtime)
        for gpu in cfg.gpus:
            gpu.backend = kwargs["backend"]
        cfg.save(kwargs["config_path"])
        return SimpleNamespace(tag="b12345", binary_path=runtime)

    def fake_add_local_model(cfg, *, name, path, gpu_pci_slot, **_kwargs):
        model = ModelConfig(
            name=name,
            display_name="Qwen 3 8B",
            path=path,
            port=18080,
            gpu_pci_slot=gpu_pci_slot,
            recipe={"ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )
        cfg.models.append(model)
        return model

    monkeypatch.setattr("arc_llama.runtime.install_runtime", fake_install_runtime)
    monkeypatch.setattr("arc_llama.cli.add_local_model", fake_add_local_model)
    monkeypatch.setattr("arc_llama.router.estimate_model_vram_quick_mb", lambda _model: 10000)

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_path), "run", str(gguf), "--setup-only"],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["vulkan"]
    assert "Detecting Intel Arc hardware" in result.output
    assert "Installing verified vulkan" in result.output
    assert "registered" in result.output
    assert config_path.exists()


def test_run_without_source_requires_choice_when_multiple_models(tmp_path, monkeypatch):
    cfg = _config(tmp_path, models=2)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli.detect_backends", lambda _path: {Backend.VULKAN})
    monkeypatch.setattr("arc_llama.cli._do_scan", lambda _cfg, _paths: [])

    result = CliRunner().invoke(
        cli,
        ["--config", str(tmp_path / "config.toml"), "run", "--setup-only"],
    )

    assert result.exit_code == 1
    assert "More than one model is registered" in result.output
    assert "model-0, model-1" in result.output


def test_run_refuses_recipe_estimated_over_vram(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli.detect_backends", lambda _path: {Backend.VULKAN})
    monkeypatch.setattr("arc_llama.router.estimate_model_vram_quick_mb", lambda _model: 26000)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(tmp_path / "config.toml"),
            "run",
            "model-0",
            "--setup-only",
        ],
    )

    assert result.exit_code == 1
    assert "too large" in result.output
    assert "exceed GPU VRAM" in result.output


def test_run_keeps_recognised_sycl_runtime_by_default(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.gpus[0].backend = "sycl"
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli.detect_backends", lambda _path: {Backend.SYCL})
    monkeypatch.setattr("arc_llama.router.estimate_model_vram_quick_mb", lambda _model: 12000)

    with patch("arc_llama.runtime.install_runtime") as install:
        result = CliRunner().invoke(
            cli,
            [
                "--config",
                str(tmp_path / "config.toml"),
                "run",
                "model-0",
                "--setup-only",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "sycl" in result.output
    install.assert_not_called()


def test_run_starts_existing_serve_path(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    app = MagicMock()
    app.state.router = None
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli.detect_backends", lambda _path: {Backend.VULKAN})
    monkeypatch.setattr("arc_llama.cli._print_serve_banner", lambda _cfg: None)
    monkeypatch.setattr("arc_llama.cli._print_autotune_banner", lambda _cfg: None)
    monkeypatch.setattr("arc_llama.router.estimate_model_vram_quick_mb", lambda _model: 12000)

    with (
        patch("arc_llama.server.create_app", return_value=app),
        patch("uvicorn.run") as uvicorn_run,
        patch("signal.signal"),
        patch("atexit.register"),
    ):
        result = CliRunner().invoke(
            cli,
            ["--config", str(tmp_path / "config.toml"), "run", "model-0"],
        )

    assert result.exit_code == 0, result.output
    assert "Press Ctrl+C to stop" in result.output
    uvicorn_run.assert_called_once_with(
        app,
        host="127.0.0.1",
        port=11437,
        log_level="info",
    )
