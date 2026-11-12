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
        models=[],
    )


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_benchmark_rejects_unknown_model(monkeypatch, tmp_path, cli_runner):
    """`arc-llama benchmark` must exit cleanly when the model isn't registered."""
    config_file = tmp_path / "config.toml"
    cfg = _make_test_config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    result = cli_runner.invoke(
        cli,
        ["--config", str(config_file), "benchmark", "not-a-model"],
    )
    assert result.exit_code == 1
    assert "not registered" in result.output


def _bench_config(tmp_path: Path) -> Config:
    from arc_llama.config import ModelConfig

    cfg = _make_test_config(tmp_path)
    cfg.gpus[0].backend = "sycl"
    cfg.models.append(
        ModelConfig(
            name="qwen-27b",
            path=str(tmp_path / "secret-user" / "qwen-27b-Q4_K_M.gguf"),
            port=8091,
            gpu_pci_slot="0000:03:00.0",
            recipe={"ctx": 8192, "cache_type_k": "q4_0", "cache_type_v": "q4_0",
                    "spec_type": "draft-mtp,ngram-mod"},
        )
    )
    return cfg


class _Health:
    def __init__(self, loaded):
        self._loaded = loaded

    def json(self):
        return {"status": "ok", "loaded_models": self._loaded}


def test_bench_runs_on_the_loaded_model_and_prints_plain_text(monkeypatch, tmp_path, cli_runner):
    from arc_llama import benchmark

    cfg = _bench_config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Health(["qwen-27b"]))
    monkeypatch.setattr(benchmark, "llama_cpp_build", lambda binary: "b12243 (abc1234)")
    seen = {}

    async def fake_depths(url, model, depths, *, gen_tokens):
        seen.update(model=model, depths=depths)
        return [
            benchmark.DepthResult(depth=d, measured_depth=d, prompt_eval_tok_s=900.0,
                                  generation_tok_s=29.2)
            for d in depths
        ]

    monkeypatch.setattr(benchmark, "benchmark_depths", fake_depths)
    result = cli_runner.invoke(cli, ["--config", str(tmp_path / "c.toml"), "bench"])

    assert result.exit_code == 0, result.output
    assert seen == {"model": "qwen-27b", "depths": [512]}  # 16384 > ctx 8192
    out = result.output
    assert "Backend: sycl on Arc Pro B60 (battlemage, 24576 MiB)" in out
    assert "llama.cpp: b12243 (abc1234)" in out
    assert "Settings: ctx 8192, KV q4_0/q4_0, speculation draft-mtp,ngram-mod" in out
    assert "Prompt 512: prefill 900.0 tok/s, decode 29.2 tok/s" in out
    assert "Prompt 16384: skipped" in out
    assert "n-gram speculation reads high" in out
    assert "secret-user" not in out and str(tmp_path) not in out


def test_bench_stops_cleanly_without_a_server(monkeypatch, tmp_path, cli_runner):
    import httpx

    cfg = _bench_config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", refuse)
    result = cli_runner.invoke(cli, ["--config", str(tmp_path / "c.toml"), "bench"])
    assert result.exit_code == 1
    assert "No arc-llama server answered" in result.output


def test_bench_needs_a_model_when_none_is_loaded(monkeypatch, tmp_path, cli_runner):
    cfg = _bench_config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda path: cfg)
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Health([]))
    result = cli_runner.invoke(cli, ["--config", str(tmp_path / "c.toml"), "bench"])
    assert result.exit_code == 1
    assert "No model is loaded" in result.output


def test_llama_cpp_build_reads_version_output(monkeypatch):
    import subprocess

    from arc_llama.benchmark import llama_cpp_build

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="version: 12243 (8f1d2e3)\nbuilt with gcc")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_cpp_build("llama-server") == "b12243 (8f1d2e3)"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert llama_cpp_build("llama-server") is None
