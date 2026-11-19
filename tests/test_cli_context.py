from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.recipes import KVCacheType, estimate_kv_bytes, suggest_ctx


@pytest.fixture
def context_config(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    path = tmp_path / "config.toml"
    cfg = Config(
        gpus=[GPUConfig("gpu", 0, "battlemage", vram_mb=8192)],
        models=[
            ModelConfig(
                name="demo",
                path=str(model_path),
                port=18080,
                gpu_pci_slot="gpu",
                aliases=["demo-alias"],
                recipe={"ctx": 8192, "cache_type_k": "q8_0", "parallel": 2},
            )
        ],
    )
    cfg.save(path)
    monkeypatch.setattr(
        "arc_llama.gguf_meta.read_gguf_meta",
        lambda _path: {
            "architecture": "llama",
            "block_count": 16,
            "attention.head_count_kv": 8,
            "attention.key_length": 128,
            "attention.value_length": 128,
            "context_length": 32768,
        },
    )
    return path, cfg


def invoke_context(context_config, monkeypatch, *args):
    path, _cfg = context_config
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("context must not write config, detect GPUs, or start a server")

    with monkeypatch.context() as guard:
        guard.setattr(Config, "save", forbidden)
        guard.setattr("arc_llama.cli._save_or_die", forbidden)
        guard.setattr("arc_llama.cli.detect_gpus", forbidden)
        guard.setattr("arc_llama.launcher.LlamaServer.start", forbidden)
        result = CliRunner().invoke(cli, ["-c", str(path), "context", *args])
    assert path.read_bytes() == before
    return result


def test_context_metadata_text(context_config, monkeypatch):
    result = invoke_context(context_config, monkeypatch, "demo-alias")

    assert result.exit_code == 0, result.output
    assert "Model: demo" in result.output
    assert "KV source: gguf_metadata" in result.output
    assert "KV per token (f16): 64 KiB" in result.output
    assert "KV per token (q8_0): 34 KiB" in result.output
    assert "Configured context: 8192; KV type: q8_0" in result.output
    assert "KV at configured context: 272 MiB" in result.output
    assert "VRAM estimate: 1297 MiB" in result.output
    assert "Card VRAM: 8192 MiB" in result.output
    assert "Largest context that fits (q8_0, estimate): 32768" in result.output
    assert "Largest context that fits (q4_0, estimate): 32768" in result.output


def test_context_json_fields_and_fit_inputs(context_config, monkeypatch):
    fit = Mock(wraps=suggest_ctx)
    monkeypatch.setattr("arc_llama.cli_models.suggest_ctx", fit)
    result = invoke_context(context_config, monkeypatch, "demo", "--json")

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert set(report) == {
        "model",
        "kv_source",
        "kv_class",
        "kv_kib_per_token",
        "configured_ctx",
        "configured_kv_type",
        "configured_kv_mib",
        "parallel",
        "trained_ctx",
        "estimated_vram_mb",
        "gpu_vram_mb",
        "estimated_max_ctx",
    }
    assert report["kv_kib_per_token"] == {
        kv.value: estimate_kv_bytes(1, kv, f16_bytes_per_token=65536) / 1024
        for kv in (KVCacheType.F16, KVCacheType.Q8_0, KVCacheType.Q4_0)
    }
    assert report["estimated_vram_mb"] == 1297
    assert report["estimated_max_ctx"] == {"q8_0": 32768, "q4_0": 32768}
    assert fit.call_count == 2
    for call in fit.call_args_list:
        assert call.kwargs["vram_mb"] == 8192
        assert call.kwargs["model_file_mb"] == 1
        assert call.kwargs["parallel"] == 2
        assert call.kwargs["ctx_cap"] == call.kwargs["trained_ctx"] == 32768
        assert call.kwargs["f16_bytes_per_token"] == 65536


def test_context_family_fallback_and_router_defaults(context_config, monkeypatch):
    path, cfg = context_config
    cfg.models[0].recipe = {}
    cfg.models[0].kv_class = "moe_a3b"
    cfg.save(path)
    monkeypatch.setattr("arc_llama.gguf_meta.read_gguf_meta", lambda _path: {})
    fit = Mock(wraps=suggest_ctx)
    monkeypatch.setattr("arc_llama.cli_models.suggest_ctx", fit)
    result = invoke_context(context_config, monkeypatch, "demo", "--json")

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["kv_source"] == "family_estimate"
    assert report["kv_kib_per_token"]["f16"] == 20
    assert report["configured_ctx"] == 8192
    assert report["configured_kv_type"] == "f16"
    assert report["configured_kv_mib"] == 160
    assert report["parallel"] == 1
    assert report["trained_ctx"] is None
    assert fit.call_args.kwargs["ctx_cap"] == 262144
    assert fit.call_args.kwargs["trained_ctx"] is None


@pytest.mark.parametrize("missing_gpu", [True, False])
def test_context_unknown_vram_skips_fit(context_config, monkeypatch, missing_gpu):
    path, cfg = context_config
    if missing_gpu:
        cfg.gpus = []
    else:
        cfg.gpus[0].vram_mb = None
    cfg.save(path)
    fit = Mock(side_effect=AssertionError("cannot suggest context without VRAM"))
    monkeypatch.setattr("arc_llama.cli_models.suggest_ctx", fit)

    text = invoke_context(context_config, monkeypatch, "demo")
    result = invoke_context(context_config, monkeypatch, "demo", "--json")

    assert text.exit_code == result.exit_code == 0
    assert "VRAM estimate: 1297 MiB" in text.output
    assert "Card VRAM: unknown" in text.output
    assert "Largest context" not in text.output
    report = json.loads(result.output)
    assert report["gpu_vram_mb"] is None
    assert report["estimated_max_ctx"] is None
    fit.assert_not_called()


def test_context_unknown_model_fails(context_config, monkeypatch):
    result = invoke_context(context_config, monkeypatch, "no-such-model")

    assert result.exit_code != 0
    assert "Unknown model: no-such-model" in result.output


def test_context_missing_config_is_not_created(tmp_path):
    path = tmp_path / "missing.toml"

    result = CliRunner().invoke(cli, ["-c", str(path), "context", "demo"])

    assert result.exit_code != 0
    assert "Unknown model: demo" in result.output
    assert not path.exists()


def test_context_unavailable_vram_estimate_is_unknown(context_config, monkeypatch):
    monkeypatch.setattr("arc_llama.cli_models.estimate_model_vram_quick_mb", lambda _model: None)
    text = invoke_context(context_config, monkeypatch, "demo")
    result = invoke_context(context_config, monkeypatch, "demo", "--json")

    assert text.exit_code == result.exit_code == 0
    assert "VRAM estimate: unknown" in text.output
    assert json.loads(result.output)["estimated_vram_mb"] is None


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("split_model", [False, True])
def test_context_missing_model_file_stops_before_estimates(
    context_config, monkeypatch, as_json, split_model
):
    config_path, cfg = context_config
    model_path = Path(cfg.models[0].path)
    if split_model:
        first = model_path.with_name("model-00001-of-00002.gguf")
        model_path.rename(first)
        cfg.models[0].path = str(first)
        missing = model_path.with_name("model-00002-of-00002.gguf")
        cfg.save(config_path)
    else:
        model_path.unlink()
        missing = model_path

    def forbidden(*args, **kwargs):
        pytest.fail("missing model files must stop the report before metadata or sizing")

    for helper in (
        "kv_bytes_per_token_f16",
        "trained_context_length",
        "gguf_total_bytes",
        "estimate_model_vram_quick_mb",
        "suggest_ctx",
    ):
        monkeypatch.setattr(f"arc_llama.cli_models.{helper}", forbidden)

    args = ["demo", "--json"] if as_json else ["demo"]
    result = invoke_context(context_config, monkeypatch, *args)

    assert result.exit_code != 0
    assert f"Model file missing: {missing}." in result.output
    assert "Re-download it or remove the model with arc-llama remove." in result.output
    assert "Largest context" not in result.output
    assert "estimated_max_ctx" not in result.output
