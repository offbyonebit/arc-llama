from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.benchmark import BenchmarkResult
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, TuneConfig
from arc_llama.recipe_share import SharedRecipe


def _config(tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(
            llama_server=str(tmp_path / "llama-server"),
            models_dir=str(tmp_path / "models"),
        ),
        tune=TuneConfig(auto=False, prompt_tokens=64, gen_tokens=16),
        gpus=[
            GPUConfig(
                pci_slot="gpu",
                sycl_index=0,
                arch="battlemage",
                backend="sycl",
                vram_mb=24 * 1024,
            )
        ],
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "qwen.gguf"),
                port=18080,
                gpu_pci_slot="gpu",
                recipe={
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                    "ubatch_size": 512,
                },
            )
        ],
    )


def _entry() -> SharedRecipe:
    return SharedRecipe(
        fingerprint="a" * 64,
        edits={"kv": "q8_0", "ubatch": 1024},
        submits=5,
        prompt_eval_tok_s=100.0,
        generation_tok_s=20.0,
        gpu_name="Arc",
        arc_llama_version="0.8.0",
        provenance={"llama_server_backend": "sycl"},
        confidence_score=0.75,
    )


def _result(prompt: float, generation: float) -> BenchmarkResult:
    return BenchmarkResult(
        model="qwen",
        ctx=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        prompt_tokens=64,
        gen_tokens=16,
        prompt_eval_tok_s=prompt,
        generation_tok_s=generation,
    )


@pytest.fixture
def recipe_cli(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr(
        "arc_llama.recipe_share.RecipeRegistry.lookup",
        lambda _self, _fingerprint: _entry(),
    )
    monkeypatch.setattr(
        "arc_llama.recipe_share.llama_server_build_identity",
        lambda _path: {"llama_server_backend": "sycl"},
    )
    return cfg


def test_recipes_apply_dry_run_is_non_mutating(recipe_cli, tmp_path):
    original = dict(recipe_cli.models[0].recipe)
    result = CliRunner().invoke(
        cli,
        ["--config", str(tmp_path / "config.toml"), "recipes", "apply", "qwen", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "confidence 75%" in result.output
    assert recipe_cli.models[0].recipe == original


def test_recipes_apply_rolls_back_a_regression(recipe_cli, tmp_path, monkeypatch):
    measurements = iter([_result(100, 20), _result(81, 20)])

    async def benchmark(*_args, **_kwargs):
        return next(measurements)

    applied: list[dict] = []
    restored: list[dict] = []

    async def apply(_client, _model, edits):
        applied.append(dict(edits))
        return None

    async def restore(_client, _model, edits, cfg=None):
        restored.append(dict(edits))
        return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    monkeypatch.setattr("arc_llama.cli.benchmark_mod.benchmark_model", benchmark)
    monkeypatch.setattr("arc_llama.tune._apply_edits", apply)
    monkeypatch.setattr("arc_llama.tune._restore_final_state", restore)
    monkeypatch.setattr("arc_llama.cli.httpx.AsyncClient", Client)

    result = CliRunner().invoke(
        cli,
        ["--config", str(tmp_path / "config.toml"), "recipes", "apply", "qwen"],
    )
    assert result.exit_code == 1
    assert "original recipe restored" in result.output
    assert applied == [
        {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "ubatch_size": 1024,
        }
    ]
    assert restored == [
        {
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "ubatch_size": 512,
        }
    ]


def test_recipes_apply_keeps_a_verified_win(recipe_cli, tmp_path, monkeypatch):
    measurements = iter([_result(100, 20), _result(121, 20)])

    async def benchmark(*_args, **_kwargs):
        return next(measurements)

    restored: list[dict] = []

    async def apply(_client, _model, _edits):
        return None

    async def restore(_client, _model, edits, cfg=None):
        restored.append(dict(edits))
        return None

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    monkeypatch.setattr("arc_llama.cli.benchmark_mod.benchmark_model", benchmark)
    monkeypatch.setattr("arc_llama.tune._apply_edits", apply)
    monkeypatch.setattr("arc_llama.tune._restore_final_state", restore)
    monkeypatch.setattr("arc_llama.cli.httpx.AsyncClient", Client)

    result = CliRunner().invoke(
        cli,
        ["--config", str(tmp_path / "config.toml"), "recipes", "apply", "qwen"],
    )
    assert result.exit_code == 0, result.output
    assert "improved 10.0%" in result.output
    assert restored == []


class _RegistryResponse:
    def __init__(self, payload: bytes, *, content_length: str | None = None):
        self.payload = payload
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield self.payload


def test_recipes_update_validates_and_atomically_installs(tmp_path, monkeypatch):
    import json

    destination = tmp_path / "recipes.json"
    doc = {
        "schema": 1,
        "recipes": {
            "a" * 64: {
                "recipe": {"kv": "q8_0"},
                "submits": 1,
            }
        },
    }
    payload = json.dumps(doc).encode()
    monkeypatch.setenv("ARC_LLAMA_RECIPES_PATH", str(destination))
    monkeypatch.setattr(
        "httpx.stream",
        lambda *_args, **_kwargs: _RegistryResponse(payload, content_length=str(len(payload))),
    )

    result = CliRunner().invoke(cli, ["recipes", "update", "--url", "https://example.test/r.json"])

    assert result.exit_code == 0, result.output
    assert json.loads(destination.read_text()) == doc


def test_recipes_update_rejects_invalid_registry_without_replacing_old_file(tmp_path, monkeypatch):
    destination = tmp_path / "recipes.json"
    destination.write_text("old registry")
    monkeypatch.setenv("ARC_LLAMA_RECIPES_PATH", str(destination))
    monkeypatch.setattr(
        "httpx.stream",
        lambda *_args, **_kwargs: _RegistryResponse(b'{"schema":1,"recipes":[]}'),
    )

    result = CliRunner().invoke(cli, ["recipes", "update", "--url", "https://example.test/r.json"])

    assert result.exit_code == 1
    assert "recipes must be a JSON object" in result.output
    assert destination.read_text() == "old registry"
