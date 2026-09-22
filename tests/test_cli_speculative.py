from __future__ import annotations

from click.testing import CliRunner

from arc_llama.benchmark import BenchmarkResult
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, TuneConfig
from arc_llama.server_caps import ServerCaps
from arc_llama.speculation import DraftCandidate


def _result(generation: float) -> BenchmarkResult:
    return BenchmarkResult(
        model="target",
        ctx=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        prompt_tokens=64,
        gen_tokens=16,
        prompt_eval_tok_s=100.0,
        generation_tok_s=generation,
    )


def test_speculative_auto_rolls_back_when_draft_is_slower(tmp_path, monkeypatch):
    cfg = Config(
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
                name="target",
                path=str(tmp_path / "target.gguf"),
                port=18080,
                gpu_pci_slot="gpu",
                recipe={"spec_type": "ngram-simple", "spec_draft_n_max": 2},
            ),
            ModelConfig(
                name="draft",
                path=str(tmp_path / "draft.gguf"),
                port=18081,
                gpu_pci_slot="gpu",
            ),
        ],
    )
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr(
        "arc_llama.server_caps.probe_server_caps",
        lambda _path: ServerCaps(
            probed=True,
            supports_speculative=True,
            supports_draft_model=True,
        ),
    )
    monkeypatch.setattr(
        "arc_llama.speculation.discover_drafts",
        lambda _cfg, _target: [
            DraftCandidate(
                name="draft",
                path=str(tmp_path / "draft.gguf"),
                family="qwen",
                estimated_mb=4096,
                fits=True,
                tokenizer_compatible=True,
                reason="verified",
            )
        ],
    )
    measurements = iter([_result(20.0), _result(18.0)])

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
        ["--config", str(tmp_path / "config.toml"), "speculative", "target", "--auto"],
    )
    assert result.exit_code == 1
    assert "original speculation recipe restored" in result.output
    assert applied[0]["spec_type"] is None
    assert applied[1] == {
        "spec_type": "draft-simple",
        "spec_draft_name": "draft",
        "spec_draft_n_max": 4,
    }
    assert restored == [
        {
            "spec_type": "ngram-simple",
            "spec_draft_name": None,
            "spec_draft_n_max": 2,
            "speculation_result": None,
        }
    ]
