"""Speculation candidate selection, depth benchmarks, driver retune, tensor split."""

from __future__ import annotations

import httpx
import pytest
from click.testing import CliRunner
from helpers import AsyncContextClient

import arc_llama.autotune as autotune
from arc_llama.arch import Arch, profile_for
from arc_llama.benchmark import BenchmarkResult, benchmark_depths, usable_depths
from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, TuneConfig
from arc_llama.launcher import build_env, build_plan, resolve_split_gpus
from arc_llama.server_caps import ServerCaps
from arc_llama.speculation import DraftCandidate

# -- speculation: measure every candidate, keep the fastest -------------------


def _bench(generation: float) -> BenchmarkResult:
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


def _spec_cfg(tmp_path) -> Config:
    return Config(
        paths=PathsConfig(llama_server=str(tmp_path / "llama-server")),
        tune=TuneConfig(auto=False, prompt_tokens=64, gen_tokens=16),
        gpus=[GPUConfig("gpu", 0, "battlemage", backend="sycl", vram_mb=24576)],
        models=[ModelConfig("target", str(tmp_path / "t.gguf"), 18080, "gpu")],
    )


def _candidate(name: str) -> DraftCandidate:
    return DraftCandidate(
        name=name,
        path=f"/{name}.gguf",
        family="qwen",
        estimated_mb=2048,
        fits=True,
        tokenizer_compatible=True,
        reason="verified",
    )


@pytest.mark.parametrize(
    ("speeds", "expected"),
    [
        ([20.0, 22.0, 26.0, 21.0], "draft-b"),
        ([20.0, 22.0, 21.0, 30.0], None),  # n-gram wins
        ([20.0, 22.0, 26.0, 19.0], "draft-b"),  # winner is not the last tried
    ],
)
def test_auto_verify_keeps_the_fastest_candidate(tmp_path, monkeypatch, speeds, expected):
    cfg = _spec_cfg(tmp_path)
    saved: list[Config] = []
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    monkeypatch.setattr("arc_llama.cli._save_or_die", lambda c, _p: saved.append(c))
    monkeypatch.setattr(
        "arc_llama.server_caps.probe_server_caps",
        lambda _p: ServerCaps(
            probed=True, supports_speculative=True, supports_draft_model=True, supports_ngram=True
        ),
    )
    monkeypatch.setattr(
        "arc_llama.speculation.discover_drafts",
        lambda _c, _t: [_candidate("draft-a"), _candidate("draft-b")],
    )
    measurements = iter(_bench(s) for s in speeds)

    async def benchmark(*_a, **_k):
        return next(measurements)

    applied: list[dict] = []

    async def apply(_client, _model, edits):
        applied.append(dict(edits))

    async def restore(*_a, **_k):
        raise AssertionError("a winner was found; nothing to restore")

    monkeypatch.setattr("arc_llama.cli.benchmark_mod.benchmark_model", benchmark)
    monkeypatch.setattr("arc_llama.tune._apply_edits", apply)
    monkeypatch.setattr("arc_llama.tune._restore_final_state", restore)
    monkeypatch.setattr("arc_llama.cli.httpx.AsyncClient", AsyncContextClient)

    result = CliRunner().invoke(
        cli, ["--config", str(tmp_path / "c.toml"), "speculative", "target", "--auto"]
    )
    assert result.exit_code == 0, result.output
    recipe = cfg.models[0].recipe
    if expected is None:
        assert recipe["spec_type"] == "ngram-simple"
        assert "spec_draft_name" not in recipe
    else:
        assert recipe["spec_draft_name"] == expected
    # The server ends on the winner, whatever order they were tried in.
    assert applied[-1]["spec_type"] == recipe["spec_type"]
    assert applied[-1]["spec_draft_name"] == recipe.get("spec_draft_name")
    assert "verified at" in recipe["speculation_result"]
    assert saved


# -- depth benchmark ----------------------------------------------------------


def test_usable_depths_respects_context() -> None:
    assert usable_depths([0, 4096, 16384, 32768], 16384, 128) == ([0, 4096], [16384, 32768])
    assert usable_depths([4096, 0, 4096], None, 128) == ([0, 4096], [])


async def test_benchmark_depths_reports_engine_timings() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        seen.append(body)
        depth = len(body.get("prompt", "")) // 4
        return httpx.Response(
            200,
            json={
                "timings": {
                    "prompt_n": depth,
                    "prompt_per_second": 500.0,
                    "predicted_n": body["max_tokens"],
                    "predicted_per_second": 30.0 - depth / 1000,
                },
                "usage": {"completion_tokens": body["max_tokens"]},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://t"
    ) as client:
        rows = await benchmark_depths("http://t", "m", [0, 8000], gen_tokens=32, client=client)
    assert [r.depth for r in rows] == [0, 8000]
    assert rows[1].generation_tok_s == pytest.approx(22.0)
    assert rows[1].measured_depth == 8000
    measured = [b for b in seen if b.get("ignore_eos")]
    assert all(b["cache_prompt"] is False and b["max_tokens"] == 32 for b in measured)


async def test_benchmark_depths_keeps_going_after_an_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"timings": {"predicted_per_second": 10.0}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://t"
    ) as client:
        rows = await benchmark_depths("http://t", "m", [0, 4096], client=client)
    assert rows[0].error is not None
    assert rows[1].generation_tok_s == 10.0


def test_benchmark_cli_rejects_bad_depths(tmp_path, monkeypatch) -> None:
    cfg = _spec_cfg(tmp_path)
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _path: cfg)
    result = CliRunner().invoke(cli, ["benchmark", "target", "--depths", "a,b"])
    assert result.exit_code != 0
    result = CliRunner().invoke(cli, ["benchmark", "target", "--depths", "-5"])
    assert result.exit_code != 0


# -- driver stack in the tune fingerprint -------------------------------------


def test_driver_stack_change_invalidates_tuned_recipe(tmp_path) -> None:
    model = ModelConfig("m", str(tmp_path / "m.gguf"), 1, "gpu")
    gpu = GPUConfig("gpu", 0, "battlemage")
    before = autotune.compute_fingerprint(model, "llama-server", gpu, "1.0", stack_key="mesa 25.1")
    after = autotune.compute_fingerprint(model, "llama-server", gpu, "1.0", stack_key="mesa 25.2")
    assert before != after
    assert before == autotune.compute_fingerprint(
        model, "llama-server", gpu, "1.0", stack_key="mesa 25.1"
    )


def test_driver_stack_key_is_cached_and_describes_the_host() -> None:
    autotune.driver_stack_key.cache_clear()
    key = autotune.driver_stack_key()
    assert key.startswith("os=")
    assert autotune.driver_stack_key() is key


# -- tensor split ---------------------------------------------------------------


def _split_cfg(backend: str = "sycl", **recipe) -> Config:
    cfg = Config(
        gpus=[
            GPUConfig("a", 0, "battlemage", backend=backend, vram_mb=24576, vulkan_index=1),
            GPUConfig("b", 1, "battlemage", backend=backend, vram_mb=16384, vulkan_index=2),
        ],
        models=[ModelConfig("m", "/m.gguf", 18080, "a", recipe=dict(recipe))],
    )
    cfg.paths.llama_server = "/nonexistent/llama-server"
    return cfg


def test_split_gpus_resolution() -> None:
    cfg = _split_cfg(tensor_split=[3, 2], split_gpus=["b", "a"])
    assert [g.pci_slot for g in resolve_split_gpus(cfg, cfg.models[0]) or []] == ["a", "b"]
    for bad in (
        {"tensor_split": [1, 1], "split_gpus": ["b"]},
        {"tensor_split": [1, 1, 1], "split_gpus": ["a", "b"]},
        {"tensor_split": [1, 1], "split_gpus": ["a", "zz"]},
    ):
        cfg = _split_cfg(**bad)
        assert resolve_split_gpus(cfg, cfg.models[0]) is None
    cfg = _split_cfg(tensor_split=[1, 1], split_gpus=["a", "b"])
    cfg.gpus[1].backend = "vulkan"
    assert resolve_split_gpus(cfg, cfg.models[0]) is None


def test_split_exposes_every_device() -> None:
    profile = profile_for(Arch.BATTLEMAGE)
    cfg = _split_cfg()
    env = build_env(profile, cfg.gpus[0], extra_gpus=[cfg.gpus[1]])
    assert env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0,1"
    vk = _split_cfg(backend="vulkan")
    env = build_env(profile, vk.gpus[0], extra_gpus=[vk.gpus[1]])
    assert env["GGML_VK_VISIBLE_DEVICES"] == "1,2"


def test_plan_drops_tensor_split_without_valid_gpus() -> None:
    good = _split_cfg(tensor_split=[1, 1], split_gpus=["a", "b"])
    plan = build_plan(good, good.models[0], good.gpus[0])
    assert plan.argv[plan.argv.index("--tensor-split") + 1] == "1,1"
    assert plan.env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0,1"
    bad = _split_cfg(tensor_split=[1, 1])
    plan = build_plan(bad, bad.models[0], bad.gpus[0])
    assert "--tensor-split" not in plan.argv
    assert plan.env["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"


def test_split_fit_counts_both_gpus(monkeypatch) -> None:
    import arc_llama.router as router_mod
    from arc_llama.failures import StartupFailureError

    monkeypatch.setattr(router_mod, "_estimate_model_vram_mb", lambda _m: 30 * 1024)
    cfg = _split_cfg(ctx=4096)
    router = router_mod.Router(cfg)
    with pytest.raises(StartupFailureError):
        router._check_vram_fit(cfg.models[0], cfg.gpus[0])
    cfg.models[0].recipe.update({"tensor_split": [3, 2], "split_gpus": ["a", "b"]})
    router._check_vram_fit(cfg.models[0], cfg.gpus[0])
