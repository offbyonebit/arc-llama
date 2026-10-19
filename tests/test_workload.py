"""Tests for the workload profile (AUTOTUNE_ROUND3 phase 2).

All fakes: no GPU, no llama-server, no systemd. The sweep tests drive the
real tune_model with a faked edit/benchmark backend, so they exercise the
actual pruning and measurement-depth code paths rather than helpers alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from arc_llama import workload
from arc_llama.autotune import compute_fingerprint, reset_tuned_state_if_stale
from arc_llama.benchmark import BenchmarkResult
from arc_llama.cli import cli
from arc_llama.config import (
    Config,
    GPUConfig,
    ModelConfig,
    PathsConfig,
    TuneConfig,
    load_config,
)
from arc_llama.tune import score_result, tune_model


def _make_cfg(tmp_path: Path, *, vram_mb: int = 24 * 1024, ctx: int = 32768) -> Config:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00" * 64)
    return Config(
        paths=PathsConfig(llama_server=str(tmp_path / "no-such-llama-server")),
        tune=TuneConfig(auto=False),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=vram_mb,
            ),
        ],
        models=[
            ModelConfig(
                name="m",
                path=str(gguf),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": ctx, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            ),
        ],
    )


class PairRecorder:
    """Records (edit body, prompt_tokens) pairs from the real sweep loop."""

    def __init__(self, cfg: Config) -> None:
        self.state = dict(cfg.models[0].recipe)
        self.pairs: list[tuple[dict[str, Any], int]] = []
        self.bodies: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None

    async def apply(self, _client, _name, edits: dict[str, Any]) -> None:
        self.bodies.append(dict(edits))
        self._pending = dict(edits)
        for k, v in edits.items():
            if v is None:
                self.state.pop(k, None)
            else:
                self.state[k] = v

    async def bench(self, _server_url, _model_name, **kw) -> BenchmarkResult:
        self.pairs.append((self._pending or {}, kw.get("prompt_tokens", 0)))
        kv = str(self.state.get("cache_type_k", "f16"))
        pp = {"q8_0": 800.0, "f16": 1000.0}.get(kv, 100.0)
        return BenchmarkResult(
            model="m",
            ctx=8192,
            cache_type_k=kv,
            cache_type_v=kv,
            prompt_tokens=kw.get("prompt_tokens", 0),
            gen_tokens=128,
            prompt_eval_tok_s=pp,
            generation_tok_s=30.0,
        )


# ---------------------------------------------------------------------------
# Profile round-trips through config.
# ---------------------------------------------------------------------------


def test_workload_round_trip_through_config(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.workload.context_length = "long"
    cfg.workload.style = "agentic"
    cfg.workload.priority = "first_token"

    path = tmp_path / "config.toml"
    cfg.save(path)
    loaded = load_config(path)

    assert loaded.workload.context_length == "long"
    assert loaded.workload.style == "agentic"
    assert loaded.workload.priority == "first_token"


def test_workload_defaults_round_trip(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    path = tmp_path / "config.toml"
    cfg.save(path)
    loaded = load_config(path)
    assert loaded.workload.context_length == ""
    assert loaded.workload.style == ""
    assert loaded.workload.priority == ""


# ---------------------------------------------------------------------------
# Style maps to the sweep target.
# ---------------------------------------------------------------------------


def test_agentic_maps_to_prompt_target(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.workload.style = "agentic"
    assert workload.tune_target(cfg) == "prompt"


def test_conversational_maps_to_generation_target(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.workload.style = "conversational"
    assert workload.tune_target(cfg) == "generation"


def test_unanswered_style_keeps_tune_target(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert workload.tune_target(cfg) == "balanced"
    cfg.tune.target = "generation"
    assert workload.tune_target(cfg) == "generation"


# ---------------------------------------------------------------------------
# Priority weights the balanced score.
# ---------------------------------------------------------------------------


def test_priority_weights_balanced_score() -> None:
    r = BenchmarkResult(
        model="m",
        ctx=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        prompt_tokens=1024,
        gen_tokens=128,
        prompt_eval_tok_s=1000.0,
        generation_tok_s=40.0,
    )
    balanced = score_result(r, "balanced")
    first_token = score_result(r, "balanced", "first_token")
    throughput = score_result(r, "balanced", "throughput")
    assert balanced == pytest.approx((1000.0 * 40.0) ** 0.5)
    assert first_token == pytest.approx(1000.0**0.75 * 40.0**0.25)
    assert throughput == pytest.approx(1000.0**0.25 * 40.0**0.75)
    # Prompt-heavy result ranks higher under a first-token priority.
    assert first_token > balanced > throughput


# ---------------------------------------------------------------------------
# Fingerprint includes the profile.
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_profile_changes(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    gpu = cfg.find_gpu("0000:03:00.0")
    model = cfg.models[0]
    fp1 = compute_fingerprint(
        model,
        cfg.paths.llama_server,
        gpu,
        "0.6.0",
        workload.fingerprint_key(cfg.workload),
    )
    cfg.workload.context_length = "very_long"
    fp2 = compute_fingerprint(
        model,
        cfg.paths.llama_server,
        gpu,
        "0.6.0",
        workload.fingerprint_key(cfg.workload),
    )
    assert fp1 != fp2


def test_tuned_model_becomes_untuned_when_profile_changes(tmp_path: Path) -> None:
    """The real invalidation path: a tuned recipe dies when answers change."""
    cfg = _make_cfg(tmp_path)
    gpu = cfg.find_gpu("0000:03:00.0")
    model = cfg.models[0]
    model.tune_state = "tuned"
    model.tune_fingerprint = compute_fingerprint(
        model,
        cfg.paths.llama_server,
        gpu,
        "0.6.0",
        workload.fingerprint_key(cfg.workload),
    )

    # Same profile: untouched.
    reset_tuned_state_if_stale(cfg, model, "0.6.0")
    assert model.tune_state == "tuned"

    # Changed profile: fingerprint mismatch -> untuned again.
    cfg.workload.style = "conversational"
    reset_tuned_state_if_stale(cfg, model, "0.6.0")
    assert model.tune_state == "untuned"
    assert model.tune_fingerprint == ""


# ---------------------------------------------------------------------------
# Long-context profile prunes KV types that cannot fit.
# ---------------------------------------------------------------------------


async def test_long_context_prunes_f16_when_it_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """f16 KV at 131k is ~9 GB on kv_class=default; on a 6 GB card it must
    never be offered, while q8_0 (~4.5 GB) still is. This is the exact wrong
    answer the round-3 hardware run produced at 1k depth."""
    cfg = _make_cfg(tmp_path, vram_mb=6 * 1024, ctx=131072)
    cfg.workload.context_length = "very_long"

    recorder = PairRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", recorder.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    # No edit body may ever set f16: the candidate was pruned, not just beaten.
    assert all(b.get("cache_type_k") != "f16" for b in recorder.bodies)
    # q8_0 still swept (baseline and the q8_0 candidate state).
    assert any(b.get("cache_type_k") == "q8_0" for b in recorder.bodies)
    assert recorder.state.get("cache_type_k") == "q8_0"


async def test_short_context_does_not_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path, vram_mb=6 * 1024, ctx=8192)
    cfg.workload.context_length = "short"

    recorder = PairRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", recorder.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert any(b.get("cache_type_k") == "f16" for b in recorder.bodies)


# ---------------------------------------------------------------------------
# Deep measurement at the declared depth.
# ---------------------------------------------------------------------------


async def test_deep_measurement_uses_declared_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a long-context profile, KV and ubatch stages measure at 32k;
    flash-attn stays shallow."""
    # ctx is set above the 32k target so the deep prompt still fits after
    # reserving headroom below n_ctx.
    cfg = _make_cfg(tmp_path, ctx=65536)
    cfg.workload.context_length = "long"

    recorder = PairRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", recorder.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert recorder.pairs
    pts = [pt for _, pt in recorder.pairs]
    # Baseline is measured shallow.
    assert pts[0] == 1024
    # Deep measurement happened...
    assert 32768 in pts
    # ...for the whole KV+ubatch portion of the sweep (stage order is kv,
    # ubatch, fa), then flash-attn — anchor included — stays shallow.
    fa_start = pts.index(1024, pts.index(32768))
    assert set(pts[1:fa_start]) == {32768}
    assert set(pts[fa_start:]) == {1024}


async def test_no_profile_measures_shallow_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path, ctx=32768)

    recorder = PairRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", recorder.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert {pt for _, pt in recorder.pairs} == {1024}


# ---------------------------------------------------------------------------
# Deep prompt token capping.
# ---------------------------------------------------------------------------


def test_deep_prompt_tokens_capped_below_ctx(tmp_path: Path) -> None:
    """A known ctx limits the deep prompt, leaving headroom for the
    benchmark's max_tokens=1 plus server bookkeeping."""
    cfg = _make_cfg(tmp_path)
    cfg.workload.context_length = "very_long"
    recipe = {"ctx": 131072}

    depth = workload.deep_prompt_tokens(cfg, recipe)

    assert depth is not None
    assert depth < 131072
    assert depth == 131072 - workload.DEEP_PROMPT_CTX_RESERVE_TOKENS
    assert depth >= workload.DEEP_MEASUREMENT_MIN_TOKENS


def test_deep_prompt_tokens_returns_none_when_ctx_too_close_to_floor(
    tmp_path: Path,
) -> None:
    """If subtracting the reserve drops below the deep-measurement floor,
    measure shallow rather than request a prompt that cannot fit."""
    cfg = _make_cfg(tmp_path)
    cfg.workload.context_length = "long"
    recipe = {"ctx": 32768}

    assert workload.deep_prompt_tokens(cfg, recipe) is None


def test_deep_prompt_tokens_unknown_ctx_unchanged(tmp_path: Path) -> None:
    """When the recipe does not declare ctx, the declared target is used
    exactly; the caller is responsible for ensuring it fits."""
    cfg = _make_cfg(tmp_path)
    cfg.workload.context_length = "very_long"

    assert workload.deep_prompt_tokens(cfg, {}) == 131072
    assert workload.deep_prompt_tokens(cfg, {"ctx": 0}) == 131072


def test_deep_prompt_tokens_short_profile_returns_none(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.workload.context_length = "short"

    assert workload.deep_prompt_tokens(cfg, {"ctx": 131072}) is None


# ---------------------------------------------------------------------------
# init gathers the profile via flags (never blocking Docker/CI).
# ---------------------------------------------------------------------------


def _stub_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: Config) -> Path:
    config_file = tmp_path / "config.toml"
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
    monkeypatch.setattr("arc_llama.cli._print_gpu_table", lambda gpus: None)
    return config_file


def test_init_workload_flags_persist_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_cfg(tmp_path)
    config_file = _stub_init(monkeypatch, tmp_path, cfg)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_file),
            "init",
            "--no-scan",
            "--workload-context",
            "long",
            "--workload-style",
            "agentic",
            "--workload-priority",
            "throughput",
        ],
    )

    assert result.exit_code == 0, result.output
    loaded = load_config(config_file)
    assert loaded.workload.context_length == "long"
    assert loaded.workload.style == "agentic"
    assert loaded.workload.priority == "throughput"


def test_init_not_sure_keeps_defaults_and_never_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-interactive stdin (Docker/CI) must not block on questions."""
    cfg = _make_cfg(tmp_path)
    config_file = _stub_init(monkeypatch, tmp_path, cfg)

    result = CliRunner().invoke(
        cli,
        ["--config", str(config_file), "init", "--no-scan"],
    )

    assert result.exit_code == 0, result.output
    loaded = load_config(config_file)
    assert loaded.workload.context_length == ""
    assert loaded.workload.style == ""
    assert loaded.workload.priority == ""
