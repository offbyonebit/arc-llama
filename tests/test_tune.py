"""Tests for arc_llama.tune — candidate generation, scoring, greedy sweep."""
from __future__ import annotations

import pytest

from arc_llama.benchmark import BenchmarkResult
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig
from arc_llama.tune import (
    _restore_edits,
    _ubatch_candidates,
    build_stages,
    score_result,
    tune_model,
)


def _result(pp: float | None, gen: float | None, error: str | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        model="m", ctx=8192, cache_type_k="q8_0", cache_type_v="q8_0",
        prompt_tokens=1024, gen_tokens=128,
        prompt_eval_tok_s=pp, generation_tok_s=gen, error=error,
    )


class TestScore:
    def test_targets(self):
        r = _result(1000.0, 40.0)
        assert score_result(r, "prompt") == 1000.0
        assert score_result(r, "generation") == 40.0
        balanced = score_result(r, "balanced")
        assert balanced == pytest.approx((1000.0 * 40.0) ** 0.5)

    def test_error_loses(self):
        assert score_result(_result(1000.0, 40.0, error="boom")) is None

    def test_missing_measurement_loses(self):
        assert score_result(_result(None, 40.0), "balanced") is None
        assert score_result(_result(None, 40.0), "prompt") is None
        assert score_result(_result(None, 40.0), "generation") == 40.0


class TestUbatchCandidates:
    def test_default_current(self):
        # current 512 → try 512, 256, 1024
        assert _ubatch_candidates(None, 24 * 1024) == [512, 256, 1024]

    def test_from_1024(self):
        assert _ubatch_candidates(1024, 24 * 1024) == [1024, 512, 2048]

    def test_2048_blocked_on_small_cards(self):
        assert 2048 not in _ubatch_candidates(1024, 8 * 1024)

    def test_top_of_ladder(self):
        assert _ubatch_candidates(2048, 24 * 1024) == [2048, 1024]


class TestBuildStages:
    def test_all_stages(self):
        stages = build_stages({}, safe_kv_q8=True, fa_supported=True, vram_mb=24 * 1024)
        labels = [[s.label for s in st] for st in stages]
        assert labels[0] == ["kv=f16", "kv=q8_0"]
        assert labels[1][0].startswith("ubatch=")
        assert labels[2] == ["fa=on", "fa=off", "fa=auto"]

    def test_mtp_model_skips_ubatch_stage(self):
        stages = build_stages({"spec_type": "draft-mtp"}, vram_mb=24 * 1024)
        for stage in stages:
            for step in stage:
                assert "ubatch_size" not in step.edits

    def test_no_fa_support_drops_fa_stage(self):
        stages = build_stages({}, fa_supported=False)
        assert all(not s.label.startswith("fa=") for st in stages for s in st)

    def test_old_style_fa_binary_gets_boolean_options(self):
        stages = build_stages({}, fa_supported=True, fa_takes_value=False)
        fa_stage = [st for st in stages if st[0].label.startswith("fa=")][0]
        assert [s.label for s in fa_stage] == ["fa=on", "fa=off"]

    def test_unsafe_kv_q8_only_offers_f16(self):
        stages = build_stages({}, safe_kv_q8=False)
        assert [s.label for s in stages[0]] == ["kv=f16"]

    def test_ubatch_stage_keeps_batch_above_ubatch(self):
        stages = build_stages({}, vram_mb=24 * 1024)
        for step in stages[1]:
            assert step.edits["batch_size"] >= step.edits["ubatch_size"]


class TestRestoreEdits:
    def test_restores_original_values(self):
        original = {"cache_type_k": "q8_0", "cache_type_v": "q8_0", "ubatch_size": 8}
        out = _restore_edits(original, {"cache_type_k", "cache_type_v", "ubatch_size"})
        assert out == original

    def test_unset_axes_restore_to_explicit_defaults(self):
        out = _restore_edits({}, {"ubatch_size", "batch_size", "flash_attn"})
        assert out == {"ubatch_size": 512, "batch_size": 2048, "flash_attn": None}


# ---------------------------------------------------------------------------
# End-to-end greedy loop with a faked measurement backend
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00" * 16)
    return Config(
        paths=PathsConfig(llama_server=str(tmp_path / "no-such-llama-server")),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage",
                        vram_mb=24 * 1024)],
        models=[ModelConfig(
            name="m", path=str(gguf), port=18080, gpu_pci_slot="0000:03:00.0",
            recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        )],
    )


class FakeMeasurements:
    """Deterministic tok/s per config, keyed by (kv, ubatch, fa)."""

    def __init__(self, table, cfg):
        self.table = table
        self.cfg = cfg
        self.edits_seen: list[dict] = []
        # Applied state, mirroring what the admin endpoint would persist.
        self.state = dict(cfg.models[0].recipe)

    async def apply(self, client, name, edits):
        self.edits_seen.append(dict(edits))
        self.state.update({k: v for k, v in edits.items() if v is not None})
        for k, v in edits.items():
            if v is None:
                self.state.pop(k, None)
        return None

    async def bench(self, server_url, model_name, **kw):
        key = (
            self.state.get("cache_type_k", "f16"),
            self.state.get("ubatch_size", 512),
            self.state.get("flash_attn"),
        )
        pp, gen = self.table.get(key, (100.0, 10.0))
        return _result(pp, gen)


async def test_tune_picks_and_applies_winner(cfg, monkeypatch):
    # q8_0 KV baseline. f16 KV slightly better gen; ubatch 1024 much better
    # prompt; fa=on better still. Winner should combine all three.
    table = {
        ("q8_0", 512, None): (800.0, 30.0),
        ("f16", 512, None): (820.0, 33.0),
        ("f16", 256, None): (500.0, 33.0),
        ("f16", 1024, None): (1300.0, 33.0),
        ("f16", 1024, "on"): (1400.0, 34.0),
        ("f16", 1024, "off"): (700.0, 20.0),
        ("f16", 1024, "auto"): (1350.0, 33.5),
    }
    fake = FakeMeasurements(table, cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert report.best_edits == {
        "cache_type_k": "f16", "cache_type_v": "f16",
        "ubatch_size": 1024, "batch_size": 2048,
        "flash_attn": "on",
    }
    assert report.applied
    # The final persisted state must equal original + winning edits.
    assert fake.state["cache_type_k"] == "f16"
    assert fake.state["ubatch_size"] == 1024
    assert fake.state["flash_attn"] == "on"
    imp = report.improvement_pct
    assert imp["prompt_eval"] == pytest.approx(75.0, abs=0.2)


async def test_tune_keeps_baseline_when_nothing_beats_it(cfg, monkeypatch):
    table = {("q8_0", 512, None): (2000.0, 50.0)}  # everything else: (100, 10)
    fake = FakeMeasurements(table, cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert report.best_edits == {}
    assert not report.applied
    # Final state must be back to the original recipe values.
    assert fake.state.get("cache_type_k") == "q8_0"
    assert fake.state.get("ubatch_size") == 512  # explicit llama.cpp default
    assert fake.state.get("flash_attn") is None


async def test_tune_dry_run_restores_original(cfg, monkeypatch):
    table = {
        ("q8_0", 512, None): (800.0, 30.0),
        ("f16", 512, None): (2000.0, 60.0),
    }
    fake = FakeMeasurements(table, cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg, apply=False)

    assert report.best_edits.get("cache_type_k") == "f16"
    assert not report.applied
    assert fake.state.get("cache_type_k") == "q8_0"


async def test_tune_failing_candidate_loses(cfg, monkeypatch):
    # ubatch=1024 OOMs (error) — tuner must not select it and must not die.
    table = {
        ("q8_0", 512, None): (800.0, 30.0),
        ("f16", 512, None): (700.0, 25.0),
    }
    fake = FakeMeasurements(table, cfg)

    async def bench(server_url, model_name, **kw):
        if fake.state.get("ubatch_size") == 1024:
            return _result(None, None, error="llama-server did not become healthy")
        return await FakeMeasurements.bench(fake, server_url, model_name, **kw)

    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)
    assert report.error is None
    assert "ubatch_size" not in report.best_edits


async def test_tune_unknown_model(cfg):
    report = await tune_model("http://127.0.0.1:11437", "nope", cfg=cfg)
    assert report.error is not None


async def test_tune_baseline_failure_aborts(cfg, monkeypatch):
    async def bench(server_url, model_name, **kw):
        return _result(None, None, error="model never came up")

    async def apply(client, name, edits):
        return None

    monkeypatch.setattr("arc_llama.tune._apply_edits", apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)
    assert report.error is not None
    assert "baseline" in report.error
