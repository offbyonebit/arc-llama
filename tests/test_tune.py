"""Tests for arc_llama.tune — candidate generation, scoring, greedy sweep."""

from __future__ import annotations

import asyncio
import time

import pytest

from arc_llama.benchmark import BenchmarkResult
from arc_llama.config import (
    Config,
    GPUConfig,
    ModelConfig,
    PathsConfig,
    ServerConfig,
    TuneConfig,
)
from arc_llama.tune import (
    TuneReport,
    _probe_offload_info,
    _restore_edits,
    _ubatch_candidates,
    build_stages,
    print_multi_summary,
    score_result,
    tune_all,
    tune_model,
)


def _result(pp: float | None, gen: float | None, error: str | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        model="m",
        ctx=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        prompt_tokens=1024,
        gen_tokens=128,
        prompt_eval_tok_s=pp,
        generation_tok_s=gen,
        error=error,
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

    def test_mtp_model_still_sweeps_ubatch(self):
        # Measured B60/MTP: large ubatch is fine; do not skip the stage.
        stages = build_stages({"spec_type": "draft-mtp"}, vram_mb=24 * 1024)
        assert any("ubatch_size" in step.edits for st in stages for step in st)

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
        gpus=[
            GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24 * 1024)
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
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "ubatch_size": 1024,
        "batch_size": 2048,
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


async def test_tune_raising_candidate_loses(cfg, monkeypatch):
    # f16 KV crashes the backend (exception) — tuner must not abort.
    table = {
        ("q8_0", 512, None): (800.0, 30.0),
    }
    fake = FakeMeasurements(table, cfg)

    async def bench(server_url, model_name, **kw):
        if fake.state.get("cache_type_k") == "f16":
            raise RuntimeError("llama-server exited with SIGSEGV")
        return await FakeMeasurements.bench(fake, server_url, model_name, **kw)

    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)
    assert report.error is None
    # No KV candidate beat the baseline, so best_edits stays empty and the
    # baseline recipe (q8_0) is restored.
    assert report.best_edits == {}
    assert report.baseline.cache_type_k == "q8_0"
    failed = [s for s in report.steps if s.edits.get("cache_type_k") == "f16"]
    assert failed
    assert all(s.result is not None and s.result.error and not s.chosen for s in failed)


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


async def test_on_stage_called_for_each_stage_in_order(cfg, monkeypatch):
    """The stage callback must fire synchronously for every stage, in order.

    Regression test for the async-callback defect: tune.py calls on_stage
    synchronously (like tune_all's on_start/on_done), so a coroutine callback
    was never awaited and the reported stage never advanced past baseline.
    """
    table = {("q8_0", 512, None): (800.0, 30.0)}
    fake = FakeMeasurements(table, cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    calls: list[tuple[str, int, int]] = []
    report = await tune_model(
        "http://127.0.0.1:11437",
        "m",
        cfg=cfg,
        on_stage=lambda name, i, total: calls.append((name, i, total)),
    )

    assert report.error is None
    assert [i for _, i, _ in calls] == [1, 2, 3]
    assert all(total == 3 for _, _, total in calls)
    assert calls[0][0].startswith("kv=")
    assert calls[1][0].startswith("ubatch=")
    assert calls[2][0].startswith("fa=")


async def test_probe_server_caps_does_not_block_event_loop(cfg, monkeypatch):
    """A slow `llama-server --help` probe must not stall the event loop.

    /admin/tune/status was observed timing out at sweep start; the probe is a
    synchronous subprocess.run, so tune_model must run it in a thread.
    """
    import arc_llama.server_caps as caps_mod
    from arc_llama.server_caps import ServerCaps

    probe_start = probe_end = None

    def slow_probe(path):
        nonlocal probe_start, probe_end
        probe_start = time.monotonic()
        time.sleep(0.25)
        probe_end = time.monotonic()
        return ServerCaps(supports_flash_attn=True)

    monkeypatch.setattr(caps_mod, "probe_server_caps", slow_probe)
    fake = FakeMeasurements({("q8_0", 512, None): (800.0, 30.0)}, cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    tick_times: list[float] = []

    async def ticker():
        for _ in range(40):
            await asyncio.sleep(0.02)
            tick_times.append(time.monotonic())

    await asyncio.gather(tune_model("http://127.0.0.1:11437", "m", cfg=cfg), ticker())

    assert probe_start is not None and probe_end is not None
    # At least one ticker callback must have run while the probe was sleeping.
    assert any(probe_start < t < probe_end for t in tick_times)


# ---------------------------------------------------------------------------
# tune_all: fleet-wide sweep orchestration
# ---------------------------------------------------------------------------


async def test_tune_all_runs_each_model_in_order(monkeypatch, cfg):
    calls: list[str] = []
    starts: list[tuple[str, int, int]] = []

    async def fake_tune_model(server_url, model_name, **kwargs):
        calls.append(model_name)
        return TuneReport(model=model_name, target="balanced", applied=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    def on_start(name, i, total):
        starts.append((name, i, total))

    reports = await tune_all("http://x", ["a", "b", "c"], cfg=cfg, on_start=on_start)

    assert len(reports) == 3
    assert [r.model for r in reports] == ["a", "b", "c"]
    assert calls == ["a", "b", "c"]
    assert starts == [("a", 1, 3), ("b", 2, 3), ("c", 3, 3)]


async def test_tune_all_one_failure_does_not_abort(monkeypatch, cfg):
    async def fake_tune_model(server_url, model_name, **kwargs):
        if model_name == "b":
            raise RuntimeError("boom")
        return TuneReport(model=model_name, target="balanced", applied=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    reports = await tune_all("http://x", ["a", "b", "c"], cfg=cfg)

    assert len(reports) == 3
    assert [r.model for r in reports] == ["a", "b", "c"]
    assert reports[0].error is None
    assert reports[1].error is not None
    assert "boom" in reports[1].error
    assert reports[2].error is None


def test_print_multi_summary_smoke():
    ok = TuneReport(model="a", target="balanced", applied=True)
    ok.baseline = _result(100.0, 50.0)
    ok.best = _result(120.0, 60.0)

    errored = TuneReport(model="b", target="balanced", error="broke")

    # Must not raise.
    print_multi_summary([ok, errored])


# ---------------------------------------------------------------------------
# Round 5 phase 2: MoE expert offload as a measured axis
# ---------------------------------------------------------------------------


def _patch_moe_scan(monkeypatch, total_gib, per_layer_mib, n_layers):
    """Fake the GGUF tensor table everywhere the sweep reads it."""
    scan = (int(total_gib * 1024**3), {i: per_layer_mib * 1024**2 for i in range(n_layers)})
    monkeypatch.setattr("arc_llama.gguf_meta.scan_weight_tensors", lambda _p: scan)
    monkeypatch.setattr("arc_llama.router.scan_weight_tensors", lambda _p: scan)
    return scan


class MoEMeasurements(FakeMeasurements):
    """Fake backend that knows about n_cpu_moe: configs with fewer than
    `min_loadable_n` offloaded layers OOM at load; perf is keyed by
    (kv, n) with a (kv, None) fallback."""

    def __init__(self, cfg, min_loadable_n=0, perf=None):
        super().__init__({}, cfg)
        self.min_loadable_n = min_loadable_n
        self.perf = perf or {}
        self.bench_calls: list[tuple] = []

    async def bench(self, server_url, model_name, **kw):
        n = self.state.get("n_cpu_moe") or 0
        kv = self.state.get("cache_type_k", "f16")
        ub = self.state.get("ubatch_size", 512)
        self.bench_calls.append((kv, ub, n))
        if n < self.min_loadable_n:
            return _result(None, None, error="llama-server did not become healthy")
        pp, gen = self.perf.get((kv, n), self.perf.get((kv, None), (1000.0, 30.0)))
        return _result(pp, gen)


def _offload_steps(report):
    return [s for s in report.steps if "n_cpu_moe" in s.edits]


async def test_offload_stage_skipped_when_model_fits_without_offload(cfg, monkeypatch):
    """MoE model that fits comfortably at zero offload: the stage records
    its skip reason and never measures an offload candidate."""
    _patch_moe_scan(monkeypatch, 4, 512, 4)
    fake = MoEMeasurements(cfg, perf={("q8_0", None): (800.0, 30.0)})
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    calls: list[tuple[str, int, int]] = []
    report = await tune_model(
        "http://127.0.0.1:11437",
        "m",
        cfg=cfg,
        on_stage=lambda name, i, total: calls.append((name, i, total)),
    )

    assert report.error is None
    assert _offload_steps(report) == []
    skips = [
        s for s in report.steps if s.skipped_reason and "fits without offload" in s.skipped_reason
    ]
    assert len(skips) == 1
    assert "n_cpu_moe" not in report.best_edits
    # The skipped stage frees its slot: ubatch/FA are numbered 2..3 of 3.
    # (The KV stage fired before the skip was known, so its total still
    # counts the offload slot — a status-display wobble, not a stage order.)
    assert [i for _, i, _ in calls] == [1, 2, 3]
    assert [total for _, _, total in calls[1:]] == [3, 3]


async def test_offload_stage_runs_after_kv_and_uses_winning_kv(cfg, monkeypatch):
    """28 GiB MoE on a 25 GB card: f16 KV needs 6 offloaded layers, q8_0
    needs 5. When f16 wins the KV stage the offload stage must target 6."""
    cfg.gpus[0].vram_mb = 25000
    _patch_moe_scan(monkeypatch, 28, 1024, 16)
    perf = {("f16", None): (1200.0, 40.0), ("q8_0", None): (1000.0, 30.0)}
    fake = MoEMeasurements(cfg, perf=perf)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    calls: list[tuple[str, int, int]] = []
    report = await tune_model(
        "http://127.0.0.1:11437",
        "m",
        cfg=cfg,
        on_stage=lambda name, i, total: calls.append((name, i, total)),
    )

    assert report.error is None
    assert report.best_edits["cache_type_k"] == "f16"
    # 6 = minimum against the winning f16 KV; 4 = one probe step below.
    assert [s.edits["n_cpu_moe"] for s in _offload_steps(report)] == [6, 4]
    assert report.best_edits["n_cpu_moe"] == 6
    assert fake.state["n_cpu_moe"] == 6
    # Ordering: the first edit carrying the tuned offload comes after the
    # KV winner was applied.
    first_kv_win = next(i for i, e in enumerate(fake.edits_seen) if e.get("cache_type_k") == "f16")
    first_offload = next(i for i, e in enumerate(fake.edits_seen) if e.get("n_cpu_moe") == 6)
    assert first_kv_win < first_offload
    # Stage numbering: KV, offload, ubatch, FA = 4 stages.
    assert [i for _, i, _ in calls] == [1, 2, 3, 4]
    assert calls[1][0].startswith("n_cpu_moe")


async def test_offload_oom_candidate_loses_and_recipe_stays_loadable(cfg, monkeypatch):
    """The estimated minimum OOMs at load (estimator optimistic): the sweep
    escalates, the failing candidate is recorded but never wins, and the
    model is left on a recipe that actually loads."""
    cfg.gpus[0].vram_mb = 20000
    _patch_moe_scan(monkeypatch, 20, 512, 16)  # estimated minimum: 4 layers
    perf = {("q8_0", None): (1000.0, 30.0), ("f16", None): (800.0, 25.0)}
    fake = MoEMeasurements(cfg, min_loadable_n=6, perf=perf)
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert report.best_edits["n_cpu_moe"] == 6
    assert fake.state["n_cpu_moe"] == 6  # left on a loadable recipe
    failed = [s for s in _offload_steps(report) if s.result and s.result.error]
    assert failed and all(not s.chosen for s in failed)
    assert {s.edits["n_cpu_moe"] for s in failed} == {4}
    # The baseline itself was rescued by the offload search.
    assert report.baseline is not None and report.baseline.error is None
    assert any(s.label.startswith("baseline (n_cpu_moe=6)") for s in report.steps)


async def test_sweep_starts_from_min_feasible_offload_when_zero_would_oom(cfg, monkeypatch):
    """A model that does not load at zero offload: the baseline must be
    measured from the minimum feasible offload, not the current recipe."""
    _patch_moe_scan(monkeypatch, 30, 1024, 16)  # minimum feasible: 8 layers
    fake = MoEMeasurements(cfg, min_loadable_n=8, perf={("q8_0", None): (950.0, 30.0)})
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    # The very first measurement already runs at the seeded minimum.
    assert fake.bench_calls[0][2] == 8
    # The below-probe (6) OOMs: recorded as information, 8 stands.
    tried = [s.edits["n_cpu_moe"] for s in _offload_steps(report)]
    assert tried == [8, 6]
    below = _offload_steps(report)[1]
    assert below.result is not None and below.result.error and not below.chosen
    assert report.best_edits["n_cpu_moe"] == 8
    assert fake.state["n_cpu_moe"] == 8


async def test_ubatch_candidates_pruned_under_chosen_offload(cfg, monkeypatch):
    """With 8 layers offloaded, ubatch 1024's larger compute buffer no
    longer fits: it must be skipped, never measured into an OOM. Also
    covers: the tuned value (8) overrides the registration guess (10)."""
    _patch_moe_scan(monkeypatch, 30, 1024, 16)
    cfg.models[0].recipe["n_cpu_moe"] = 10  # registration guess
    fake = MoEMeasurements(cfg, min_loadable_n=8, perf={("q8_0", None): (1000.0, 30.0)})
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg)

    assert report.error is None
    assert report.best_edits["n_cpu_moe"] == 8  # tuned value, not the guess
    assert fake.state["n_cpu_moe"] == 8
    ub1024 = [s for s in report.steps if s.edits.get("ubatch_size") == 1024]
    assert len(ub1024) == 1
    assert ub1024[0].result is None  # never measured
    assert "would not fit" in (ub1024[0].skipped_reason or "")
    assert not any(ub == 1024 for _, ub, _ in fake.bench_calls)
    # The fitting ubatch candidates were still measured.
    measured_ub = {
        s.edits.get("ubatch_size") for s in report.steps if s.result and "ubatch_size" in s.edits
    }
    assert measured_ub == {512, 256}


async def test_offload_seed_not_leaked_into_dry_run_restore(cfg, monkeypatch):
    """Dry run with an offload seed: the seed lives in best_edits, so the
    final restore returns the recipe to its true original (no n_cpu_moe)."""
    _patch_moe_scan(monkeypatch, 30, 1024, 16)  # minimum feasible: 8 layers
    fake = MoEMeasurements(cfg, min_loadable_n=8, perf={("q8_0", None): (950.0, 30.0)})
    monkeypatch.setattr("arc_llama.tune._apply_edits", fake.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", fake.bench)

    report = await tune_model("http://127.0.0.1:11437", "m", cfg=cfg, apply=False)

    assert report.error is None
    assert report.best_edits["n_cpu_moe"] == 8
    assert not report.applied
    assert "n_cpu_moe" not in fake.state  # original recipe restored


class TestProbeOffloadInfo:
    def test_returns_none_when_gpu_missing(self):
        model = ModelConfig(name="m", path="/m.gguf", port=1, gpu_pci_slot="pci", recipe={})
        assert _probe_offload_info(model, None) is None

    def test_returns_none_when_vram_unknown(self):
        model = ModelConfig(name="m", path="/m.gguf", port=1, gpu_pci_slot="pci", recipe={})
        gpu = GPUConfig(pci_slot="pci", sycl_index=0, arch="battlemage", vram_mb=None)
        assert _probe_offload_info(model, gpu) is None


class TestBaselineErrorGuard:
    @pytest.mark.asyncio
    async def test_aborted_baseline_does_not_dereference_none_result(self, monkeypatch):
        import arc_llama.workload as wl
        from arc_llama import tune as tune_mod

        cfg = Config(
            server=ServerConfig(admin_token=None),
            models=[
                ModelConfig(
                    name="m", path="/m.gguf", port=1, gpu_pci_slot="pci", recipe={"ctx": 4096}
                )
            ],
            gpus=[GPUConfig(pci_slot="pci", sycl_index=0, arch="battlemage")],
            tune=TuneConfig(auto=False),
        )

        monkeypatch.setattr(
            "arc_llama.arch.profile_for", lambda arch: type("P", (), {"safe_kv_q8": True})()
        )
        monkeypatch.setattr(
            "arc_llama.server_caps.probe_server_caps",
            lambda path: type(
                "C", (), {"supports_flash_attn": True, "flash_attn_takes_value": True}
            )(),
        )
        monkeypatch.setattr(wl, "target_ctx", lambda cfg: None)
        monkeypatch.setattr(wl, "score_priority", lambda cfg: None)
        monkeypatch.setattr(wl, "deep_prompt_tokens", lambda cfg, recipe: None)
        monkeypatch.setattr(wl, "stage_is_depth_sensitive", lambda axes: False)

        async def noop_apply(*a, **kw):
            return None

        monkeypatch.setattr(tune_mod, "_apply_edits", noop_apply)

        report = await tune_model(
            "http://127.0.0.1:1",
            "m",
            cfg=cfg,
            should_abort=lambda: True,
        )
        assert report.aborted is True
        assert report.error == "baseline measurement failed"


class TestOverrideTensorViability:
    @pytest.mark.asyncio
    async def test_unknown_vram_does_not_crash_viability_filter(self, monkeypatch, tmp_path):
        import arc_llama.workload as wl
        from arc_llama import tune as tune_mod

        cfg = Config(
            server=ServerConfig(admin_token=None),
            models=[
                ModelConfig(
                    name="m",
                    path=str(tmp_path / "m.gguf"),
                    port=1,
                    gpu_pci_slot="pci",
                    recipe={"ctx": 4096},
                )
            ],
            gpus=[GPUConfig(pci_slot="pci", sycl_index=0, arch="battlemage", vram_mb=None)],
            tune=TuneConfig(auto=False),
        )

        monkeypatch.setattr(
            "arc_llama.arch.profile_for", lambda arch: type("P", (), {"safe_kv_q8": True})()
        )
        monkeypatch.setattr(
            "arc_llama.server_caps.probe_server_caps",
            lambda path: type(
                "C", (), {"supports_flash_attn": False, "flash_attn_takes_value": True}
            )(),
        )
        monkeypatch.setattr(wl, "target_ctx", lambda cfg: None)
        monkeypatch.setattr(wl, "score_priority", lambda cfg: None)
        monkeypatch.setattr(wl, "deep_prompt_tokens", lambda cfg, recipe: None)
        monkeypatch.setattr(wl, "stage_is_depth_sensitive", lambda axes: False)

        monkeypatch.setattr(
            tune_mod,
            "_probe_offload_info",
            lambda model, gpu: tune_mod._OffloadInfo(n_layers=8, vram_mb=16000),
        )
        monkeypatch.setattr(tune_mod, "min_moe_offload_layers", lambda *a, **kw: 5)
        monkeypatch.setattr(tune_mod, "weight_tensor_table", lambda path: {"t": 1024 * 1024})
        monkeypatch.setattr(
            tune_mod, "propose_override_tensor_patterns", lambda table: ["p1", "p2"]
        )
        monkeypatch.setattr(tune_mod, "validate_override_patterns", lambda table, pats: (True, ""))
        monkeypatch.setattr(
            tune_mod, "override_tensor_saved_bytes", lambda table, pats: 1024 * 1024
        )
        monkeypatch.setattr(tune_mod, "_estimate_model_vram_mb", lambda *a, **kw: 1000)

        async def fake_benchmark(*a, **kw):
            return BenchmarkResult(
                model="m",
                ctx=4096,
                cache_type_k="f16",
                cache_type_v="f16",
                prompt_tokens=1,
                gen_tokens=1,
                prompt_eval_tok_s=10.0,
                generation_tok_s=10.0,
            )

        monkeypatch.setattr(tune_mod, "benchmark_model", fake_benchmark)

        async def noop_apply(*a, **kw):
            return None

        monkeypatch.setattr(tune_mod, "_apply_edits", noop_apply)

        report = await tune_model("http://127.0.0.1:1", "m", cfg=cfg, apply=False)
        assert report.error is None
        assert report.best_edits.get("n_cpu_moe") == 5
        assert "override_tensor" not in report.best_edits


# ---------------------------------------------------------------------------
# #30 / #37 — abort re-check inside measure, restore retry + fallback
# ---------------------------------------------------------------------------


async def test_measure_rechecks_abort_after_the_edit_await(cfg, monkeypatch):
    """#30: a real request registers while the candidate edit POST is in
    flight — after the sweep-level abort check has already run. measure()
    must re-check after that await and skip the benchmark, whose load=True
    would evict the user's just-started model."""
    fake = FakeMeasurements({}, cfg)
    bench_calls = 0
    apply_calls = 0
    abort_now = False

    async def apply(client, name, edits):
        nonlocal apply_calls, abort_now
        apply_calls += 1
        if apply_calls == 2:
            # Lands mid-sweep: baseline applied+measured, first candidate edit
            # is in flight, and the user's request has just registered.
            abort_now = True
        return await fake.apply(client, name, edits)

    async def bench(server_url, model_name, **kw):
        nonlocal bench_calls
        bench_calls += 1
        return await fake.bench(server_url, model_name, **kw)

    monkeypatch.setattr("arc_llama.tune._apply_edits", apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", bench)

    report = await tune_model(
        "http://127.0.0.1:11437",
        "m",
        cfg=cfg,
        should_abort=lambda: abort_now,
    )

    assert report.aborted
    assert bench_calls == 1, "a candidate was benchmarked after the abort landed"
    aborted = [s for s in report.steps if s.skipped_reason == "aborted before measurement"]
    assert aborted, "the aborted candidate was not recorded"


async def test_restore_final_state_retries_then_succeeds(cfg, monkeypatch):
    """#37: a flaky edit endpoint must not leave the last candidate persisted
    when a retry would have worked."""
    from arc_llama.tune import _restore_final_state

    calls = 0

    async def flaky(client, name, edits):
        nonlocal calls
        calls += 1
        return "boom" if calls < 3 else None

    monkeypatch.setattr("arc_llama.tune._apply_edits", flaky)
    err = await _restore_final_state(None, "m", {"cache_type_k": "f16"}, cfg)
    assert err is None
    assert calls == 3


async def test_restore_final_state_direct_write_fallback(cfg, monkeypatch):
    """#37: endpoint permanently down → the intended recipe is written to the
    local config directly rather than leaving the last candidate on disk.
    XDG_CONFIG_HOME is redirected to tmp by conftest."""
    from arc_llama.config import default_config_path, load_config
    from arc_llama.tune import _restore_final_state

    async def dead(client, name, edits):
        return "connection refused"

    monkeypatch.setattr("arc_llama.tune._apply_edits", dead)
    # What the last candidate edit left behind.
    cfg.models[0].recipe = {"cache_type_k": "q8_0", "n_cpu_moe": 4}

    err = await _restore_final_state(None, "m", {"cache_type_k": "f16", "n_cpu_moe": None}, cfg)

    assert err is None
    assert cfg.models[0].recipe.get("cache_type_k") == "f16"
    assert "n_cpu_moe" not in cfg.models[0].recipe
    on_disk = load_config(default_config_path())
    restored = on_disk.find_model("m")
    assert restored is not None
    assert restored.recipe.get("cache_type_k") == "f16"
    assert "n_cpu_moe" not in restored.recipe


async def test_restore_final_state_reports_error_when_everything_fails(cfg, monkeypatch):
    """#37: endpoint down AND local write failing → the caller gets the error
    (and a critical log), never a silent bad recipe."""
    from arc_llama.tune import _restore_final_state

    async def dead(client, name, edits):
        return "connection refused"

    def bad_save(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("arc_llama.tune._apply_edits", dead)
    monkeypatch.setattr(type(cfg), "save", bad_save)

    err = await _restore_final_state(None, "m", {"cache_type_k": "f16"}, cfg)
    assert err == "connection refused"


async def test_apply_state_to_recipe_mirrors_endpoint_rules():
    """None clears, n_cpu_moe=0 clears, n_cpu_moe and override_tensor exclude
    each other — the fallback must persist exactly what the endpoint would."""
    from arc_llama.tune import _apply_state_to_recipe

    m = ModelConfig(
        name="m",
        path="/m.gguf",
        port=1,
        gpu_pci_slot="g",
        recipe={"cache_type_k": "q8_0", "n_cpu_moe": 4},
    )
    _apply_state_to_recipe(m, {"override_tensor": ["p"]})
    assert m.recipe == {"cache_type_k": "q8_0", "override_tensor": ["p"]}
    _apply_state_to_recipe(m, {"n_cpu_moe": 2})
    assert m.recipe == {"cache_type_k": "q8_0", "n_cpu_moe": 2}
    _apply_state_to_recipe(m, {"n_cpu_moe": 0})
    assert m.recipe == {"cache_type_k": "q8_0"}
    _apply_state_to_recipe(m, {"flash_attn": None, "ubatch_size": 1024})
    assert m.recipe == {"cache_type_k": "q8_0", "ubatch_size": 1024}
