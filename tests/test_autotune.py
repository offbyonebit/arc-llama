"""Tests for arc_llama.autotune.

All tests use fakes. No llama-server process, no GPU probing, no systemd, no
tuning search implementation — autotune.py owns *when* to run, tune.py owns
*what* to search. The tests make sure the two are wired correctly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from arc_llama.autotune import (
    Autotuner,
    compute_fingerprint,
    set_tuned_state,
)
from arc_llama.config import Config, GPUConfig, ModelConfig, PathsConfig, TuneConfig
from arc_llama.tune import TuneReport, tune_model


def _make_gguf(path: Path) -> str:
    # A minimal GGUF header-ish blob so estimate_weight_vram_bytes doesn't
    # bail out of the metadata path.
    path.write_bytes(b"\x00" * 64)
    return str(path)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00" * 64)
    cfg = Config(
        paths=PathsConfig(
            llama_server=str(tmp_path / "llama-server"),
            models_dir=str(tmp_path / "models"),
        ),
        tune=TuneConfig(auto=True, idle_seconds=2, min_uses=1),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24 * 1024,
            ),
        ],
        models=[
            ModelConfig(
                name="m",
                path=str(gguf),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            ),
        ],
    )
    return cfg


@pytest.fixture
def router(cfg: Config):
    """A fake router with the two activity counters autotune reads."""

    class FakeRouter:
        def __init__(self) -> None:
            self.cfg = cfg
            self.last_activity = 0.0
            self.inflight = 0
            self._servers: dict[str, Any] = {}

        def all_models(self):
            return cfg.models

        def running_models(self):
            return [n for n, s in self._servers.items() if s is not None and s.is_running]

    return FakeRouter()


# ---------------------------------------------------------------------------
# 1. Fingerprint changes when inputs change; stable otherwise.
# ---------------------------------------------------------------------------


def test_fingerprint_stable_unchanged_inputs(cfg: Config) -> None:
    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    fp1 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    fp2 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    assert fp1 == fp2


def test_fingerprint_changes_when_llama_server_mtime_changes(cfg: Config, tmp_path: Path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("x")
    cfg.paths.llama_server = str(server)
    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    fp1 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    import os

    # Bump mtime while keeping size the same.
    os.utime(server, (1234567890.0, 1234567891.0))
    fp2 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    assert fp1 != fp2


def test_fingerprint_changes_when_model_path_changes(cfg: Config, tmp_path: Path) -> None:
    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    fp1 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    cfg.models[0].path = str(tmp_path / "other.gguf")
    (tmp_path / "other.gguf").write_bytes(b"\x00" * 64)
    fp2 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    assert fp1 != fp2


def test_fingerprint_changes_when_gpu_arch_changes(cfg: Config) -> None:
    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    fp1 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    gpu.arch = "alchemist"
    fp2 = compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.5.0")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# 7. State round-trips through Config.save / load_config.
# ---------------------------------------------------------------------------


def test_tune_state_round_trip_through_config(cfg: Config, tmp_path: Path) -> None:
    from arc_llama.config import load_config

    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    path = tmp_path / "config.toml"
    set_tuned_state(
        cfg, cfg.models[0], compute_fingerprint(cfg.models[0], cfg.paths.llama_server, gpu, "0.6.0")
    )

    cfg.save(path)
    loaded = load_config(path)

    assert len(loaded.models) == 1
    assert loaded.models[0].tune_state == "tuned"
    assert loaded.models[0].tune_fingerprint == cfg.models[0].tune_fingerprint
    assert loaded.models[0].tuned_at == pytest.approx(cfg.models[0].tuned_at, abs=0.1)
    assert loaded.tune.auto is True
    assert loaded.tune.idle_seconds == 2


# ---------------------------------------------------------------------------
# 4 / 5. Abort / cancellation runs the final restore edit.
# ---------------------------------------------------------------------------


class EditRecorder:
    """Records all edit bodies; use as the _apply_edits fixture."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bodies: list[dict[str, Any]] = []
        self.state = dict(cfg.models[0].recipe)

    async def apply(self, _client, _name, edits: dict[str, Any]) -> None:
        self.bodies.append(dict(edits))
        for k, v in edits.items():
            if v is None:
                self.state.pop(k, None)
            else:
                self.state[k] = v

    async def bench(self, _server_url, _model_name, **kw) -> Any:
        from arc_llama.benchmark import BenchmarkResult

        key = (
            self.state.get("cache_type_k", "f16"),
            self.state.get("ubatch_size", 512),
            self.state.get("flash_attn"),
        )
        pp = {"q8_0": 800.0, "f16": 1000.0}.get(key[0], 100.0)
        gen = {"on": 40.0, None: 30.0}.get(key[2], 30.0)
        return BenchmarkResult(
            model="m",
            ctx=8192,
            cache_type_k=str(key[0]),
            cache_type_v=str(key[0]),
            prompt_tokens=1024,
            gen_tokens=128,
            prompt_eval_tok_s=pp,
            generation_tok_s=gen,
        )


async def test_preemption_restores_baseline_via_final_edit(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep aborted mid-stage must leave the recipe at the baseline state."""
    recorder = EditRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)
    monkeypatch.setattr("arc_llama.tune.benchmark_model", recorder.bench)

    calls: list[bool] = []

    def should_abort() -> bool:
        calls.append(True)
        return len(calls) >= 3

    report = await tune_model(
        "http://127.0.0.1:11437",
        "m",
        cfg=cfg,
        should_abort=should_abort,
    )

    assert report.aborted
    assert not report.error
    assert recorder.state.get("cache_type_k") == "q8_0"
    assert recorder.state.get("ubatch_size") == 512
    assert recorder.state.get("flash_attn") is None
    # The final edit body should equal the explicit baseline.
    assert recorder.bodies[-1] == {
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "ubatch_size": 512,
        "batch_size": 2048,
        "flash_attn": None,
    }


async def test_cancelled_error_in_measure_triggers_restore(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EditRecorder(cfg)
    monkeypatch.setattr("arc_llama.tune._apply_edits", recorder.apply)

    async def raise_cancel(*a, **kw):  # noqa: ARG001
        raise asyncio.CancelledError("simulated")

    monkeypatch.setattr("arc_llama.tune.benchmark_model", raise_cancel)

    with pytest.raises(asyncio.CancelledError):
        await tune_model(
            "http://127.0.0.1:11437",
            "m",
            cfg=cfg,
            should_abort=lambda: False,
        )

    assert recorder.bodies
    assert recorder.bodies[-1] == {
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "ubatch_size": 512,
        "batch_size": 2048,
        "flash_attn": None,
    }


# ---------------------------------------------------------------------------
# 2 / 3 / 6 / 8. Autotuner loop gates with fakes.
# ---------------------------------------------------------------------------


class FakeRouter:
    def __init__(self, cfg: Config, *, running: set[str] | None = None) -> None:
        self.cfg = cfg
        self.last_activity = 0.0
        self.inflight = 0
        self._running = running or set()
        self._servers: dict[str, Any] = {}

    def all_models(self):
        return self.cfg.models

    def running_models(self):
        return [n for n, s in self._servers.items() if s is not None and s.is_running]

    def _servers_for(self, name: str) -> Any:
        class Srv:
            is_running = name in self._running

        return Srv()


def _make_tuner(cfg: Config, router: Any, version: str = "0.6.0") -> Autotuner:
    return Autotuner(cfg, router, version=version, loop_interval=0.05)


async def test_candidate_skips_failed_models(cfg: Config) -> None:
    cfg.models[0].tune_state = "failed"
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    assert tuner._pick_candidate() is None


async def test_candidate_skips_models_under_min_uses(cfg: Config) -> None:
    cfg.tune.min_uses = 5
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    assert tuner._pick_candidate() is None


async def test_candidate_skips_models_with_matching_fingerprint(
    cfg: Config, tmp_path: Path
) -> None:
    gpu = cfg.find_gpu("0000:03:00.0")
    assert gpu is not None
    cfg.models[0].tune_fingerprint = compute_fingerprint(
        cfg.models[0], cfg.paths.llama_server, gpu, "0.6.0"
    )
    cfg.models[0].tune_state = "tuned"
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    assert tuner._pick_candidate() is None


async def test_idle_gate_recent_activity_produces_no_sweep(cfg: Config) -> None:
    calls: list[str] = []

    async def fake_tune(*args, **kwargs):  # noqa: ARG001
        calls.append("tune")
        return TuneReport(model="m", target="balanced")

    router = FakeRouter(cfg)
    router.last_activity = 99.0
    cfg.tune.idle_seconds = 120
    tuner = _make_tuner(cfg, router)
    await tuner.start()
    await asyncio.sleep(0.1)
    await tuner.stop()
    assert not calls


async def test_idle_gate_inflight_produces_no_sweep(cfg: Config) -> None:
    calls: list[str] = []

    async def fake_tune(*args, **kwargs):  # noqa: ARG001
        calls.append("tune")
        return TuneReport(model="m", target="balanced")

    router = FakeRouter(cfg)
    router.inflight = 1
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    await tuner.start()
    await asyncio.sleep(0.1)
    await tuner.stop()
    assert not calls


async def test_auto_false_starts_no_task(cfg: Config) -> None:
    cfg.tune.auto = False
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    await tuner.start()
    assert not tuner.is_running


async def test_multi_resident_core_running_marks_skipped(cfg: Config) -> None:
    cfg.server.single_resident = False
    router = FakeRouter(cfg)
    router._servers["other"] = type("Srv", (), {"is_running": True})()
    # Prevent autotune from reusing a stale matching fingerprint.
    cfg.models[0].tuned_at = None
    cfg.models[0].tune_fingerprint = ""
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")
    assert cfg.models[0].tune_state != "skipped"
    # Patch Autotuner._run_sweep so we never actually hit the network.
    import arc_llama.autotune as autotune_mod

    called_with: list = []

    async def fake_run_sweep(self, model):
        called_with.append(model.name)
        return TuneReport(model="m", target="balanced", applied=True)

    original = autotune_mod.Autotuner._run_sweep
    autotune_mod.Autotuner._run_sweep = fake_run_sweep
    try:
        await tuner._tick()
    finally:
        autotune_mod.Autotuner._run_sweep = original
    assert cfg.models[0].tune_state == "skipped"
    assert not called_with


async def test_successful_sweep_sets_tuned_and_fingerprint(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = FakeRouter(cfg)
    router.last_activity = -1000.0
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")

    async def fake_tune_model(*args, **kwargs):  # noqa: ARG001
        return TuneReport(model="m", target="balanced", applied=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    await tuner._tick()

    assert cfg.models[0].tune_state == "tuned"
    assert cfg.models[0].tune_fingerprint
    assert cfg.models[0].tuned_at is not None


async def test_failed_sweep_sets_failed(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(cfg)
    router.last_activity = -1000.0
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")

    async def fake_tune_model(*args, **kwargs):  # noqa: ARG001
        return TuneReport(model="m", target="balanced", error="boom")

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    await tuner._tick()

    assert cfg.models[0].tune_state == "failed"
    assert cfg.models[0].tune_error == "boom"


async def test_aborted_sweep_leaves_untuned(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(cfg)
    router.last_activity = -1000.0
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")

    async def fake_tune_model(*args, **kwargs):  # noqa: ARG001
        return TuneReport(model="m", target="balanced", aborted=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    await tuner._tick()

    assert cfg.models[0].tune_state == "untuned"
    assert cfg.models[0].tune_error == ""


async def test_deferred_restore_waits_for_inflight_to_drop(cfg: Config) -> None:
    """An aborted sweep must not POST the restore while a request is in flight."""
    router = FakeRouter(cfg)
    router.last_activity = -1000.0
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")

    bodies: list[dict[str, Any]] = []
    restore_started = asyncio.Event()
    restore_done = asyncio.Event()

    import arc_llama.tune as tune_mod

    async def fake_apply_edits(client, name, edits):
        bodies.append(dict(edits))
        restore_done.set()
        return None

    tune_mod._apply_edits = fake_apply_edits

    async def fake_tune_model(*args, **kwargs):
        on_deferred_restore = kwargs.get("on_deferred_restore")

        # Use the real deferred restore callback from Autotuner, but schedule
        # a concurrent task that drops inflight shortly after it starts waiting.
        # Wait until the restore is blocked, then return an aborted report.
        class Sentinel:
            pass

        sentinel = Sentinel()

        async def delayed_drop():
            while True:
                await asyncio.sleep(0.05)
                # The restore will have started and be polling on inflight.
                if getattr(sentinel, "started", False):
                    router.inflight = 0
                    return

        async def wrapper(final_state: dict[str, Any]) -> None:
            sentinel.started = True
            await on_deferred_restore(final_state)

        drop_task = asyncio.create_task(delayed_drop())
        await wrapper({"cache_type_k": "q8_0", "ubatch_size": 512})
        restore_started.set()
        await drop_task
        return TuneReport(model="m", target="balanced", aborted=True)

    tune_mod.tune_model = fake_tune_model

    router.inflight = 1
    tick_task = asyncio.create_task(tuner._run_sweep(cfg.models[0]))
    await asyncio.wait_for(restore_started.wait(), timeout=2.0)
    # Restore has run to completion after inflight dropped.
    await asyncio.wait_for(restore_done.wait(), timeout=2.0)
    await tick_task
    assert bodies == [{"cache_type_k": "q8_0", "ubatch_size": 512}]


async def test_run_sweep_stage_callback_advances_running_stage(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The autotuner's on_stage must be synchronous and advance running_stage.

    Regression test for the observed defect: an `async def` stage callback was
    passed to tune.py, which calls on_stage synchronously, so it was never
    awaited and /admin/tune/status showed sweep_stage "baseline" for the whole
    sweep.
    """
    import inspect

    router = FakeRouter(cfg)
    router.last_activity = -1000.0
    cfg.tune.idle_seconds = 0
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")

    seen: list[str | None] = []

    async def fake_tune_model(*args, **kwargs):
        on_stage = kwargs.get("on_stage")
        assert on_stage is not None
        assert not inspect.iscoroutinefunction(on_stage)
        on_stage("kv=f16", 1, 3)
        seen.append(tuner.running_stage)
        on_stage("ubatch=512", 2, 3)
        seen.append(tuner.running_stage)
        on_stage("fa=on", 3, 3)
        seen.append(tuner.running_stage)
        return TuneReport(model="m", target="balanced", applied=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", fake_tune_model)

    await tuner._run_sweep(cfg.models[0])

    assert seen == ["kv=f16 (1/3)", "ubatch=512 (2/3)", "fa=on (3/3)"]
    assert tuner.running_stage is None


async def test_pick_candidate_prefers_never_tuned_over_stale_fingerprint(cfg: Config) -> None:
    """A model that was just used but never tuned must be picked before a stale-fingerprint one."""
    stale = ModelConfig(
        name="stale",
        path=cfg.models[0].path,
        port=18081,
        gpu_pci_slot="0000:03:00.0",
        recipe={"ctx": 8192},
        tune_state="tuned",
        tuned_at=1234567890.0,
        tune_fingerprint="old",
    )
    cfg.models.append(stale)
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    tuner.bump_use("m")
    # stale was never used, m was just used.
    picked = tuner._pick_candidate()
    assert picked is not None
    assert picked.name == "m"


async def test_stop_terminates_the_loop_on_every_python(cfg: Config) -> None:
    """stop() must actually end the background loop, not just ask nicely.

    On Python < 3.12 `asyncio.wait_for` discards a CancelledError that arrives
    after its inner future has already resolved. stop() sets the abort event
    and cancels in the same tick, hitting that window every time, so the loop
    used to survive its own cancellation and `await self._task` never
    returned. In the server that showed up as lifespan shutdown hanging
    forever, i.e. every TestClient context manager in the suite deadlocking.
    """
    router = FakeRouter(cfg)
    # A long interval parks the loop inside the wait, which is where stop()
    # finds it in practice.
    tuner = Autotuner(cfg, router, version="0.6.0", loop_interval=60)
    await tuner.start()
    await asyncio.sleep(0.1)
    task = tuner._task
    assert task is not None and not task.done()

    # Deliberately not asyncio.wait_for: timing stop() out would *cancel* it,
    # and stop() swallows CancelledError, so wait_for would report success on
    # exactly the broken code this test exists to catch. asyncio.wait leaves
    # the coroutine alone and just tells us whether it finished.
    stopper = asyncio.ensure_future(tuner.stop())
    done, _pending = await asyncio.wait({stopper}, timeout=5)
    try:
        assert stopper in done, "stop() never returned: the loop outlived its own cancellation"
        assert task.done()
        assert tuner._task is None
    finally:
        # Awaited so a failing assertion doesn't leave pending tasks behind for
        # whichever unrelated test happens to run next.
        stopper.cancel()
        task.cancel()
        await asyncio.gather(stopper, task, return_exceptions=True)


async def test_deferred_restore_does_not_consume_abort_signal(cfg: Config) -> None:
    """The drain loop used to clear the shared _abort_event each lap, eating
    signals that belong to the outer loop and to abort_sweep() callers: an
    abort posted while a restore was draining silently vanished."""
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)

    import arc_llama.tune as tune_mod

    applied: list[dict[str, Any]] = []

    async def fake_apply_edits(client, name, edits):
        applied.append(dict(edits))
        return None

    tune_mod._apply_edits = fake_apply_edits

    router.inflight = 1

    async def drain_soon():
        # Post an abort while the restore is mid-drain, then let it finish.
        await asyncio.sleep(0.1)
        tuner._abort_event.set()
        await asyncio.sleep(0.1)
        router.inflight = 0

    # Reach the deferred-restore callback the way tune_model would.
    restore = None

    async def fake_tune_model(*args, **kwargs):
        nonlocal restore
        restore = kwargs.get("on_deferred_restore")
        return TuneReport(model="m", target="balanced", aborted=True)

    tune_mod.tune_model = fake_tune_model
    await tuner._run_sweep(cfg.models[0])
    assert restore is not None

    drainer = asyncio.create_task(drain_soon())
    await asyncio.wait_for(restore({"ubatch_size": 512}), timeout=5.0)
    await drainer

    assert tuner._abort_event.is_set(), (
        "the deferred restore consumed an abort signal it does not own"
    )
    assert applied, "restore never applied after the drain"


# ---------------------------------------------------------------------------
# #34 — start()/stop() serialization
# ---------------------------------------------------------------------------


async def test_start_is_noop_while_running(cfg: Config) -> None:
    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    await tuner.start()
    first = tuner._task
    await tuner.start()
    assert tuner._task is first
    await tuner.stop()


async def test_start_racing_stop_leaves_a_live_loop(cfg: Config) -> None:
    """#34: start() while stop() is still reaping the old task used to see
    the not-yet-done task, no-op, and leave the tuner permanently dead with
    _stopping set. Both take the lock now; start runs after stop finishes
    and must come up clean. The lock is pre-held so the ordering is
    deterministic: stop queues first, start behind it."""
    router = FakeRouter(cfg)
    tuner = Autotuner(cfg, router, version="0.6.0", loop_interval=60)
    await tuner.start()
    assert tuner.is_running

    await tuner._lock.acquire()
    stopper = asyncio.ensure_future(tuner.stop())
    starter = asyncio.ensure_future(tuner.start())
    await asyncio.sleep(0)  # both queued on the lock, stop first
    tuner._lock.release()
    await asyncio.gather(stopper, starter)

    assert tuner.is_running, "start behind an in-flight stop left no loop running"
    assert not tuner._stopping
    await tuner.stop()
    assert not tuner.is_running


async def test_completed_sweep_retains_measured_before_after(cfg, monkeypatch):
    from arc_llama.benchmark import BenchmarkResult

    router = FakeRouter(cfg)
    tuner = _make_tuner(cfg, router)
    before = BenchmarkResult("m", 8192, "f16", "f16", 64, 32, prompt_eval_tok_s=100, generation_tok_s=20)
    after = BenchmarkResult("m", 8192, "q8_0", "q8_0", 64, 32, prompt_eval_tok_s=120, generation_tok_s=30)

    async def measured_report(*args, **kwargs):
        return TuneReport(model="m", target="balanced", baseline=before, best=after, applied=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", measured_report)
    await tuner._run_sweep(cfg.models[0])
    result = tuner.last_results["m"]
    assert result["before"]["generation_tok_s"] == 20
    assert result["after"]["generation_tok_s"] == 30
    assert result["improvement_pct"]["generation"] == 50
    assert result["applied"] is True

    async def aborted_report(*args, **kwargs):
        return TuneReport(model="m", target="balanced", baseline=after, aborted=True)

    monkeypatch.setattr("arc_llama.tune.tune_model", aborted_report)
    await tuner._run_sweep(cfg.models[0])
    assert tuner.last_results["m"] == result
