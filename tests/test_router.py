from __future__ import annotations

import asyncio
import time

import pytest

from arc_llama.config import ModelConfig
from arc_llama.failures import StartupFailureError
from arc_llama.router import Router, estimate_model_vram_quick_mb


def test_quick_vram_estimate_uses_file_kv_and_fixed_overhead(tmp_path, monkeypatch):
    model_file = tmp_path / "dense.gguf"
    model_file.write_bytes(b"x" * (1_048_576 + 1))
    model = ModelConfig(
        name="dense",
        path=str(model_file),
        port=18080,
        gpu_pci_slot="gpu",
        recipe={"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
    )
    monkeypatch.setattr("arc_llama.router.kv_bytes_per_token_f16", lambda _path: 1024)

    # 2 MiB rounded-up weights + 4 MiB q8 KV + 768 MiB compute + 256 MiB safety.
    assert estimate_model_vram_quick_mb(model) == 1030


def test_quick_vram_estimate_defers_offloaded_models(tmp_path):
    model_file = tmp_path / "moe.gguf"
    model_file.write_bytes(b"GGUF")
    model = ModelConfig(
        name="moe",
        path=str(model_file),
        port=18080,
        gpu_pci_slot="gpu",
        recipe={"n_cpu_moe": 12},
    )

    assert estimate_model_vram_quick_mb(model) is None


class FakeServer:
    starts: list[str] = []
    stops: list[str] = []

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False
        self.ready = False

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.ready = False
        self.starts.append(self.name)

    async def wait_ready(self):
        self.ready = True
        return True

    def stop(self):
        self.running = False
        self.ready = False
        self.stops.append(self.name)

    async def astop(self, drain_seconds=3.0):
        # Mirrors LlamaServer.astop, which offloads the blocking stop() to a
        # thread. The router awaits this from the event loop.
        self.stop()


async def test_single_resident_policy_stops_other_models_before_starting_target(
    tmp_path, monkeypatch
):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=True)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")
    await rt.ensure_active("gemma")

    assert FakeServer.starts == ["qwen", "gemma"]
    assert FakeServer.stops == ["qwen"]


async def test_multi_resident_policy_keeps_models_on_different_gpus_running(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")
    await rt.ensure_active("gemma")

    assert FakeServer.starts == ["qwen", "gemma"]
    assert FakeServer.stops == []


async def test_vram_guard_refuses_oversized_model(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    monkeypatch.setattr(router_mod, "_estimate_model_vram_mb", lambda m: 999_999)
    rt = Router(cfg)

    with pytest.raises(RuntimeError, match="needs ~"):
        await rt.ensure_active("qwen")
    assert FakeServer.starts == []


def test_single_resident_vram_guard_ignores_incumbent_that_will_be_evicted(
    tmp_path, monkeypatch
):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=True)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)

    # Each model fits alone, but the pair does not. A single-resident swap
    # must admit the target because the incumbent is stopped first.
    sizes = {"qwen": 14_000, "gemma": 14_000}
    monkeypatch.setattr(router_mod, "_estimate_model_vram_mb", lambda model: sizes[model.name])
    rt = Router(cfg)

    cfg.models[1].gpu_pci_slot = cfg.models[0].gpu_pci_slot
    incumbent = rt._servers["qwen"]
    incumbent.running = True
    rt._check_vram_fit(cfg.models[1], cfg.gpus[0])


async def test_preflight_failure_does_not_evict_healthy_resident(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=True)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)
    await rt.ensure_active("qwen")

    def reject_gemma(model, _gpu, _plan):
        if model.name == "gemma":
            raise StartupFailureError(
                "model_missing", "Model file not found.", "Update the model path."
            )

    monkeypatch.setattr(router_mod, "preflight_launch", reject_gemma)
    with pytest.raises(StartupFailureError):
        await rt.ensure_active("gemma")

    assert rt._servers["qwen"].is_running
    assert FakeServer.stops == []


async def test_metrics_increment_on_load_and_stop(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")
    assert rt.metrics["loads"] == 1
    assert rt.metrics["load_errors"] == 0

    await rt.stop_one("qwen")
    assert rt.metrics["stops"] == 1


# ---------------------------------------------------------------------------
# Round 4: fast-path readiness race. A request arriving while llama-server is
# still loading must wait for readiness (bounded by the wait_ready budget) and
# then be served — never forwarded into a port that is not listening yet.
# ---------------------------------------------------------------------------


class SlowReadyServer:
    """Process reports is_running immediately but becomes ready only after a
    delay — mirrors a real cold start, where llama-server takes 20-30s to bind
    its port and pass /health."""

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False
        self.ready = False
        self.start_count = 0

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.ready = False
        self.start_count += 1

    async def wait_ready(self):
        await asyncio.sleep(0.2)
        self.ready = True
        return True

    def tail_log(self, lines=50):
        return ""

    def stop(self):
        self.running = False
        self.ready = False


class NeverReadyServer:
    """Starts, but the health check never passes (e.g. bad recipe flags)."""

    def __init__(self, plan, name):
        self.plan = plan
        self.name = name
        self.running = False
        self.ready = False
        self.start_count = 0

    @property
    def is_running(self):
        return self.running

    def start(self, log_dir=None):
        self.running = True
        self.ready = False
        self.start_count += 1

    async def wait_ready(self):
        await asyncio.sleep(0.05)
        return False

    def tail_log(self, lines=50):
        return "boom: failed to bind port"

    def stop(self):
        self.running = False
        self.ready = False


async def test_request_during_cold_start_waits_for_readiness(tmp_path, monkeypatch):
    """Regression test for the exact observed 500: the subprocess is alive but
    not yet listening, so the request must not complete until readiness."""
    from conftest import make_config

    import arc_llama.router as router_mod

    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", SlowReadyServer)
    rt = Router(cfg)

    starter = asyncio.create_task(rt.ensure_active("qwen"))
    await asyncio.sleep(0.05)  # starter is now inside wait_ready
    srv = rt._servers["qwen"]
    assert srv.is_running and not srv.ready

    waiter = asyncio.create_task(rt.ensure_active("qwen"))
    await asyncio.sleep(0.05)
    # The old fast path returned here on is_running alone; the caller would
    # then forward into a closed port and 500.
    assert not waiter.done()

    _, waited_srv = await waiter
    assert waited_srv.ready
    await starter


async def test_concurrent_cold_start_requests_share_one_process_start(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", SlowReadyServer)
    rt = Router(cfg)

    starter = asyncio.create_task(rt.ensure_active("qwen"))
    await asyncio.sleep(0.05)  # load in progress
    results = await asyncio.gather(
        rt.ensure_active("qwen"),
        rt.ensure_active("qwen"),
        starter,
    )

    srv = rt._servers["qwen"]
    assert srv.start_count == 1
    assert all(s.ready for _, s in results)


async def test_failed_load_raises_with_log_tail_and_waiter_fails_fast(tmp_path, monkeypatch):
    from conftest import make_config

    import arc_llama.router as router_mod

    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", NeverReadyServer)
    rt = Router(cfg)

    starter = asyncio.create_task(rt.ensure_active("qwen"))
    await asyncio.sleep(0.02)  # starter is inside wait_ready

    # A waiter arriving mid-load must get the same RuntimeError (which
    # _proxy_post turns into a 503), promptly — not hang for a fresh budget.
    t0 = time.monotonic()
    with pytest.raises(StartupFailureError, match="timed out while loading"):
        await rt.ensure_active("qwen")
    assert time.monotonic() - t0 < 5

    # The starter's structured error retains the log tail without placing it
    # in the concise public message, and waiters receive the same failure.
    with pytest.raises(StartupFailureError) as caught:
        await starter
    assert "failed to bind port" in caught.value.details["log_tail"]
    assert rt.metrics["load_errors"] >= 1


async def test_failed_load_without_waiter_does_not_leak_future_exception(tmp_path, monkeypatch):
    """A lone failed starter must not trigger asyncio's unhandled-future warning."""
    from conftest import make_config

    import arc_llama.router as router_mod

    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", NeverReadyServer)
    rt = Router(cfg)
    loop = asyncio.get_running_loop()
    contexts = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))
    try:
        with pytest.raises(StartupFailureError, match="timed out while loading"):
            await rt.ensure_active("qwen")
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not [
        context
        for context in contexts
        if context.get("message") == "Future exception was never retrieved"
    ]


async def test_warm_request_does_not_block_on_router_lock(tmp_path, monkeypatch):
    """The fast path exists so warm requests never contend on the swap lock.
    Hold the lock and assert a request to a ready model still completes."""
    from conftest import make_config

    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=False)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(cfg)

    await rt.ensure_active("qwen")  # warm: running and ready

    await rt._lock.acquire()
    try:
        _, srv = await asyncio.wait_for(rt.ensure_active("qwen"), timeout=1.0)
    finally:
        rt._lock.release()
    assert srv.ready


# ---------------------------------------------------------------------------
# Round 5 phase 1: the VRAM guard must account for --n-cpu-moe expert offload.
# ---------------------------------------------------------------------------


def _fake_scan(total_gib: float, per_layer_mib: int, n_layers: int):
    return (
        int(total_gib * 1024**3),
        {i: per_layer_mib * 1024**2 for i in range(n_layers)},
    )


def _moe_config(tmp_path, *, n_cpu_moe, ctx=8192, kv="q8_0"):
    from conftest import make_config

    cfg = make_config(tmp_path, single_resident=False)
    qwen = cfg.find_model("qwen")
    qwen.recipe = {"ctx": ctx, "cache_type_k": kv, "cache_type_v": kv}
    if n_cpu_moe is not None:
        qwen.recipe["n_cpu_moe"] = n_cpu_moe
    return cfg


async def test_vram_guard_admits_model_that_fits_only_with_offload(tmp_path, monkeypatch):
    """30 GiB model on a 24 GiB card with 4 layers of experts offloaded:
    full weights would be refused, the offloaded footprint fits."""
    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = _moe_config(tmp_path, n_cpu_moe=4)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    # 30 GiB total, 8 MoE layers of 2 GiB expert tensors each.
    scan = _fake_scan(30, 2048, 8)
    monkeypatch.setattr("arc_llama.gguf_meta.scan_weight_tensors", lambda _p: scan)
    rt = Router(cfg)

    model, srv = await rt.ensure_active("qwen")
    assert model.name == "qwen"
    assert FakeServer.starts == ["qwen"]


async def test_vram_guard_refuses_when_even_max_offload_does_not_fit(tmp_path, monkeypatch):
    """50 GiB model: even with every MoE layer's experts on the host the
    remainder does not fit, so the guard must still refuse."""
    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = _moe_config(tmp_path, n_cpu_moe=8)  # all 8 layers offloaded
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    scan = _fake_scan(50, 2048, 8)
    monkeypatch.setattr("arc_llama.gguf_meta.scan_weight_tensors", lambda _p: scan)
    rt = Router(cfg)

    with pytest.raises(RuntimeError, match="needs ~"):
        await rt.ensure_active("qwen")
    assert FakeServer.starts == []


async def test_vram_guard_skips_when_offload_bytes_unknown(tmp_path, monkeypatch):
    """Offload is set but the expert tensor bytes cannot be determined. The
    guard must skip (permit), not fall back to counting full weights — the
    fallback would refuse exactly the models offload exists to rescue."""
    import arc_llama.router as router_mod

    FakeServer.starts = []
    FakeServer.stops = []
    cfg = _moe_config(tmp_path, n_cpu_moe=4)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    # Unreadable GGUF: the scan yields nothing. A file-size fallback would
    # see 26 GiB > 24 GiB and refuse — the bug this fixes.
    monkeypatch.setattr("arc_llama.gguf_meta.scan_weight_tensors", lambda _p: None)
    model_path = tmp_path / "models" / "Qwen3-7B-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        f.truncate(26 * 1024**3)  # sparse: st_size without the disk usage
    rt = Router(cfg)

    model, srv = await rt.ensure_active("qwen")
    assert model.name == "qwen"
    assert FakeServer.starts == ["qwen"]


async def test_non_moe_model_footprint_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """Dense models — with or without an (inert) n_cpu_moe in the recipe —
    keep the exact pre-offload accounting."""
    from arc_llama.recipes import KVCacheType, estimate_kv_bytes
    from arc_llama.router import (
        _VRAM_COMPUTE_BUFFER_MB,
        _VRAM_SAFETY_MARGIN_MB,
        _estimate_model_vram_mb,
    )

    dense = _moe_config(tmp_path, n_cpu_moe=None).find_model("qwen")
    flagged = _moe_config(tmp_path, n_cpu_moe=4).find_model("qwen")
    # Dense tensor table: no routed-expert tensors at all.
    monkeypatch.setattr(
        "arc_llama.gguf_meta.scan_weight_tensors",
        lambda _p: (8 * 1024**3, {}),
    )
    expected = (
        8 * 1024
        + estimate_kv_bytes(8192, KVCacheType.Q8_0, dense.kv_class) // (1_048_576)
        + _VRAM_COMPUTE_BUFFER_MB
        + _VRAM_SAFETY_MARGIN_MB
    )
    assert _estimate_model_vram_mb(dense) == expected
    assert _estimate_model_vram_mb(dense, n_cpu_moe=0) == expected
    assert _estimate_model_vram_mb(flagged) == expected
    assert _estimate_model_vram_mb(flagged, n_cpu_moe=4) == expected


def test_min_moe_offload_layers_against_ctx_and_kv(tmp_path, monkeypatch):
    """Minimum feasible N from the estimator, per (ctx, KV type): q8_0 KV is
    half the bytes of f16, so the minimum must move with the KV choice."""
    from arc_llama.recipes import KVCacheType
    from arc_llama.router import min_moe_offload_layers

    model = _moe_config(tmp_path, n_cpu_moe=None).find_model("qwen")
    # 28 GiB total, 16 MoE layers of 1 GiB expert tensors each.
    scan = _fake_scan(28, 1024, 16)
    monkeypatch.setattr("arc_llama.gguf_meta.scan_weight_tensors", lambda _p: scan)
    monkeypatch.setattr("arc_llama.router.scan_weight_tensors", lambda _p: scan)

    vram = 25000
    assert min_moe_offload_layers(model, vram, ctx=8192, kv_type=KVCacheType.F16) == 6
    assert min_moe_offload_layers(model, vram, ctx=8192, kv_type=KVCacheType.Q8_0) == 5
    # Larger context consumes more KV: the minimum must move with ctx too.
    assert min_moe_offload_layers(model, vram, ctx=8192, kv_type=KVCacheType.Q8_0) < (
        min_moe_offload_layers(model, vram, ctx=65536, kv_type=KVCacheType.Q8_0)
    )


def test_min_moe_offload_layers_zero_and_unknown(tmp_path, monkeypatch):
    from arc_llama.recipes import KVCacheType
    from arc_llama.router import min_moe_offload_layers

    model = _moe_config(tmp_path, n_cpu_moe=None).find_model("qwen")
    # Small model: fits with no offload at all.
    scan = _fake_scan(4, 512, 4)
    monkeypatch.setattr("arc_llama.router.scan_weight_tensors", lambda _p: scan)
    assert min_moe_offload_layers(model, 24576, ctx=8192, kv_type=KVCacheType.Q8_0) == 0
    # Unknown expert bytes or unknown VRAM budget: no offload math possible.
    monkeypatch.setattr("arc_llama.router.scan_weight_tensors", lambda _p: None)
    assert min_moe_offload_layers(model, 24576) is None
    assert min_moe_offload_layers(model, None) is None
