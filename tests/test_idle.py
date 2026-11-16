from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from helpers import FakeServer, fake_router

from arc_llama.config import load_config
from arc_llama.idle import hold_model
from arc_llama.server import _idle_busy


@pytest.fixture
def idle_router(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("arc_llama.router.time.time", lambda: clock[0])
    rt = fake_router(tmp_path, monkeypatch)
    rt.cfg.models[0].idle_unload_minutes = 2
    return rt, clock


async def test_idle_timeout_reuses_stop_and_next_request_reloads(idle_router, caplog):
    rt, clock = idle_router
    caplog.set_level("INFO", logger="arc_llama.router")
    await rt.ensure_active("qwen")
    clock[0] += 119
    assert await rt.unload_idle(lambda: False) == 0
    clock[0] += 1
    assert await rt.unload_idle(lambda: False) == 1
    assert FakeServer.stops == ["qwen"]
    assert rt.metrics["stops"] == 1
    await rt.ensure_active("qwen")
    assert FakeServer.starts == ["qwen", "qwen"]
    assert "idle-unloaded qwen" in caplog.text
    assert "reloaded qwen after idle unload" in caplog.text


async def test_inflight_and_completion_reset_the_idle_clock(idle_router):
    rt, clock = idle_router
    await rt.ensure_active("qwen", acquire=True)
    rt.inflight = 1
    clock[0] += 1000
    assert await rt.unload_idle(lambda: False) == 0
    rt.inflight = 0
    rt.release_model("qwen")
    assert await rt.unload_idle(lambda: False) == 0
    clock[0] += 120
    assert await rt.unload_idle(lambda: False) == 1


async def test_idle_off_keeps_model_loaded(idle_router):
    rt, clock = idle_router
    rt.cfg.models[0].idle_unload_minutes = 0
    await rt.ensure_active("qwen")
    clock[0] += 100000
    assert await rt.unload_idle(lambda: False) == 0


@pytest.mark.parametrize("activity", ["tune", "trial", "benchmark", "agent", "work"])
async def test_idle_skips_whole_protected_operations(idle_router, activity, monkeypatch):
    rt, clock = idle_router
    await rt.ensure_active("qwen")
    clock[0] += 1000
    state = SimpleNamespace(
        idle_holds={}, cfg=rt.cfg, tuner=None, work=None, active_harness_runs={},
    )
    if activity == "tune":
        state.tuner = SimpleNamespace(is_sweep_running=True)
    elif activity == "trial":
        rt.cfg.model_saved_settings["qwen"] = {}
    elif activity == "benchmark":
        state.idle_holds["bench"] = ("qwen", 200.0)
    elif activity == "agent":
        state.active_harness_runs["builtin"] = 1
    else:
        state.work = SimpleNamespace(active="job")
    app = SimpleNamespace(state=state)
    assert await rt.unload_idle(lambda: _idle_busy(app, now=100.0)) == 0
    assert rt.running_models() == ["qwen"]


async def test_idle_rechecks_after_waiting_for_swap_lock(idle_router):
    rt, clock = idle_router
    await rt.ensure_active("qwen")
    clock[0] += 120
    await rt._lock.acquire()
    pending = asyncio.create_task(rt.unload_idle(lambda: False))
    # Yield to the queued task without advancing real or fake time.
    await asyncio.sleep(0)
    rt.inflight = 1
    rt._lock.release()
    assert await pending == 0


def test_idle_setting_roundtrip_and_default(idle_router, tmp_path):
    rt, _ = idle_router
    path = tmp_path / "config.toml"
    assert rt.cfg.models[1].idle_unload_minutes == 0
    rt.cfg.save(path)
    assert load_config(path).models[0].idle_unload_minutes == 2


async def test_operation_hold_releases_on_error_and_off_avoids_requests():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"id": "operation"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test") as client:
        async with hold_model(client, "qwen", enabled=False):
            assert not calls
        with pytest.raises(ValueError):
            async with hold_model(client, "qwen", enabled=True):
                raise ValueError("measurement failed")
    assert len(calls) == 2
    assert b'"release":true' in calls[-1].content


async def test_depth_benchmark_holds_between_measurements():
    import json

    from arc_llama.benchmark import benchmark_depths

    active = False
    measurements = []

    def respond(request):
        nonlocal active
        if request.url.path.startswith("/admin/idle-hold/"):
            active = not json.loads(request.content).get("release", False)
            return httpx.Response(200, json={"id": "benchmark"})
        assert active
        measurements.append(request.url.path)
        return httpx.Response(200, json={"timings": {"predicted_per_second": 1.0}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test") as client:
        rows = await benchmark_depths("http://test", "qwen", [0, 512], client=client, idle_unload_minutes=1)
    assert len(rows) == 2 and not any(row.error for row in rows)
    assert len(measurements) == 3 and not active
