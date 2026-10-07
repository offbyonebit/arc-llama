"""Eviction must drain an incumbent's in-flight requests before killing it.

`_evict_for` used to call astop() on a running incumbent immediately, so a
request for model B killed model A's llama-server while A was still streaming
a generation — A's client saw a mid-stream error for no reason it could act
on. The global `inflight` counter cannot gate this: the evicting request
itself holds it above zero, so waiting on it would deadlock. The router now
tracks per-model in-flight counts and drains the incumbent (bounded) before
stopping it.

No GPU or llama.cpp backend needed.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers import FakeServer
from helpers import fake_router as _router


async def test_eviction_waits_for_incumbent_to_finish(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")

    # A request is mid-generation on qwen.
    rt.acquire_model("qwen")

    async def finish_soon():
        await asyncio.sleep(0.3)
        rt.release_model("qwen")

    finisher = asyncio.create_task(finish_soon())
    # B arrives; eviction must not fire until qwen's request completes.
    await rt.ensure_active("gemma")
    await finisher

    assert FakeServer.stops == ["qwen"]
    # The drain outlasted the in-flight request: at stop time it was idle.
    assert rt.model_inflight.get("qwen", 0) == 0


async def test_eviction_does_not_wait_when_incumbent_is_idle(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")

    start = asyncio.get_event_loop().time()
    await rt.ensure_active("gemma")
    elapsed = asyncio.get_event_loop().time() - start

    assert FakeServer.stops == ["qwen"]
    assert elapsed < 1.0, f"idle eviction stalled for {elapsed:.1f}s"


async def test_eviction_proceeds_after_drain_timeout(tmp_path, monkeypatch):
    """Liveness: a generation that never ends must not block the new model
    forever. After the bounded drain the eviction goes ahead (and the stuck
    request's client sees the error it was always going to see)."""
    rt = _router(tmp_path, monkeypatch)
    rt.cfg.server.switch_interrupt_policy = "interrupt"
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")  # never released

    orig = rt._evict_for

    async def fast_drain(target, gpu, drain_seconds=30.0):
        return await orig(target, gpu, drain_seconds=0.3)

    monkeypatch.setattr(rt, "_evict_for", fast_drain)
    await rt.ensure_active("gemma")

    assert FakeServer.stops == ["qwen"], "timed-out drain still has to evict"


async def test_release_model_tolerates_unmatched_release(tmp_path, monkeypatch):
    """A double release must warn, not push the count negative — a negative
    count would make the incumbent look permanently busy or permanently idle
    depending on sign handling."""
    rt = _router(tmp_path, monkeypatch)
    rt.acquire_model("qwen")
    rt.release_model("qwen")
    rt.release_model("qwen")  # unmatched
    assert rt.model_inflight.get("qwen", 0) == 0
    rt.acquire_model("qwen")
    assert rt.model_inflight["qwen"] == 1


async def test_per_model_counts_are_independent(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch, single=False)
    rt.acquire_model("qwen")
    rt.acquire_model("qwen")
    rt.acquire_model("gemma")
    assert rt.model_inflight == {"qwen": 2, "gemma": 1}
    rt.release_model("qwen")
    assert rt.model_inflight == {"qwen": 1, "gemma": 1}


async def test_unconfirmed_shutdown_blocks_swap_and_same_model_retry(tmp_path, monkeypatch):
    import pytest

    from arc_llama.failures import StartupFailureError

    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")
    srv = rt._servers["qwen"]

    async def stuck_stop(*args, **kwargs):
        srv.ready = False
        raise StartupFailureError(
            "gpu_unavailable", "Child has not exited", "Wait for child exit",
            details={"reason": "shutdown_timeout"},
        )

    monkeypatch.setattr(srv, "astop", stuck_stop)
    for model in ("gemma", "qwen"):
        with pytest.raises(StartupFailureError, match="has not exited"):
            await rt.ensure_active(model)
        assert srv.is_running
        assert FakeServer.starts == ["qwen"]
        assert not rt._stopping
        assert not rt._loading_futures


@pytest.mark.parametrize("action", ["stop_one", "stop_all", "switch", "rebuild"])
async def test_exited_child_releases_retained_resources(tmp_path, monkeypatch, action):
    from unittest.mock import Mock

    from arc_llama.launcher import LlamaServer

    rt = _router(tmp_path, monkeypatch)
    previous = rt._servers["qwen"]
    srv = LlamaServer(previous.plan, "qwen")
    srv.process = Mock(pid=12345)
    srv.process.poll.return_value = -9
    lock = open(tmp_path / "retained.lock", "a+")
    srv._resident_lock = lock
    rt._servers["qwen"] = srv
    if action == "stop_one":
        assert await rt.stop_one("qwen") is False
    elif action == "stop_all":
        assert await rt.stop_all() == 0
    elif action == "switch":
        await rt.ensure_active("gemma")
    else:
        assert await rt.rebuild_model("qwen") == (True, False)
    assert srv.process is None and srv._resident_lock is None
    assert lock.closed
