"""Model-switch safety regression tests.

The switch path must be safe by default even under concurrent arrivals:

1. The incumbent is marked as draining BEFORE the first await, so a request
   arriving through the lock-free fast path during the drain can never
   acquire the draining model and restart the teardown clock.
2. The drain is bounded and configurable (``server.switch_drain_seconds``).
3. The interrupt policy is explicit (``server.switch_interrupt_policy``):
   the default stops the incumbent after the bounded wait; ``reject_new``
   refuses the switch with an actionable 409 and leaves the incumbent alive.
4. Draining state is always cleaned up on failure and cancellation — a
   stale mark would permanently lock a model out of the fast path.

No GPU, no real llama-server: FakeServer doubles stand in throughout.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import make_config
from test_router import FakeServer

import arc_llama.router as router_mod
from arc_llama.config import ServerConfig
from arc_llama.failures import StartupFailureError
from arc_llama.router import Router


def _router(tmp_path, monkeypatch, *, single=True, **server_kwargs) -> Router:
    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=single)
    for key, value in server_kwargs.items():
        setattr(cfg.server, key, value)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    return Router(cfg)


async def test_draining_incumbent_rejects_fast_path_arrivals(tmp_path, monkeypatch):
    """A request arriving for the incumbent mid-drain must NOT acquire it
    while the drain is in progress.

    The incumbent is marked in _stopping before the drain's first await, so
    the lock-free ready-check in ensure_active refuses to hand the draining
    model out; the arrival must land on the slow path behind the swap lock
    the switcher already holds.
    """
    rt = _router(tmp_path, monkeypatch, switch_drain_seconds=2.0)
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")  # a generation is running and won't finish

    switch = asyncio.create_task(rt.ensure_active("gemma"))
    # Let the switcher take the lock and enter the drain sleep.
    await asyncio.sleep(0.15)
    assert "qwen" in rt._stopping, "incumbent must be marked draining before awaiting"
    assert not switch.done()

    late = asyncio.create_task(rt.ensure_active("qwen"))
    await asyncio.sleep(0.15)
    # The late arrival must not have completed against the draining model
    # during the drain: acquiring it would restart the teardown clock and
    # extend the bounded wait beyond its deadline.
    assert not late.done()
    late.cancel()
    try:
        await late
    except asyncio.CancelledError:
        pass
    rt.release_model("qwen")
    # With the in-flight request gone the drain completes and the switch
    # proceeds normally.
    await switch
    assert FakeServer.stops == ["qwen"]


async def test_concurrent_arrivals_during_drain_do_not_prolong_it(tmp_path, monkeypatch):
    """Fast-path arrivals during the drain never extend the bounded wait.

    The drain deadline is fixed when it starts; arrivals for the incumbent
    cannot restart it because they cannot acquire the draining model.
    """
    rt = _router(tmp_path, monkeypatch, switch_drain_seconds=0.4)
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")  # never released: drain must expire

    t0 = time.monotonic()
    with pytest.raises(StartupFailureError):
        # Default policy rejects the switch after the bounded drain.
        await rt.ensure_active("gemma")
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"drain outlasted its bound: {elapsed:.2f}s"
    assert FakeServer.stops == []


async def test_switch_timeout_interrupt_policy_stops_incumbent(tmp_path, monkeypatch):
    """interrupt: after the bounded drain the incumbent is stopped anyway."""
    rt = _router(
        tmp_path, monkeypatch,
        switch_drain_seconds=0.2,
        switch_interrupt_policy="interrupt",
    )
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")

    await rt.ensure_active("gemma")

    assert FakeServer.stops == ["qwen"]
    assert "qwen" not in rt._stopping


async def test_switch_timeout_reject_new_policy_keeps_incumbent(tmp_path, monkeypatch):
    """reject_new: after the bounded drain the switch fails with an action.

    The incumbent stays ready and keeps serving; the error names both models
    and points at the two knobs that change the outcome.
    """
    rt = _router(
        tmp_path, monkeypatch,
        switch_drain_seconds=0.2,
        switch_interrupt_policy="reject_new",
    )
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")  # active generation, never finishes

    with pytest.raises(StartupFailureError) as caught:
        await rt.ensure_active("gemma")

    failure = caught.value
    assert failure.category == "switch_busy"
    assert failure.http_status == 409
    assert "gemma" in failure.message
    assert "qwen" in failure.message
    assert failure.details["busy_model"] == "qwen"
    assert failure.details["policy"] == "reject_new"
    # The incumbent survived, is still ready, and is no longer marked.
    assert FakeServer.starts == ["qwen"]
    assert FakeServer.stops == []
    assert rt._servers["qwen"].is_running
    assert rt._servers["qwen"].ready
    assert "qwen" not in rt._stopping
    # ...and it is acquirable again through the fast path.
    model, srv = await rt.ensure_active("qwen", acquire=True)
    assert srv.ready


async def test_switch_busy_error_carries_config_knobs(tmp_path, monkeypatch):
    rt = _router(
        tmp_path, monkeypatch,
        switch_drain_seconds=1.25,
        switch_interrupt_policy="reject_new",
    )
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")

    with pytest.raises(StartupFailureError) as caught:
        await rt.ensure_active("gemma")

    details = caught.value.details
    assert details["drain_seconds"] == 1.25
    assert details["inflight"] == 1
    assert details["model"] == "gemma"


async def test_drain_succeeds_when_generation_finishes_in_time(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch, switch_interrupt_policy="reject_new")
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")

    async def finish_soon():
        await asyncio.sleep(0.2)
        rt.release_model("qwen")

    finisher = asyncio.create_task(finish_soon())
    await rt.ensure_active("gemma")
    await finisher

    assert FakeServer.stops == ["qwen"]
    assert "qwen" not in rt._stopping


async def test_draining_mark_cleaned_when_stop_raises(tmp_path, monkeypatch):
    """A failing teardown must never leave the model marked as draining."""
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")

    async def boom(drain_seconds=3.0):
        raise RuntimeError("teardown exploded")

    srv = rt._servers["qwen"]
    monkeypatch.setattr(srv, "astop", boom)

    with pytest.raises(RuntimeError, match="teardown exploded"):
        await rt.ensure_active("gemma")

    assert "qwen" not in rt._stopping, "stale draining mark would wedge the fast path"


async def test_draining_mark_cleaned_on_cancellation(tmp_path, monkeypatch):
    """Cancelling the switcher mid-drain clears the draining mark."""
    rt = _router(tmp_path, monkeypatch, switch_drain_seconds=5.0)
    await rt.ensure_active("qwen")
    rt.acquire_model("qwen")  # keeps the drain waiting

    switch = asyncio.create_task(rt.ensure_active("gemma"))
    await asyncio.sleep(0.2)
    assert "qwen" in rt._stopping
    switch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await switch

    assert "qwen" not in rt._stopping
    # The incumbent was never stopped and remains servable.
    rt.release_model("qwen")
    model, srv = await rt.ensure_active("qwen")
    assert srv.is_running


def test_switch_config_validated_at_construction():
    with pytest.raises(ValueError, match="switch_interrupt_policy"):
        ServerConfig(switch_interrupt_policy="bogus")
    with pytest.raises(ValueError, match="switch_drain_seconds"):
        ServerConfig(switch_drain_seconds=0)
    # The documented values all construct cleanly.
    for policy in ("interrupt", "prefer_new_request", "reject_new"):
        ServerConfig(switch_interrupt_policy=policy)


def test_config_defaults_preserve_historical_behaviour():
    # The default drain matches the old hard-coded 30s, and the default
    # policy is the safe one: an active response is never killed.
    cfg = ServerConfig()
    assert cfg.switch_drain_seconds == 30.0
    assert cfg.switch_interrupt_policy == "reject_new"


def test_migrated_config_gains_new_server_keys(tmp_path):
    from arc_llama.config import load_config

    config_dir = tmp_path / "arc-llama"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(
        "[server]\nhost = \"127.0.0.1\"\nport = 11437\nsingle_resident = true\n"
    )

    cfg = load_config(config_path)
    assert cfg.server.switch_drain_seconds == 30.0
    assert cfg.server.switch_interrupt_policy == "reject_new"
