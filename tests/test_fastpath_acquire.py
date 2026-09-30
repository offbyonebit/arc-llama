"""Lock-free fast-path acquire must be atomic with eviction/rebuild teardown.

``ensure_active`` hands out an already-running server without taking the swap
lock. Two races lived in that gap:

  * #29 — the request acquired the model only AFTER ensure_active returned,
    so an eviction drain could read ``model_inflight[name] == 0`` in the
    window between the readiness check and the counter bump, and stop the
    server out from under the forward. The acquire now happens inside
    ensure_active, in the same synchronous segment as the check.
  * The final drain-read-to-astop window is closed by ``_stopping``: the
    evictor marks the model before the first await of teardown, and the fast
    path refuses to hand out a marked server (#29), with ``rebuild_model``
    getting the same drain + mark treatment (#31).

No GPU or llama.cpp backend needed.
"""

from __future__ import annotations

import asyncio

from conftest import make_config
from test_router import FakeServer

import arc_llama.router as router_mod
from arc_llama.router import Router


def _router(tmp_path, monkeypatch, *, single=True) -> Router:
    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path, single_resident=single)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)

    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(router_mod.asyncio, "to_thread", inline_to_thread)
    return Router(cfg)


async def test_fast_path_acquire_counts_atomically(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")

    # Warm fast path: acquire must be part of the call, not a follow-up.
    model, srv = await rt.ensure_active("qwen", acquire=True)
    assert rt.model_inflight.get("qwen") == 1

    # An eviction starting now must see the in-flight request and drain it.
    async def finish_soon():
        await asyncio.sleep(0.3)
        rt.release_model("qwen")

    finisher = asyncio.create_task(finish_soon())
    start = asyncio.get_event_loop().time()
    await rt.ensure_active("gemma")
    elapsed = asyncio.get_event_loop().time() - start
    await finisher

    assert FakeServer.stops == ["qwen"]
    assert elapsed >= 0.25, f"eviction did not drain the fast-path request ({elapsed:.2f}s)"


async def test_slow_path_load_acquire_counts(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen", acquire=True)
    assert rt.model_inflight.get("qwen") == 1


async def test_fast_path_never_hands_out_a_stopping_server(tmp_path, monkeypatch):
    """While an eviction's astop is in flight, a request for the evicted model
    must not receive the dying server. It waits (slow path, on the lock) and
    ends up with a fresh instance after the swap settles."""
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")
    old_srv = rt._servers["qwen"]

    astop_blocker = asyncio.Event()
    orig_astop = old_srv.astop

    async def blocked_astop(*a, **kw):
        await astop_blocker.wait()
        await orig_astop(*a, **kw)

    old_srv.astop = blocked_astop  # type: ignore[method-assign]

    evictor = asyncio.create_task(rt.ensure_active("gemma"))
    # Let the evictor reach the blocked astop.
    for _ in range(100):
        await asyncio.sleep(0.01)
        if "qwen" in rt._stopping:
            break
    assert "qwen" in rt._stopping, "evictor never marked qwen as stopping"

    waiter = asyncio.create_task(rt.ensure_active("qwen", acquire=True))
    await asyncio.sleep(0.1)
    assert not waiter.done(), "fast path handed out a server mid-teardown"

    astop_blocker.set()
    await evictor
    model, srv = await waiter

    assert srv.is_running
    assert srv is old_srv  # eviction reuses the entry; it was stopped, then restarted
    assert FakeServer.starts == ["qwen", "gemma", "qwen"]
    assert rt.model_inflight.get("qwen") == 1
    rt.release_model("qwen")


async def test_rebuild_drains_inflight_requests(tmp_path, monkeypatch):
    """#31: the deferred autotune restore POSTs an edit that rebuilds the
    model entry. A request that slipped in just before must be drained, not
    killed mid-generation."""
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen", acquire=True)

    async def finish_soon():
        await asyncio.sleep(0.3)
        rt.release_model("qwen")

    finisher = asyncio.create_task(finish_soon())
    start = asyncio.get_event_loop().time()
    rebuilt, was_running = await rt.rebuild_model("qwen", drain_seconds=5.0)
    elapsed = asyncio.get_event_loop().time() - start
    await finisher

    assert rebuilt and was_running
    assert elapsed >= 0.25, f"rebuild did not drain ({elapsed:.2f}s)"
    assert FakeServer.stops == ["qwen"]
    # The rebuilt entry is fresh and not running; the next request cold-starts.
    assert not rt._servers["qwen"].is_running


async def test_rebuild_proceeds_after_drain_timeout(tmp_path, monkeypatch):
    """Liveness: a stuck generation must not block the restore forever."""
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen", acquire=True)  # never released

    rebuilt, was_running = await rt.rebuild_model("qwen", drain_seconds=0.2)

    assert rebuilt and was_running
    assert FakeServer.stops == ["qwen"]


async def test_rebuild_marks_stopping_during_teardown(tmp_path, monkeypatch):
    rt = _router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")
    old_srv = rt._servers["qwen"]

    astop_blocker = asyncio.Event()
    orig_astop = old_srv.astop

    async def blocked_astop(*a, **kw):
        await astop_blocker.wait()
        await orig_astop(*a, **kw)

    old_srv.astop = blocked_astop  # type: ignore[method-assign]

    rebuilder = asyncio.create_task(rt.rebuild_model("qwen"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if "qwen" in rt._stopping:
            break
    assert "qwen" in rt._stopping, "rebuild never marked qwen as stopping"

    astop_blocker.set()
    rebuilt, was_running = await rebuilder
    assert rebuilt and was_running
    assert "qwen" not in rt._stopping


async def test_running_models_snapshot(tmp_path, monkeypatch):
    """#33: the lock-free helper the autotuner now consults."""
    rt = _router(tmp_path, monkeypatch)
    assert rt.running_models() == []
    await rt.ensure_active("qwen")
    assert rt.running_models() == ["qwen"]
    await rt.stop_all()
    assert rt.running_models() == []
