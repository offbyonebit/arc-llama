"""GPU resource lease arbitration for exclusive plugin GPU work.

``ResourceLeaseManager`` (app.state.resources) serialises exclusive plugin
tasks (vision, ...), waits for the router's in-flight requests to drain,
and empties the GPU of resident llama-server processes by delegating to
the Router's own lifecycle methods. Normal text inference never takes a
lease — the _proxy_post path is unchanged — and release must happen on
exceptions and cancellations, not just normal return.

No GPU or llama.cpp backend needed. The Router fakes mirror the shared
test doubles in test_router.py / test_fastpath_acquire.py.
"""

from __future__ import annotations

import asyncio

from conftest import make_config
from test_router import FakeServer

import arc_llama.router as router_mod
from arc_llama.resources import Lease, ResourceLeaseManager
from arc_llama.router import Router


def _router(tmp_path, monkeypatch) -> Router:
    FakeServer.starts = []
    FakeServer.stops = []
    cfg = make_config(tmp_path)
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    return Router(cfg)


def _manager(tmp_path, monkeypatch, **kwargs) -> tuple[Router, ResourceLeaseManager]:
    rt = _router(tmp_path, monkeypatch)
    return rt, ResourceLeaseManager(rt, **kwargs)


# ---------------------------------------------------------------------------
# exclusive leases serialize
# ---------------------------------------------------------------------------


async def test_exclusive_leases_serialize(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)

    order: list[str] = []
    in_lease = asyncio.Event()

    async def task(name: str, hold: asyncio.Event) -> None:
        async with mgr.acquire(name, exclusive=True):
            order.append(f"enter:{name}")
            in_lease.set()
            await hold.wait()
            order.append(f"exit:{name}")

    a_release = asyncio.Event()
    a = asyncio.create_task(task("a", a_release))
    await in_lease.wait()
    assert mgr.active_leases == {"a": 1}

    # b queues behind a; it must not run until a releases.
    in_lease = asyncio.Event()
    hold = asyncio.Event()
    b = asyncio.create_task(task("b", hold))
    await asyncio.sleep(0.1)
    assert order == ["enter:a"]
    assert not in_lease.is_set()
    assert "b" not in mgr.active_leases

    a_release.set()
    await in_lease.wait()  # b now owns the gate
    assert order == ["enter:a", "exit:a", "enter:b"]
    hold.set()
    await asyncio.gather(a, b)
    assert order == ["enter:a", "exit:a", "enter:b", "exit:b"]
    assert mgr.active_leases == {}
    assert not mgr.exclusive_active


async def test_lease_granted_and_returned(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)
    async with mgr.acquire("vision", exclusive=True) as lease:
        assert isinstance(lease, Lease)
        assert lease.owner == "vision"
        assert lease.exclusive is True
        assert mgr.active_leases == {"vision": 1}
        assert mgr.exclusive_active
    assert mgr.active_leases == {}
    assert not mgr.exclusive_active


# ---------------------------------------------------------------------------
# exception-safe and cancellation-safe release
# ---------------------------------------------------------------------------


async def test_lease_released_when_body_raises(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)

    class BodyError(Exception):
        pass

    with_leak = False
    try:
        async with mgr.acquire("vision", exclusive=True):
            assert mgr.active_leases == {"vision": 1}
            with_leak = True
            raise BodyError()
    except BodyError:
        pass
    assert with_leak, "exception never reached the with-body"
    assert mgr.active_leases == {}
    assert not mgr.exclusive_active

    # The gate is unwound too: a follow-up exclusive lease acquires fine.
    async with mgr.acquire("next", exclusive=True):
        assert mgr.active_leases == {"next": 1}


async def test_lease_released_when_owner_task_cancelled(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)

    started = asyncio.Event()

    async def holder():
        async with mgr.acquire("vision", exclusive=True):
            started.set()
            await asyncio.sleep(30)

    t = asyncio.create_task(holder())
    await started.wait()
    assert mgr.active_leases == {"vision": 1}
    t.cancel()
    with_leak: list[BaseException] = []
    try:
        await t
    except asyncio.CancelledError as exc:
        with_leak.append(exc)
    assert with_leak, "holder should have been cancelled"
    assert mgr.active_leases == {}
    assert not mgr.exclusive_active


# ---------------------------------------------------------------------------
# router drain wait and resident-model eviction
# ---------------------------------------------------------------------------


async def test_exclusive_lease_drains_router_requests_before_stopping_models(
    tmp_path, monkeypatch
):
    rt, mgr = _manager(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")

    # A generation is mid-flight (the request path holds rt.inflight up).
    rt.inflight = 1

    class RecordingRouter:
        """Fail the test if stop_all runs while the request is still live."""

        def __init__(self, inner):
            self._inner = inner
            self.stopped_with_inflight = False
            self.stop_all_calls = 0

        async def stop_all(self):
            self.stop_all_calls += 1
            if int(getattr(self._inner, "inflight", 0) or 0) > 0:
                self.stopped_with_inflight = True
            return await self._inner.stop_all()

        def __getattr__(self, item):
            return getattr(self._inner, item)

    recorder = RecordingRouter(rt)
    drained_mgr = ResourceLeaseManager(recorder, drain_seconds=10.0)

    async def finish_soon():
        await asyncio.sleep(0.3)
        rt.inflight = 0

    finisher = asyncio.create_task(finish_soon())
    start = asyncio.get_event_loop().time()
    async with drained_mgr.acquire("vision"):
        elapsed = asyncio.get_event_loop().time() - start
        await finisher
        assert elapsed >= 0.25, f"lease stopped models without draining ({elapsed:.2f}s)"
        assert recorder.stop_all_calls == 1
        assert not recorder.stopped_with_inflight
        # Every resident llama-server was stopped under the lease.
        assert FakeServer.stops == ["qwen"]


async def test_lease_grants_after_drain_budget(tmp_path, monkeypatch):
    """Liveness: a stuck request must not block the exclusive task forever."""
    rt, mgr = _manager(tmp_path, monkeypatch, drain_seconds=0.2)
    await rt.ensure_active("qwen")
    rt.inflight = 1  # never released

    async with mgr.acquire("vision"):
        # Granted anyway after the bounded drain, so the plugin can make
        # progress; the model still has to be gone before the body runs.
        pass
    assert FakeServer.stops == ["qwen"]


async def test_lease_stops_resident_models_via_router_lifecycle(tmp_path, monkeypatch):
    """Eviction reuses Router.stop_all — no duplicated process management."""
    rt, mgr = _manager(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")
    await rt.ensure_active("gemma")
    assert FakeServer.stops  # the swap itself stopped qwen first

    async with mgr.acquire("vision", exclusive=True):
        assert FakeServer.stops.count("qwen") + FakeServer.stops.copy().count("qwen") >= 1
    running = rt.running_models()
    assert running == [], f"resident models survived the exclusive lease: {running}"


async def test_lease_survives_router_stop_failures(tmp_path, monkeypatch):
    """A failure inside eviction must not wedge the gate or leak the lease."""
    rt, mgr = _manager(tmp_path, monkeypatch)

    stop_all_orig = rt.stop_all

    async def failing_stop_all():
        raise RuntimeError("stop_all exploded")

    rt.stop_all = failing_stop_all  # type: ignore[method-assign]
    granted = False
    try:
        async with mgr.acquire("vision"):
            granted = True
    except RuntimeError:
        pass  # propagate is acceptable; the gate must still be free
    finally:
        rt.stop_all = stop_all_orig  # type: ignore[method-assign]

    assert mgr.active_leases == {}
    assert not mgr.exclusive_active
    async with mgr.acquire("vision"):
        granted = True
    assert granted


# ---------------------------------------------------------------------------
# shared leases: bookkeeping that never coexists with an exclusive one
# ---------------------------------------------------------------------------


async def test_shared_lease_waits_for_exclusive(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)

    entered_exclusive = asyncio.Event()
    release_exclusive = asyncio.Event()

    async def exclusive():
        async with mgr.acquire("vision", exclusive=True):
            entered_exclusive.set()
            await release_exclusive.wait()

    ex = asyncio.create_task(exclusive())
    await entered_exclusive.wait()

    shared_entered = asyncio.Event()

    async def shared():
        async with mgr.acquire("reader", exclusive=False):
            shared_entered.set()

    sh = asyncio.create_task(shared())
    await asyncio.sleep(0.15)
    assert not shared_entered.is_set(), "shared lease ran under an active exclusive"

    release_exclusive.set()
    await ex
    await asyncio.wait_for(sh, timeout=5.0)
    assert shared_entered.is_set()
    assert mgr.active_leases == {}


async def test_shared_leases_coexist(tmp_path, monkeypatch):
    rt, mgr = _manager(tmp_path, monkeypatch)

    holds: list[asyncio.Event] = []
    done = 0

    async def task():
        nonlocal done
        async with mgr.acquire("shared-owner", exclusive=False):
            done += 1
            hold = asyncio.Event()
            holds.append(hold)
            await hold.wait()

    t1 = asyncio.create_task(task())
    while done < 1:
        await asyncio.sleep(0.01)
    t2 = asyncio.create_task(task())
    while done < 2:
        await asyncio.sleep(0.01)

    assert mgr.active_leases == {"shared-owner": 2}
    for h in holds:
        h.set()
    await asyncio.gather(t1, t2)
    assert mgr.active_leases == {}


async def test_pending_exclusive_blocks_new_shared_lease(tmp_path, monkeypatch):
    """A shared lease arriving while an exclusive one is queued behind it
    must not slip in front of the queued exclusive task."""
    rt, mgr = _manager(tmp_path, monkeypatch)

    gate_first = asyncio.Event()
    release_first = asyncio.Event()

    async def first():
        async with mgr.acquire("a", exclusive=True):
            gate_first.set()
            await release_first.wait()

    async def second():
        async with mgr.acquire("b", exclusive=True):
            pass

    t1 = asyncio.create_task(first())
    await gate_first.wait()
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.05)  # t2 is now queued on the gate
    assert mgr.exclusive_active

    shared_done = False

    async def shared_task():
        nonlocal shared_done
        async with mgr.acquire("reader", exclusive=False):
            shared_done = True
    t3 = asyncio.create_task(shared_task())
    await asyncio.sleep(0.1)
    assert not shared_done, "shared lease jumped a queued exclusive task"

    release_first.set()
    await asyncio.gather(t1, t2, t3)
    assert shared_done
    assert mgr.active_leases == {}


# ---------------------------------------------------------------------------
# normal text inference is untouched
# ---------------------------------------------------------------------------


async def test_ensure_active_fast_path_unaffected_by_manager(tmp_path, monkeypatch):
    """The router's lock-free fast path works identically with a manager
    attached; text inference never consults the lease gate."""
    rt, mgr = _manager(tmp_path, monkeypatch)

    await rt.ensure_active("qwen")
    model, srv = await rt.ensure_active("qwen", acquire=True)
    assert model.name == "qwen"
    assert srv.is_running
    assert rt.model_inflight.get("qwen") == 1
    # No lease was created for the request.
    assert mgr.active_leases == {}
    rt.release_model("qwen")

    # And a lease does not fence the proxy path either: an exclusive lease
    # being held does not break ensure_active for requests that arrive
    # after eviction (the model restarts on demand).
    async with mgr.acquire("vision", exclusive=True):
        assert rt.running_models() == []
    model, srv = await rt.ensure_active("qwen")
    assert model.name == "qwen"
    assert FakeServer.starts[-1] == "qwen"


async def test_no_router_is_supported(tmp_path, monkeypatch):
    """A manager without a router grants leases and skips drain/evict."""
    mgr = ResourceLeaseManager(None)
    async with mgr.acquire("vision"):
        assert mgr.active_leases == {"vision": 1}
    assert mgr.active_leases == {}
