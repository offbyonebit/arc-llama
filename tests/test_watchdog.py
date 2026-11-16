from __future__ import annotations

import asyncio

import httpx
import pytest
from helpers import FakeServer, fake_router

from arc_llama.config import ServerConfig
from arc_llama.resources import ResourceLeaseManager


@pytest.fixture
def resident(tmp_path, monkeypatch):
    rt = fake_router(tmp_path, monkeypatch)
    srv = next(iter(rt._servers.values()))
    srv.running = srv.ready = True
    now = [10000.0]
    rt.watchdog.now = lambda: now[0]
    return rt, srv, now


def probe(monkeypatch, rt, healthy=False):
    calls = []

    async def health(srv):
        calls.append(srv.name)
        return healthy

    monkeypatch.setattr(rt.watchdog, "_healthy", health)
    return calls


@pytest.mark.asyncio
async def test_healthy_untouched(resident, monkeypatch):
    rt, srv, _ = resident
    calls = probe(monkeypatch, rt, True)
    for _ in range(5):
        await rt.watchdog.tick()
    assert calls == [srv.name] * 5
    assert FakeServer.starts == FakeServer.stops == []


@pytest.mark.asyncio
async def test_three_failures_restart_same_server_and_plan(resident, monkeypatch, caplog):
    rt, srv, _ = resident
    plan = srv.plan
    probe(monkeypatch, rt)
    await rt.watchdog.tick()
    await rt.watchdog.tick()
    assert FakeServer.starts == []
    await rt.watchdog.tick()
    assert FakeServer.starts == FakeServer.stops == [srv.name]
    assert rt._servers[srv.name] is srv and srv.plan is plan and srv.ready
    assert srv.name in caplog.text


@pytest.mark.asyncio
async def test_success_resets_consecutive_failure_count(resident, monkeypatch):
    rt, _, _ = resident
    probe(monkeypatch, rt)
    await rt.watchdog.tick()
    await rt.watchdog.tick()
    probe(monkeypatch, rt, True)
    await rt.watchdog.tick()
    probe(monkeypatch, rt)
    await rt.watchdog.tick()
    await rt.watchdog.tick()
    assert FakeServer.starts == []


@pytest.mark.asyncio
async def test_dead_ready_process_recovers_immediately(resident, monkeypatch):
    rt, srv, _ = resident
    srv.running = False
    calls = probe(monkeypatch, rt)
    await rt.watchdog.tick()
    assert calls == []
    assert FakeServer.starts == [srv.name]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["request", "model_request", "tune", "lease", "exclusive"])
async def test_busy_never_probed_or_restarted(resident, monkeypatch, kind):
    rt, srv, _ = resident
    calls = probe(monkeypatch, rt)
    srv.running = False
    if kind == "request":
        rt.inflight = 1
    elif kind == "model_request":
        rt.acquire_model(srv.name)
    elif kind == "tune":
        rt.watchdog.busy = lambda: True
    elif kind == "lease":
        resources = ResourceLeaseManager(rt)
        async with resources.acquire("benchmark", exclusive=False):
            await rt.watchdog.tick()
        assert FakeServer.starts == calls == []
        return
    else:
        resources = ResourceLeaseManager(rt)
        resources._exclusive_waiters = 1
    for _ in range(5):
        await rt.watchdog.tick()
    assert FakeServer.starts == calls == []


@pytest.mark.asyncio
async def test_request_arriving_during_probe_is_never_interrupted(resident, monkeypatch):
    rt, srv, _ = resident
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def health(_):
        entered.set()
        await finish.wait()
        return False

    state = rt.watchdog
    monkeypatch.setattr(state, "_healthy", health)
    # First two failures, then a request enters during the decisive probe.
    entered.set()
    finish.set()
    await state.tick()
    await state.tick()
    entered.clear()
    finish.clear()
    check = asyncio.create_task(state.tick())
    await entered.wait()
    rt.inflight = 1
    rt.acquire_model(srv.name)
    finish.set()
    await check
    assert FakeServer.stops == []


@pytest.mark.asyncio
async def test_request_during_restart_waits_for_new_ready_server(resident, monkeypatch):
    rt, srv, _ = resident
    ResourceLeaseManager(rt)
    srv.running = False
    entered = asyncio.Event()
    finish = asyncio.Event()
    original = srv.astop

    async def stop():
        entered.set()
        await finish.wait()
        await original()

    monkeypatch.setattr(srv, "astop", stop)
    check = asyncio.create_task(rt.watchdog.tick())
    await entered.wait()
    request = asyncio.create_task(rt.ensure_active(srv.name, acquire=True))
    await asyncio.sleep(0)
    assert not request.done()
    finish.set()
    await check
    _, result = await request
    assert result is srv and srv.ready
    assert rt.model_inflight[srv.name] == 1
    assert FakeServer.starts == [srv.name]
    rt.release_model(srv.name)


@pytest.mark.asyncio
async def test_manual_stop_during_probe_does_not_revive(resident, monkeypatch):
    rt, srv, _ = resident

    async def health(_):
        await rt.stop_one(srv.name)
        return False

    monkeypatch.setattr(rt.watchdog, "_healthy", health)
    await rt.watchdog.tick()
    assert FakeServer.starts == []
    assert not srv.is_running


@pytest.mark.asyncio
async def test_restart_limit_and_backoff_surface_error(resident, monkeypatch):
    rt, srv, now = resident
    probe(monkeypatch, rt)
    for i in range(3):
        srv.running = False
        await rt.watchdog.tick()
        assert len(FakeServer.starts) == i + 1
        srv.running = False
        await rt.watchdog.tick()
        assert len(FakeServer.starts) == i + 1
        now[0] += 240
    await rt.watchdog.tick()
    assert len(FakeServer.starts) == 3
    assert "three restarts" in rt.watchdog.error(srv.name)
    now[0] += 7200
    await rt.watchdog.tick()
    assert len(FakeServer.starts) == 3  # paused until explicit manual recovery
    await rt.stop_one(srv.name)
    assert rt.watchdog.error(srv.name) is None
    await rt.watchdog.tick()
    assert len(FakeServer.starts) == 3


@pytest.mark.asyncio
async def test_failed_restart_retries_bounded_then_pauses(resident, monkeypatch):
    rt, srv, now = resident
    srv.running = False

    def fail(**kwargs):
        FakeServer.starts.append(srv.name)
        raise OSError("broken runtime")

    monkeypatch.setattr(srv, "start", fail)
    for _ in range(4):
        await rt.watchdog.tick()
        now[0] += 240
    assert FakeServer.starts == [srv.name] * 3
    assert rt.watchdog.states[srv.name].paused
    assert "three restarts" in rt.watchdog.error(srv.name)


@pytest.mark.asyncio
async def test_disabled_no_network_no_restart(resident, monkeypatch):
    rt, srv, _ = resident
    rt.cfg.server.watchdog = False
    srv.running = False
    calls = probe(monkeypatch, rt)
    rt.watchdog.start()
    await rt.watchdog.tick()
    assert calls == FakeServer.starts == []
    assert rt.watchdog.task is None


@pytest.mark.asyncio
async def test_health_http_timeout_status_and_body(resident, monkeypatch):
    rt, srv, _ = resident
    outcomes = [httpx.Response(200, json={"status": "ok"}),
                httpx.Response(503), httpx.Response(200, json={"status": "error"}),
                httpx.ReadTimeout("hung")]
    seen = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 2.0, "trust_env": False}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            seen.append(url)
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr("arc_llama.watchdog.httpx.AsyncClient", Client)
    assert [await rt.watchdog._healthy(srv) for _ in range(4)] == [True, False, False, False]
    assert seen == [srv.plan.health_url] * 4


def test_default_on_boolean_only():
    assert ServerConfig().watchdog is True
    with pytest.raises(ValueError, match="boolean"):
        ServerConfig(watchdog="false")


@pytest.mark.asyncio
async def test_failed_recovery_does_not_revive_beside_new_resident(resident, monkeypatch):
    rt, srv, now = resident
    srv.running = False
    original = srv.start
    monkeypatch.setattr(srv, "start", lambda **kw: (_ for _ in ()).throw(OSError("missing")))
    await rt.watchdog.tick()
    monkeypatch.setattr(srv, "start", original)
    other = next(s for s in rt._servers.values() if s is not srv)
    other.running = other.ready = True
    now[0] += 1000
    probe(monkeypatch, rt, True)
    await rt.watchdog.tick()
    assert not srv.is_running
    assert FakeServer.starts == []


@pytest.mark.asyncio
async def test_loop_waits_sixty_seconds_and_shutdown_cancels(resident, monkeypatch):
    rt, _, _ = resident
    slept = []
    sleeping = asyncio.Event()
    hold = asyncio.Event()

    async def sleep(seconds):
        slept.append(seconds)
        sleeping.set()
        await hold.wait()

    monkeypatch.setattr("arc_llama.watchdog.asyncio.sleep", sleep)
    rt.watchdog.start()
    task = rt.watchdog.task
    await sleeping.wait()
    assert slept == [60]
    await rt.shutdown()
    assert task.done() and rt.watchdog.task is None
