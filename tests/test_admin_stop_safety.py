"""Stopping a model must not SIGKILL a busy or slow-to-exit llama-server (it can wedge Intel Arc GPUs)."""
from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from helpers import FakeServer

import arc_llama.router as router_mod
from arc_llama import launcher
from arc_llama.failures import StartupFailureError
from arc_llama.launcher import LaunchPlan, LlamaServer
from arc_llama.router import Router


@pytest.fixture(autouse=True)
def _inline_router_threads(monkeypatch):
    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(router_mod.asyncio, "to_thread", inline_to_thread)


def _fake_server(monkeypatch, exits_on_sigterm: bool) -> tuple[LlamaServer, Mock, Mock]:
    monkeypatch.setattr(launcher, "_IS_WINDOWS", False)
    monkeypatch.setattr(launcher.signal, "SIGKILL", getattr(launcher.signal, "SIGKILL", 9), raising=False)
    killpg = Mock()
    monkeypatch.setattr(launcher.os, "killpg", killpg, raising=False)
    srv = LlamaServer(LaunchPlan(argv=["unused"], env={}), "m")
    proc = Mock(pid=4242)
    if exits_on_sigterm:
        proc.poll.side_effect = [None, 0, 0, 0]
        proc.wait.return_value = 0
    else:
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("fake", 0)
    srv.process = proc
    srv.ready = True
    return srv, proc, killpg


def test_default_grace_is_long(monkeypatch):
    monkeypatch.delenv("ARC_LLAMA_STOP_GRACE", raising=False)
    assert launcher.default_stop_grace() == launcher.STOP_GRACE_DEFAULT_SECONDS >= 30


def test_grace_env_override_and_bad_values(monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_STOP_GRACE", "12.5")
    assert launcher.default_stop_grace() == 12.5
    monkeypatch.setenv("ARC_LLAMA_STOP_GRACE", "nonsense")
    assert launcher.default_stop_grace() == launcher.STOP_GRACE_DEFAULT_SECONDS
    monkeypatch.setenv("ARC_LLAMA_STOP_GRACE", "-4")
    assert launcher.default_stop_grace() == 0.0


def test_stop_waits_the_full_grace_after_sigterm(monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_STOP_GRACE", "45")
    srv, proc, killpg = _fake_server(monkeypatch, exits_on_sigterm=True)
    srv.stop()
    proc.wait.assert_called_once_with(timeout=45.0)
    assert [c.args[1] for c in killpg.call_args_list] == [launcher.signal.SIGTERM]  # no SIGKILL


def test_explicit_drain_still_wins(monkeypatch):
    srv, proc, killpg = _fake_server(monkeypatch, exits_on_sigterm=True)
    srv.stop(drain_seconds=0.25)
    proc.wait.assert_called_once_with(timeout=0.25)


def test_sigkill_only_after_the_grace_expires(monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_STOP_GRACE", "7")
    srv, proc, killpg = _fake_server(monkeypatch, exits_on_sigterm=False)
    with pytest.raises(StartupFailureError):
        srv.stop()
    assert proc.wait.call_args_list[0].kwargs == {"timeout": 7.0}
    assert [c.args[1] for c in killpg.call_args_list] == [launcher.signal.SIGTERM, launcher.signal.SIGKILL]


async def _router_with_running(tmp_path, monkeypatch, name="qwen"):
    from conftest import make_config

    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    rt = Router(make_config(tmp_path, single_resident=False))
    await rt.ensure_active(name)
    return rt


async def test_stop_one_refuses_while_serving(tmp_path, monkeypatch):
    rt = await _router_with_running(tmp_path, monkeypatch)
    rt.acquire_model("qwen")
    with pytest.raises(StartupFailureError) as caught:
        await rt.stop_one("qwen")
    assert caught.value.category == "model_busy" and caught.value.http_status == 409
    assert FakeServer.stops == []
    rt.release_model("qwen")
    assert await rt.stop_one("qwen") is True
    assert FakeServer.stops == ["qwen"]


async def test_stop_one_force_stops_a_busy_model(tmp_path, monkeypatch):
    rt = await _router_with_running(tmp_path, monkeypatch)
    rt.acquire_model("qwen")
    assert await rt.stop_one("qwen", force=True) is True
    assert FakeServer.stops == ["qwen"]


async def test_stop_all_refuses_and_stops_nothing_while_any_model_serves(tmp_path, monkeypatch):
    rt = await _router_with_running(tmp_path, monkeypatch)
    rt.acquire_model("qwen")
    with pytest.raises(StartupFailureError) as caught:
        await rt.stop_all()
    assert caught.value.category == "model_busy"
    assert FakeServer.stops == []
    assert await rt.stop_all(force=True) == 1


def test_admin_endpoint_returns_409_when_busy_and_force_stops(tmp_path, monkeypatch):
    from conftest import make_config

    from arc_llama.server import create_app

    FakeServer.starts = []
    FakeServer.stops = []
    monkeypatch.setattr(router_mod, "LlamaServer", FakeServer)
    cfg = make_config(tmp_path, single_resident=False)
    app = create_app(cfg)
    with TestClient(app) as client:
        rt: Router = app.state.router
        client.portal.call(rt.ensure_active, "qwen")
        rt.acquire_model("qwen")
        r = client.post("/admin/stop/qwen")
        assert r.status_code == 409 and r.json()["error"]["category"] == "model_busy"
        assert FakeServer.stops == []
        r = client.post("/admin/stop/qwen?force=true")
        assert r.status_code == 200 and r.json()["was_running"] is True
