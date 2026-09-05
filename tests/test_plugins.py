from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from arc_llama.plugins import (
    Plugin,
    _instantiate,
    load_plugins,
    register_plugins,
    shutdown_plugins,
    startup_plugins,
)
from arc_llama.server import create_app


class _FakeEntryPoint:
    """Minimal stand-in for an importlib.metadata.EntryPoint."""

    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class FakePlugin(Plugin):
    name = "fake"

    def __init__(self):
        self.registered = False
        self.started = False
        self.stopped = False

    def register(self, app: FastAPI) -> None:
        self.registered = True

        @app.get("/plugin/fake")
        async def fake_route():
            return {"plugin": "fake"}

    def startup(self, app: FastAPI) -> None:
        self.started = True

    def shutdown(self, app: FastAPI) -> None:
        self.stopped = True


class AsyncFakePlugin(Plugin):
    name = "async-fake"

    def __init__(self):
        self.started = False
        self.stopped = False

    def register(self, app: FastAPI) -> None:
        pass

    async def startup(self, app: FastAPI) -> None:
        await asyncio.sleep(0)
        self.started = True

    async def shutdown(self, app: FastAPI) -> None:
        await asyncio.sleep(0)
        self.stopped = True


class BrokenPlugin(Plugin):
    name = "broken"

    def register(self, app: FastAPI) -> None:
        raise RuntimeError("boom")


def test_instantiate_accepts_class_factory_and_instance():
    class P(Plugin):
        name = "p"

    p = P()
    assert _instantiate(P) is not None
    assert _instantiate(lambda: p) is p
    assert _instantiate(p) is p


def test_load_plugins_skips_broken_import():
    class BadEP:
        name = "bad"

        def load(self):
            raise ImportError("missing dep")

    plugins = load_plugins([BadEP()])
    assert plugins == []


def test_load_plugins_skips_broken_instantiation():
    class BadFactory:
        name = "bad-factory"

        def load(self):
            def factory():
                raise RuntimeError("nope")

            return factory

    assert load_plugins([BadFactory()]) == []


def test_load_plugins_skips_missing_register():
    class NoRegister:
        name = "no-register"

        def load(self):
            return object()

    assert load_plugins([NoRegister()]) == []


def test_load_plugins_enabled_filter():
    eps = [_FakeEntryPoint("a", FakePlugin), _FakeEntryPoint("b", FakePlugin)]
    plugins = load_plugins(eps, enabled={"a"})
    assert len(plugins) == 1


def test_load_plugins_env_filter(monkeypatch):
    monkeypatch.setenv("ARC_LLAMA_PLUGINS", "a")
    eps = [_FakeEntryPoint("a", FakePlugin), _FakeEntryPoint("b", FakePlugin)]
    plugins = load_plugins(eps)
    assert len(plugins) == 1


def test_register_plugins_isolates_failures():
    app = FastAPI()
    good = FakePlugin()
    register_plugins(app, [good, BrokenPlugin()])
    assert good.registered is True


async def test_startup_shutdown_run_sync_and_async():
    app = FastAPI()
    sync = FakePlugin()
    asyncp = AsyncFakePlugin()
    await startup_plugins([sync, asyncp], app)
    assert sync.started is True
    assert asyncp.started is True
    await shutdown_plugins([sync, asyncp], app)
    assert sync.stopped is True
    assert asyncp.stopped is True


async def test_startup_isolates_failure():
    app = FastAPI()
    good = FakePlugin()

    class BadStartup(Plugin):
        name = "bad-startup"

        def register(self, app):
            pass

        def startup(self, app):
            raise RuntimeError("boom")

    await startup_plugins([good, BadStartup()], app)
    assert good.started is True


def test_create_app_registers_plugin_routes_and_lifecycle():
    plugin = FakePlugin()
    app = create_app(plugins=[plugin])

    with TestClient(app) as client:
        resp = client.get("/plugin/fake")
        assert resp.status_code == 200
        assert resp.json() == {"plugin": "fake"}
        assert plugin.started is True

    assert plugin.stopped is True


def test_create_app_without_plugins_preserves_core_routes(monkeypatch):
    import arc_llama.server as server_mod
    from tests.test_server import FakeRouter, FakeUpstreamManager

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    app = create_app(plugins=[])

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
