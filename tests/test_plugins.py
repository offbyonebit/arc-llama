from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from arc_llama.plugins import (
    Plugin,
    _instantiate,
    build_catalog,
    load_plugins,
    plugin_info,
    register_plugins,
    shutdown_plugins,
    startup_plugins,
)
from arc_llama.server import create_app


def _server_fakes():
    """Import the shared router/upstream test doubles.

    ``tests`` is importable as a package in CI (repo root on sys.path) but
    not under every local pytest invocation in rootdir mode; accept the flat
    module name too so the tests here do not hinge on import configuration.
    """
    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager

    return FakeRouter, FakeUpstreamManager


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


def test_plugin_info_defaults_to_empty():
    assert plugin_info(FakePlugin()) == {}


def test_plugin_info_returns_declared_metadata():
    class InfoPlugin(Plugin):
        name = "info"

        def register(self, app): ...

        def info(self):
            return {"version": "1.2.3", "description": "does things", "ui": {"panel": "x"}}

    data = plugin_info(InfoPlugin())
    assert data == {"version": "1.2.3", "description": "does things", "ui": {"panel": "x"}}


def test_plugin_info_swallows_hook_failures():
    class BadInfo(Plugin):
        name = "bad-info"

        def register(self, app): ...

        def info(self):
            raise RuntimeError("boom")

    assert plugin_info(BadInfo()) == {}


def test_plugin_info_swallows_non_mapping():
    class BadInfo(Plugin):
        name = "bad-info"

        def register(self, app): ...

        def info(self):
            return "not a mapping"

    assert plugin_info(BadInfo()) == {}


def test_build_catalog_shape_and_defaults():
    plugin = FakePlugin()
    catalog = build_catalog([plugin], {})
    assert catalog == [{"name": "fake", "status": "registered"}]


def test_build_catalog_passes_metadata_and_status():
    class InfoPlugin(Plugin):
        name = "vision-extra"

        def register(self, app): ...

        def info(self):
            return {
                "version": "0.3.0",
                "description": "Image add-on",
                "ui": {"panel": "Vision"},
                "api": ["/v1/images"],
                "custom": {"anything": True},
            }

    catalog = build_catalog([InfoPlugin()], {"vision-extra": "active"})
    assert catalog == [
        {
            "name": "vision-extra",
            "status": "active",
            "version": "0.3.0",
            "description": "Image add-on",
            "ui": {"panel": "Vision"},
            "api": ["/v1/images"],
            "custom": {"anything": True},
        }
    ]


def test_build_catalog_skips_unnamed_plugins():
    class Anonymous:
        def register(self, app): ...

    assert build_catalog([Anonymous()], {}) == []


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


def _catalog_app(monkeypatch, plugins):
    """create_app with fake core services and a pinned admin token so the
    /admin/plugins route is deterministic in any environment."""
    import arc_llama.server as server_mod
    from arc_llama.config import Config, ServerConfig

    FakeRouter, FakeUpstreamManager = _server_fakes()  # noqa: N806 - class doubles
    monkeypatch.setattr(server_mod, "Router", FakeRouter, raising=False)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager, raising=False)
    cfg = Config(server=ServerConfig(admin_token="test-token"))
    return create_app(cfg, plugins=list(plugins))


AUTH = {"Authorization": "Bearer test-token"}


def test_create_app_without_plugins_preserves_core_routes(monkeypatch):
    app = _catalog_app(monkeypatch, [])

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        resp = client.get("/admin/plugins", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"plugins": []}


def test_create_app_admin_plugins_requires_admin_token(monkeypatch):
    app = _catalog_app(monkeypatch, [FakePlugin()])

    with TestClient(app) as client:
        assert client.get("/admin/plugins").status_code == 401
        assert client.get("/admin/plugins", headers=AUTH).status_code == 200


def test_create_app_admin_plugins_lists_installed_plugin(monkeypatch):
    app = _catalog_app(monkeypatch, [FakePlugin()])

    with TestClient(app) as client:
        resp = client.get("/admin/plugins", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"plugins": [{"name": "fake", "status": "active"}]}


def test_create_app_admin_plugins_exposes_info_metadata(monkeypatch):
    class CatalogPlugin(Plugin):
        name = "audio-extra"

        def register(self, app): ...

        def info(self):
            return {"version": "2.0.0", "description": "Audio routes"}

    app = _catalog_app(monkeypatch, [CatalogPlugin()])

    with TestClient(app) as client:
        resp = client.get("/admin/plugins", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        entry = next(p for p in body["plugins"] if p["name"] == "audio-extra")
        assert entry["status"] == "active"
        assert entry["version"] == "2.0.0"
        assert entry["description"] == "Audio routes"


def test_create_app_admin_plugins_reports_error_status(monkeypatch):
    app = _catalog_app(monkeypatch, [BrokenPlugin(), FakePlugin()])

    with TestClient(app) as client:
        resp = client.get("/admin/plugins", headers=AUTH)
        assert resp.status_code == 200
        by_name = {p["name"]: p["status"] for p in resp.json()["plugins"]}
        assert by_name["broken"] == "error"
        assert by_name["fake"] == "active"
