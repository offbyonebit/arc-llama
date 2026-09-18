"""Plugin discovery records: failed and disabled plugins stay visible in the
admin catalog without ever running, while the existing plugin API contract
keeps working unchanged."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from arc_llama.plugins import (
    Plugin,
    PluginDiscovery,
    build_catalog,
    load_plugins,
)


class _FakeEntryPoint:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


class GoodPlugin(Plugin):
    name = "good"

    def register(self, app: FastAPI) -> None: ...

    def info(self):
        return {"version": "1.0", "description": "works"}


class BadRegisterPlugin(Plugin):
    name = "bad-register"

    def register(self, app: FastAPI) -> None:
        raise RuntimeError("boom")


def _record_loaded_plugin():
    discovery = PluginDiscovery()
    load_plugins([_FakeEntryPoint("good", GoodPlugin)], discovery=discovery)
    return discovery


def test_load_plugins_records_successful_loads():
    discovery = PluginDiscovery()
    plugins = load_plugins([_FakeEntryPoint("good", GoodPlugin)], discovery=discovery)
    assert [getattr(p, "name", "?") for p in plugins] == ["good"]
    assert discovery.loaded_names() == {"good"}


def test_load_plugins_retains_failure_records():
    discovery = PluginDiscovery()
    plugins = load_plugins(
        [
            _FakeEntryPoint("broken-import", ImportError("missing dep")),
            _FakeEntryPoint("bad-factory", lambda: (_ for _ in ()).throw(RuntimeError("no constructor"))),
        ],
        discovery=discovery,
    )
    assert plugins == []
    catalog = discovery.catalog()
    by_name = {entry["name"]: entry for entry in catalog}
    assert set(by_name) == {"broken-import", "bad-factory"}
    for entry in catalog:
        assert entry["status"] == "failed"
        assert entry["error"], "the useful error must travel with the record"
    assert "missing dep" in by_name["broken-import"]["error"]


def _named_point(name: str):
    """An entry-point-like object whose plugin reports the entry-point name."""
    named_plugin_cls = type(
        "NamedPlugin",
        (Plugin,),
        {
            "name": name,
            "register": lambda self, app: None,
            "__init__": lambda self: None,
        },
    )
    return _FakeEntryPoint(name, named_plugin_cls)


def test_load_plugins_retains_disabled_records():
    discovery = PluginDiscovery()
    eps = [_named_point("a"), _named_point("b")]
    plugins = load_plugins(eps, enabled={"a"}, discovery=discovery)
    assert [getattr(p, "name", "?") for p in plugins] == ["a"]
    catalog = discovery.catalog()
    by_name = {entry["name"]: entry for entry in catalog}
    assert set(by_name) == {"b"}
    assert by_name["b"]["status"] == "disabled"


def test_missing_register_contract_is_recorded():
    discovery = PluginDiscovery()
    plugins = load_plugins([_FakeEntryPoint("no-register", object())], discovery=discovery)
    assert plugins == []
    (entry,) = discovery.catalog()
    assert entry["status"] == "failed"
    assert "register" in entry["error"]


def test_build_catalog_merges_discovery_records_without_duplicates():
    discovery = _record_loaded_plugin()
    live = [GoodPlugin()]
    extras = discovery.catalog() + [
        {"name": "broken-import", "status": "failed", "error": "import failed: boom"}
    ]
    catalog = build_catalog(live, {"good": "active"}, extra=extras)
    by_name = {entry["name"]: entry for entry in catalog}
    # The live record wins; the failed one appears once alongside it.
    assert by_name["good"]["status"] == "active"
    assert by_name["good"]["version"] == "1.0"
    assert by_name["broken-import"]["status"] == "failed"
    assert by_name["broken-import"]["error"] == "import failed: boom"
    assert [e["name"] for e in catalog].count("good") == 1


def test_build_catalog_ignores_nameless_or_duplicate_extra_records():
    catalog = build_catalog(
        [GoodPlugin()],
        extra=[
            {"name": "", "status": "failed"},
            {"name": "good", "status": "failed", "error": "dup"},
            {"status": "failed"},
        ],
    )
    assert [entry["name"] for entry in catalog] == ["good"]
    assert catalog[0]["status"] == "registered"


def test_admin_plugins_includes_discovery_records(monkeypatch):
    import arc_llama.server as server_mod
    from arc_llama.config import Config

    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager

    def fake_load_plugins(entry_points=None, *, enabled=None, discovery=None):
        assert discovery is not None
        discovery.record("broken-import", status="failed", error="import failed: nope")
        return [GoodPlugin()]

    monkeypatch.setattr(server_mod, "Router", FakeRouter, raising=False)
    monkeypatch.setattr(
        server_mod, "UpstreamManager", FakeUpstreamManager, raising=False
    )
    monkeypatch.setattr(server_mod, "load_plugins", fake_load_plugins)
    app = server_mod.create_app(Config(), plugins=None)
    with TestClient(app) as client:
        resp = client.get("/admin/plugins")
    assert resp.status_code == 200
    plugins = {p["name"]: p for p in resp.json()["plugins"]}
    assert plugins["good"]["status"] == "active"
    assert plugins["broken-import"]["status"] == "failed"
    assert plugins["broken-import"]["error"] == "import failed: nope"


def test_explicit_plugin_list_keeps_discovery_compatible(monkeypatch):
    """Existing callers passing plugins=... still get an empty catalog of
    records without errors: backward compatibility for the old API."""
    import arc_llama.server as server_mod
    from arc_llama.config import Config

    app = server_mod.create_app(Config(), plugins=[GoodPlugin()])
    assert app.state.plugin_discovery.catalog() == []
