from __future__ import annotations

import importlib
import sys

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from arc_llama.cli import cli
from arc_llama.config import Config, GPUConfig, ModelConfig, ServerConfig
from arc_llama.plugin_api import (
    MODEL_LOADED,
    MODEL_STOPPED,
    PLUGIN_API_VERSION,
    EventBus,
    Plugin,
    api_compatible,
    plugin_context,
)
from arc_llama.plugin_scaffold import scaffold_files, validate_plugin_name, write_scaffold
from arc_llama.plugins import PluginDiscovery, load_plugins, plugin_health, plugin_info
from arc_llama.router import Router
from arc_llama.server import create_app


class _EP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def _server_fakes():
    try:
        from tests.test_server import FakeRouter, FakeUpstreamManager
    except ModuleNotFoundError:
        from test_server import FakeRouter, FakeUpstreamManager
    return FakeRouter, FakeUpstreamManager


@pytest.mark.parametrize(
    ("required", "expected"),
    [
        (None, True),
        (PLUGIN_API_VERSION, True),
        ("1", True),
        ("1.0", True),
        ("1.99", False),
        ("2.0", False),
        ("0.9", False),
        ("one", False),
        ("1.0.0", False),
    ],
)
def test_api_compatible(required, expected) -> None:
    assert api_compatible(required, "1.0") is expected


def test_incompatible_plugin_is_recorded_not_loaded() -> None:
    class Future(Plugin):
        name = "future"
        requires_api = "99.0"

    class Current(Plugin):
        name = "current"
        requires_api = PLUGIN_API_VERSION

    discovery = PluginDiscovery()
    loaded = load_plugins(
        [_EP("future", Future), _EP("current", Current)], enabled=None, discovery=discovery
    )
    assert [p.name for p in loaded] == ["current"]
    [record] = discovery.catalog()
    assert record["name"] == "future"
    assert record["status"] == "incompatible"
    assert "99.0" in record["error"]


def test_non_string_requirement_is_incompatible() -> None:
    class Odd(Plugin):
        name = "odd"
        requires_api = 1  # type: ignore[assignment]

    assert load_plugins([_EP("odd", Odd)], enabled=None) == []


def test_plugin_health_reports_loaded_and_failed() -> None:
    class Good(Plugin):
        name = "good"
        requires_api = "1.0"

        def info(self):
            return {"version": "2.3"}

    class Broken:
        def load(self):
            raise ImportError("missing torch")

        name = "broken"

    health = plugin_health([_EP("good", Good), Broken()])
    by_name = {entry["name"]: entry for entry in health}
    assert by_name["good"] == {
        "name": "good",
        "status": "loaded",
        "version": "2.3",
        "requires_api": "1.0",
    }
    assert by_name["broken"]["status"] == "failed"
    assert "missing torch" in by_name["broken"]["error"]


def test_event_bus_isolates_failures_and_unsubscribes() -> None:
    bus = EventBus()
    seen: list[tuple[str, dict]] = []

    def boom(event, payload):
        raise RuntimeError("bad subscriber")

    bus.subscribe(MODEL_LOADED, boom)
    unsubscribe = bus.subscribe(MODEL_LOADED, lambda e, p: seen.append((e, p)))
    bus.emit(MODEL_LOADED, {"model": "a"})
    unsubscribe()
    bus.emit(MODEL_LOADED, {"model": "b"})
    assert seen == [(MODEL_LOADED, {"model": "a"})]
    with pytest.raises(ValueError):
        bus.subscribe("nope", boom)


async def test_router_emits_stop_events(tmp_path) -> None:
    cfg = Config(
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[
            ModelConfig(
                name="m", path=str(tmp_path / "m.gguf"), port=18081, gpu_pci_slot="0000:03:00.0"
            )
        ],
    )
    router = Router(cfg)
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(MODEL_STOPPED, lambda e, p: seen.append(p))
    router.events = bus

    class Running:
        is_running = True
        process = object()

        async def astop(self):
            self.is_running = False

    router._servers["m"] = Running()  # type: ignore[assignment]
    assert await router.stop_one("m") is True
    assert seen == [{"model": "m", "reason": "stopped"}]


def test_plugin_pages_are_validated() -> None:
    class Paged(Plugin):
        name = "paged"

        def info(self):
            return {
                "ui": {
                    "pages": [
                        {"id": "ok", "label": "Open", "path": "/plugins/paged/"},
                        {"id": "ext", "label": "Bad", "path": "https://evil.example/"},
                        {"id": "esc", "label": "Bad", "path": "/plugins/../admin"},
                        {"id": "proto", "label": "Bad", "path": "/plugins//evil"},
                        {"id": "ok", "label": "Duplicate", "path": "/plugins/paged/2"},
                        "junk",
                    ]
                }
            }

    assert plugin_info(Paged())["ui"]["pages"] == [
        {"id": "ok", "label": "Open", "path": "/plugins/paged/"}
    ]


async def test_context_exposes_models_lease_events_and_admin(monkeypatch) -> None:
    import arc_llama.server as server_mod

    fake_router, fake_upstream = _server_fakes()
    monkeypatch.setattr(server_mod, "Router", fake_router)
    monkeypatch.setattr(server_mod, "UpstreamManager", fake_upstream)
    events: list[dict] = []

    class Probe(Plugin):
        name = "probe"
        requires_api = PLUGIN_API_VERSION

        def register(self, app: FastAPI) -> None:
            ctx = plugin_context(app)

            @app.get("/plugins/probe/models")
            async def models():
                async with ctx.gpu_lease("probe", exclusive=False) as lease:
                    return {"models": ctx.models(), "owner": lease.owner, "gpus": ctx.gpus()}

            @app.post("/plugins/probe/secret", dependencies=[ctx.require_admin])
            async def secret():
                return {"ok": True}

        def startup(self, app: FastAPI) -> None:
            plugin_context(app).subscribe(MODEL_LOADED, lambda e, p: events.append(p))

    cfg = Config(server=ServerConfig(admin_token="tok"))
    app = create_app(cfg=cfg, plugins=[Probe()])
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.get("/plugins/probe/models")
            assert response.status_code == 200
            body = response.json()
            assert body["owner"] == "probe"
            assert body["models"][0]["name"] == "qwen"
            assert (await client.post("/plugins/probe/secret")).status_code == 401
            ok = await client.post(
                "/plugins/probe/secret", headers={"Authorization": "Bearer tok"}
            )
            assert ok.status_code == 200
            catalog = await client.get(
                "/admin/plugins", headers={"Authorization": "Bearer tok"}
            )
            assert catalog.json()["api_version"] == PLUGIN_API_VERSION
        app.state.router.events.emit(MODEL_LOADED, {"model": "qwen"})
    assert events == [{"model": "qwen"}]


def test_scaffold_name_validation() -> None:
    assert validate_plugin_name("audio_tools") == "audio_tools"
    for bad in ("", "Audio", "1x", "a-b", "a" * 41):
        with pytest.raises(ValueError):
            validate_plugin_name(bad)


async def test_scaffold_generates_a_working_plugin(tmp_path, monkeypatch) -> None:
    target = tmp_path / "pkg"
    write_scaffold("demo_tool", target)
    assert set(scaffold_files("demo_tool")) == {
        str(p.relative_to(target)).replace("\\", "/") for p in target.rglob("*") if p.is_file()
    }
    monkeypatch.syspath_prepend(str(target / "src"))
    module = importlib.import_module("arc_llama_demo_tool")
    try:
        plugin = module.create_plugin()
        assert plugin.name == "demo_tool"
        assert api_compatible(plugin.requires_api)
        [loaded] = load_plugins([_EP("demo_tool", module.create_plugin)], enabled=None)
        app = FastAPI()
        loaded.register(app)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.get("/plugins/demo_tool/status")
        assert response.json()["plugin"] == "demo_tool"
        compile((target / "tests" / "test_plugin.py").read_text(), "test_plugin.py", "exec")
    finally:
        sys.modules.pop("arc_llama_demo_tool", None)


def test_scaffold_refuses_non_empty_target(tmp_path) -> None:
    (tmp_path / "existing.txt").write_text("x")
    with pytest.raises(FileExistsError):
        write_scaffold("demo", tmp_path)


def test_plugin_cli_new_and_list(tmp_path, monkeypatch) -> None:
    import arc_llama.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "discover_entry_points", lambda: [])
    runner = CliRunner()
    result = runner.invoke(cli, ["plugin", "new", "demo", "--dir", str(tmp_path / "out")], obj={})
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "src" / "arc_llama_demo" / "__init__.py").is_file()
    bad = runner.invoke(cli, ["plugin", "new", "Bad-Name"], obj={})
    assert bad.exit_code != 0
    listed = runner.invoke(cli, ["plugin", "list", "--json"], obj={})
    assert listed.exit_code == 0, listed.output
    assert '"plugins": []' in listed.output
