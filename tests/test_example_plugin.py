from __future__ import annotations

from pathlib import Path

import tomllib
from fastapi.testclient import TestClient

from arc_llama.server import create_app


def test_example_plugin_declares_entry_point() -> None:
    root = Path(__file__).parents[1] / "examples" / "hello-plugin"
    with (root / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["entry-points"]["arc_llama.plugins"]["hello"] == (
        "arc_llama_hello_plugin:create_plugin"
    )


def test_example_plugin_registers_and_runs(monkeypatch) -> None:
    root = Path(__file__).parents[1] / "examples" / "hello-plugin"
    monkeypatch.syspath_prepend(str(root / "src"))

    from arc_llama_hello_plugin import create_plugin

    import arc_llama.server as server_mod
    from tests.test_server import FakeRouter, FakeUpstreamManager

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    plugin = create_plugin()
    app = create_app(plugins=[plugin])

    with TestClient(app) as client:
        response = client.get("/plugin/hello")
        assert response.status_code == 200
        assert response.json() == {
            "plugin": "hello",
            "message": "hello from an arc-llama plugin",
        }
        assert app.state.hello_plugin_started is True

    assert app.state.hello_plugin_stopped is True
