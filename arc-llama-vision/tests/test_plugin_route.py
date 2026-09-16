"""Vision dashboard adapter route integration under a fake resource manager.

The adapter's ``/plugins/vision/generate`` must hold an exclusive GPU lease
around the outbound request to the vision companion: the lease manager
drains router requests and stops resident models, serialises concurrent
exclusive tasks, and — critically — releases the lease on error paths
(upstream failure, body exception, failed acquisition). These tests use a
recording fake manager so no Router or GPU is needed, and monkeypatch the
outbound HTTP client so no vision companion process is needed either.

Run from the repo root:

    .venv/bin/python -m pytest arc-llama-vision/tests -q
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

# Make the companion's src importable regardless of install state, mirroring
# test_vision.py.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import arc_llama_vision.plugin as plugin_mod  # noqa: E402
from arc_llama_vision.plugin import VisionPlugin, create_plugin  # noqa: E402


class _Resp:
    status_code = 200
    _json: dict[str, Any] = {"created": 1, "data": [{"b64_json": "aGk="}]}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return _Resp._json


class FakeAsyncClient:
    """httpx.AsyncClient stand-in for the outbound companion request."""

    last_payload: dict[str, Any] | None = None
    fail_with: Exception | None = None

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _Resp:
        assert url == "http://127.0.0.1:11440/v1/images/generations", f"unexpected url {url}"
        assert json is not None
        FakeAsyncClient.last_payload = json
        if FakeAsyncClient.fail_with is not None:
            raise FakeAsyncClient.fail_with
        return _Resp()


class FakeResourceLeaseManager:
    """Recording stand-in for arc_llama.resources.ResourceLeaseManager.

    Records grant/release pairs; the real manager is exercised directly in
    tests/test_resources.py. The plugin only relies on the async
    context-manager shape of ``acquire``.

    ``grant_error`` fault-injects a failure while acquiring the lease.
    """

    def __init__(self, *, grant_error: Exception | None = None) -> None:
        self.events: list[tuple[str, str, bool]] = []
        self.held = False
        self._grant_error = grant_error

    @asynccontextmanager
    async def acquire(self, owner: str, *, exclusive: bool = True) -> AsyncIterator[None]:
        if self._grant_error is not None:
            raise self._grant_error
        self.held = True
        self.events.append(("acquire", owner, exclusive))
        try:
            yield
        finally:
            self.held = False
            self.events.append(("release", owner, exclusive))


@pytest.fixture(autouse=True)
def reset_fake_client() -> AsyncIterator[None]:
    FakeAsyncClient.last_payload = None
    FakeAsyncClient.fail_with = None
    yield


def _make_app(monkeypatch, manager: FakeResourceLeaseManager | None, client: type | None = None):
    from fastapi import FastAPI

    monkeypatch.setattr(plugin_mod.httpx, "AsyncClient", client or FakeAsyncClient)
    app = FastAPI()
    app.state.resources = manager
    create_plugin().register(app)
    return app


def _route(app, path: str):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


class DummyRequest:
    """Minimal Starlette Request stand-in carrying a JSON body."""

    def __init__(self, app: Any, payload: dict[str, Any]) -> None:
        self.app = app
        self._body = json.dumps(payload).encode("utf-8")

    async def json(self) -> dict[str, Any]:
        return json.loads(self._body)


async def test_generate_acquires_and_releases_exclusive_lease(monkeypatch):
    mgr = FakeResourceLeaseManager()
    app = _make_app(monkeypatch, mgr)
    response = await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "cube"}))
    assert json.loads(response.body.decode("utf-8")) == _Resp._json
    # Exactly one grant/release pair, exclusive, attributed to the plugin.
    assert mgr.events == [
        ("acquire", "vision", True),
        ("release", "vision", True),
    ]
    assert mgr.held is False
    assert FakeAsyncClient.last_payload == {
        "model": "arc-vision-diffusion",
        "prompt": "cube",
        "size": "512x512",
    }


async def test_outbound_request_runs_under_held_lease(monkeypatch):
    """The lease must span the outbound request itself."""
    mgr = FakeResourceLeaseManager()
    hold_observed: list[bool] = []

    class ObservingClient(FakeAsyncClient):
        async def post(self, url, json=None):
            hold_observed.append(mgr.held)
            return await super().post(url, json=json)

    app = _make_app(monkeypatch, mgr, client=ObservingClient)
    await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "x"}))
    assert hold_observed == [True], "outbound request ran without holding the lease"


async def test_missing_resource_manager_refuses_cleanly(monkeypatch):
    """No app.state.resources -> a clear 503, not an unarbitrated generation."""
    from fastapi import HTTPException

    app = _make_app(monkeypatch, None)  # registers getattr-default path
    del app.state.resources  # simulate a server with no arbitration surface
    with pytest.raises(HTTPException) as caught:
        await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "x"}))
    assert caught.value.status_code == 503
    assert "resource arbitration" in caught.value.detail.lower()
    assert FakeAsyncClient.last_payload is None


async def test_upstream_failure_releases_lease(monkeypatch):
    """A dead companion must fail clearly AND release the lease."""
    from fastapi import HTTPException

    mgr = FakeResourceLeaseManager()
    FakeAsyncClient.fail_with = httpx.ConnectError("connection refused")
    app = _make_app(monkeypatch, mgr)
    with pytest.raises(HTTPException) as caught:
        await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "x"}))
    assert caught.value.status_code == 503
    assert "vision companion is not running" in caught.value.detail.lower()
    assert mgr.held is False
    assert mgr.events[-1] == ("release", "vision", True)


async def test_body_exception_releases_lease(monkeypatch):
    mgr = FakeResourceLeaseManager()

    class ExplodingClient(FakeAsyncClient):
        async def post(self, url, json=None):
            raise RuntimeError("surprise failure inside the outbound call")

    app = _make_app(monkeypatch, mgr, client=ExplodingClient)
    with pytest.raises(RuntimeError, match="surprise failure"):
        await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "x"}))
    assert mgr.held is False
    assert mgr.events[-1] == ("release", "vision", True)


async def test_grant_failure_never_reaches_the_outbound_call(monkeypatch):
    """If acquiring the lease fails, the route fails — it never generates."""
    mgr = FakeResourceLeaseManager(grant_error=RuntimeError("gate wedged"))
    app = _make_app(monkeypatch, mgr)
    with pytest.raises(RuntimeError, match="gate wedged"):
        await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "x"}))
    assert FakeAsyncClient.last_payload is None
    assert mgr.events == []


async def test_missing_prompt_rejected_before_leasing(monkeypatch):
    """Validation precedes lease acquisition — no GPU churn for bad input."""
    from fastapi import HTTPException

    mgr = FakeResourceLeaseManager()
    app = _make_app(monkeypatch, mgr)
    with pytest.raises(HTTPException) as caught:
        await _route(app, "/plugins/vision/generate")(DummyRequest(app, {"prompt": "  "}))
    assert caught.value.status_code == 400
    assert mgr.events == []
    assert FakeAsyncClient.last_payload is None


async def test_info_and_discovery_routes_stay_lease_free(monkeypatch):
    mgr = FakeResourceLeaseManager()
    app = _make_app(monkeypatch, mgr)
    with TestClient(app) as client:
        resp = client.get("/plugins/vision")
    assert resp.status_code == 200
    assert resp.json()["name"] == "vision"
    assert mgr.events == []


def test_plugin_instance_and_factory_shapes():
    plugin = create_plugin()
    assert isinstance(plugin, VisionPlugin)
    assert VisionPlugin.name == "vision"
    info = VisionPlugin().info()
    assert info["api"] == [
        "/plugins/vision",
        "http://127.0.0.1:11440/v1/images/generations",
    ]
