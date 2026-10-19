from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_server import FakeRouter, FakeUpstreamManager

from arc_llama.config import Config


@pytest.fixture
def budget_app(monkeypatch):
    import arc_llama.api.context_budget as budget_mod
    import arc_llama.server as server_mod

    calls = []
    replies = {
        "/props": {"default_generation_settings": {"n_ctx": 2048}},
        "/apply-template": {"prompt": "<bos>system: café 日本語\nuser: ```code```\nassistant:"},
        "/tokenize": {"tokens": [1, 2, 3, 4, 5]},
    }

    async def handler(request):
        import json

        calls.append((request.url.path, json.loads(request.content) if request.content else None))
        reply = replies[request.url.path]
        if reply == "stall":
            await asyncio.Event().wait()
        if isinstance(reply, int):
            return httpx.Response(reply, json={})
        return httpx.Response(200, json=reply)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        budget_mod,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
            HTTPError=httpx.HTTPError,
        ),
    )
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config()
    cfg.tune.auto = False
    return server_mod.create_app(cfg, plugins=[]), calls, replies


def _body(**kw):
    return {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "café 日本語"},
            {"role": "user", "content": "```python\nprint('hello')\n```"},
        ],
        "max_tokens": 128,
        **kw,
    }


def test_native_budget_uses_runtime_slot_and_preserves_template_inputs(budget_app):
    app, calls, replies = budget_app
    with TestClient(app) as client:
        app.state.router.model.recipe = {"ctx": 32768, "parallel": 4}
        body = _body(
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            chat_template_kwargs={"enable_thinking": False},
        )
        result = client.post("/v1/chat/budget", json=body).json()
        assert result["exact"] is True
        assert result["prompt_tokens"] == 5
        assert result["total_tokens"] == 133
        assert result["context_tokens"] == 2048  # Runtime, not configured total context.
        assert result["over_budget"] is False
        assert calls[1][1]["messages"] == body["messages"]
        assert calls[1][1]["tools"] == body["tools"]
        assert calls[1][1]["chat_template_kwargs"] == body["chat_template_kwargs"]
        assert calls[2][1] == {
            "content": replies["/apply-template"]["prompt"],
            "add_special": True,
            "parse_special": True,
        }
        assert app.state.router.model_inflight == {}
        assert app.state.router.inflight == 0
        assert app.state.resources.active_leases == {}


def test_exact_budget_includes_reply_allowance(budget_app):
    app, _, _ = budget_app
    with TestClient(app) as client:
        result = client.post("/v1/chat/budget", json=_body(max_tokens=2048)).json()
        assert result["over_budget"] is True
        assert result["reserved_tokens"] == 2048


@pytest.mark.parametrize(
    "mode", ["unloaded", "upstream", "image", "unsupported", "bad_props", "bad_tokens"]
)
def test_budget_fallback_is_explicit_and_never_loads_models(budget_app, mode, monkeypatch):
    app, calls, replies = budget_app
    with TestClient(app) as client:
        body = _body()
        if mode == "unloaded":
            app.state.router._servers["qwen"] = SimpleNamespace(is_running=False, ready=False)
        elif mode == "upstream":
            body["model"] = "external/model"
        elif mode == "image":
            body["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
                    ],
                }
            ]
        elif mode == "unsupported":
            replies["/apply-template"] = 404
        elif mode == "bad_props":
            replies["/props"] = {}
        else:
            replies["/tokenize"] = {"tokens": "wrong"}

        async def forbidden(*args, **kw):
            pytest.fail("counting must never start a model")

        monkeypatch.setattr(app.state.router, "ensure_active", forbidden)
        result = client.post("/v1/chat/budget", json=body).json()
        assert result["exact"] is False
        assert result["reason"]
        assert result["prompt_tokens"] > 0
        assert app.state.router.inflight == 0
        assert app.state.resources.active_leases == {}
        if mode in ("unloaded", "upstream", "image"):
            assert not calls


def test_budget_timeout_releases_admission_and_allows_retry(budget_app):
    app, _, replies = budget_app
    replies["/tokenize"] = "stall"
    with TestClient(app) as client:
        result = client.post("/v1/chat/budget", json=_body()).json()
        assert result["exact"] is False
        assert app.state.resources.active_leases == {}
        assert app.state.router.model_inflight == {}
        replies["/tokenize"] = {"tokens": [1]}
        assert client.post("/v1/chat/budget", json=_body()).json()["exact"] is True


@pytest.mark.parametrize(
    "change", [{"max_tokens": True}, {"max_tokens": -1}, {"messages": ["bad"]}, {"model": []}]
)
def test_budget_rejects_bad_inputs(budget_app, change):
    app, _, _ = budget_app
    with TestClient(app) as client:
        assert client.post("/v1/chat/budget", json=_body(**change)).status_code == 400


def test_budget_input_is_bounded(budget_app):
    app, _, _ = budget_app
    with TestClient(app) as client:
        assert (
            client.post("/v1/chat/budget", content=b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
        )
