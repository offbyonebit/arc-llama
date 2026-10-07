"""Authentication regression for the existing terminal UI."""
from __future__ import annotations

import httpx
import pytest

from arc_llama import tui


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_tui_authenticates_before_status_refresh(monkeypatch, explicit):
    token = "explicit-token" if explicit else "loopback-token"
    if explicit:
        monkeypatch.setenv("ARC_LLAMA_ADMIN_TOKEN", token)
    else:
        monkeypatch.delenv("ARC_LLAMA_ADMIN_TOKEN", raising=False)
    session_requests = []

    def handler(request):
        if request.url.path == "/admin/session-token":
            session_requests.append(True)
            return httpx.Response(200, json={"admin_token": token})
        assert request.headers["Authorization"] == f"Bearer {token}"
        return httpx.Response(200, json={"server": {}, "gpus": [], "models": [], "upstreams": []})

    original = httpx.AsyncClient
    monkeypatch.setattr(tui.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    app = tui.ArcLlamaTUI("http://127.0.0.1:11437")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._last_status is not None
    assert session_requests == ([] if explicit else [True])
