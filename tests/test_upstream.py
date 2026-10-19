"""Tests for arc_llama.upstream — manager, caching, proxy."""

from __future__ import annotations

import json
from typing import Any

import pytest

from arc_llama.config import UpstreamConfig
from arc_llama.upstream import UpstreamManager


class FakeHttpxResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._json or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aread(self):
        return json.dumps(self._json).encode()

    async def aclose(self):
        pass


class FakeHttpxClient:
    """Minimal async httpx client mock."""

    def __init__(self, responses: dict[str, FakeHttpxResponse] | None = None, timeout: float = 5.0):
        self.timeout = timeout
        self._responses = responses or {}
        self.closed = False
        self.calls: list[tuple[str, str]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def get(self, url: str):
        self.calls.append(("GET", url))
        resp = self._responses.get(url)
        if resp is None:
            raise RuntimeError(f"No mock for {url}")
        return resp

    async def aclose(self):
        self.closed = True


@pytest.fixture
def sample_upstreams():
    return [
        UpstreamConfig(name="ollama", url="http://127.0.0.1:11434"),
        UpstreamConfig(name="remote", url="http://192.168.1.50:8080"),
    ]


class TestFetch:
    @pytest.mark.asyncio
    async def test_fetch_models(self, sample_upstreams, monkeypatch):
        mgr = UpstreamManager(sample_upstreams)
        fake_resp = FakeHttpxResponse(
            json_data={
                "data": [
                    {"id": "llama3.1", "object": "model"},
                    {"id": "mistral", "object": "model"},
                ]
            }
        )
        fake_client = FakeHttpxClient(
            {
                "http://127.0.0.1:11434/v1/models": fake_resp,
            }
        )
        monkeypatch.setattr("arc_llama.upstream.httpx.AsyncClient", lambda **kw: fake_client)

        models = await mgr.models()
        # Only the first upstream was mocked; the second will fail gracefully.
        assert len(models) == 2
        assert {m.id for m in models} == {"llama3.1", "mistral"}
        assert all(m.upstream_name == "ollama" for m in models)

    @pytest.mark.asyncio
    async def test_fetch_failure_does_not_erase_cache(self, sample_upstreams, monkeypatch):
        mgr = UpstreamManager(sample_upstreams)
        # Prime cache with a successful fetch
        fake_resp = FakeHttpxResponse(json_data={"data": [{"id": "llama3.1"}]})
        fake_client = FakeHttpxClient(
            {
                "http://127.0.0.1:11434/v1/models": fake_resp,
            }
        )
        monkeypatch.setattr("arc_llama.upstream.httpx.AsyncClient", lambda **kw: fake_client)
        await mgr.models()

        # Now simulate a failure on the next fetch
        fail_client = FakeHttpxClient({})
        monkeypatch.setattr("arc_llama.upstream.httpx.AsyncClient", lambda **kw: fail_client)
        # Force a re-fetch by manipulating last_fetch timestamp
        mgr._last_fetch["ollama"] = 0
        models = await mgr.models()
        # Should still have the cached result
        assert len(models) == 1
        assert models[0].id == "llama3.1"


class TestFindModel:
    @pytest.mark.asyncio
    async def test_find_existing(self, sample_upstreams, monkeypatch):
        mgr = UpstreamManager(sample_upstreams)
        fake_resp = FakeHttpxResponse(json_data={"data": [{"id": "llama3.1"}]})
        fake_client = FakeHttpxClient(
            {
                "http://127.0.0.1:11434/v1/models": fake_resp,
            }
        )
        monkeypatch.setattr("arc_llama.upstream.httpx.AsyncClient", lambda **kw: fake_client)
        await mgr.models()

        found = mgr.find_model("llama3.1")
        assert found is not None
        assert found.id == "llama3.1"
        assert found.upstream_name == "ollama"

    def test_find_missing(self, sample_upstreams):
        mgr = UpstreamManager(sample_upstreams)
        assert mgr.find_model("nonexistent") is None


class TestUpstreamStatus:
    @pytest.mark.asyncio
    async def test_upstreams_status(self, sample_upstreams, monkeypatch):
        mgr = UpstreamManager(sample_upstreams)
        fake_resp = FakeHttpxResponse(json_data={"data": [{"id": "llama3.1"}]})
        fake_client = FakeHttpxClient(
            {
                "http://127.0.0.1:11434/v1/models": fake_resp,
            }
        )
        monkeypatch.setattr("arc_llama.upstream.httpx.AsyncClient", lambda **kw: fake_client)
        await mgr.models()

        status = mgr.upstreams_status()
        assert len(status) == 2
        ollama = next(s for s in status if s["name"] == "ollama")
        assert ollama["model_count"] == 1
        assert ollama["last_fetch"] is not None
        remote = next(s for s in status if s["name"] == "remote")
        assert remote["model_count"] == 0  # fetch failed
        assert remote["last_fetch"] is None
