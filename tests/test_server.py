from __future__ import annotations

from fastapi.testclient import TestClient

from arc_llama.config import ModelConfig
from arc_llama.server import create_app


class FakeServerPlan:
    backend_url = "http://fake-upstream"


class FakeBackend:
    plan = FakeServerPlan()
    is_running = True


class FakeRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self.model = ModelConfig(
            name="qwen",
            path="/models/qwen.gguf",
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen",
            aliases=["qwen.gguf"],
        )

    def all_models(self):
        return [self.model]

    async def ensure_active(self, query):
        if query not in {"qwen", "qwen.gguf"}:
            raise KeyError(query)
        return self.model, FakeBackend()

    async def shutdown(self):
        return None


class FakeResponse:
    status_code = 200
    headers = {
        "content-type": "application/json",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "ok",
    }
    content = b'{"ok": true}'


class FakeUpstreamStream:
    status_code = 200
    headers = {
        "content-type": "text/event-stream",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "ok",
    }
    closed = False

    async def aiter_raw(self):
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    async def aclose(self):
        self.closed = True


class FakeAsyncClient:
    last_stream = None

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def build_request(self, method, url, content=None, headers=None):
        return {
            "method": method,
            "url": url,
            "content": content,
            "headers": headers,
        }

    async def send(self, request, stream=False):
        assert request["url"] == "http://fake-upstream/v1/chat/completions"
        assert stream is True
        FakeAsyncClient.last_stream = FakeUpstreamStream()
        return FakeAsyncClient.last_stream

    async def post(self, url, content=None, headers=None):
        assert url == "http://fake-upstream/v1/chat/completions"
        return FakeResponse()

    async def aclose(self):
        self.closed = True


def test_non_streaming_proxy_strips_hop_by_hop_headers(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["x-upstream"] == "ok"
    assert "content-length" in response.headers
    assert "transfer-encoding" not in response.headers


def test_streaming_proxy_forwards_raw_sse_chunks_and_closes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: one\n\ndata: two\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-upstream"] == "ok"
    assert "transfer-encoding" not in response.headers
    assert FakeAsyncClient.last_stream.closed is True
