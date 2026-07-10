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
        self._servers = {"qwen": FakeBackend()}

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


class FakeUpstreamManager:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = []

    async def models(self):
        return self._models

    def find_model(self, model_id):
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        raise RuntimeError("should not be called")

    def upstreams_status(self):
        return []


class FakeUpstreamModel:
    def __init__(self, model_id, upstream_name, upstream_url):
        self.id = model_id
        self.upstream_name = upstream_name
        self.upstream_url = upstream_url
        self.metadata = {}


class FakeUpstreamResponse:
    status_code = 200
    headers = {"content-type": "application/json", "x-upstream": "upstream-ok"}
    _content = b'{"upstream": true}'
    closed = False

    async def aread(self):
        return self._content

    async def aclose(self):
        self.closed = True

    async def aiter_raw(self):
        yield self._content


class FakeUpstreamManagerWithModels:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = [FakeUpstreamModel("llama3.1", "ollama", "http://127.0.0.1:11434")]

    async def models(self):
        return self._models

    def find_model(self, model_id):
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        resp = FakeUpstreamResponse()
        return resp

    def upstreams_status(self):
        return [{"name": "ollama", "url": "http://127.0.0.1:11434", "model_count": 1, "last_fetch": 123.0}]


class FakeAsyncClientUpstream:
    """httpx.AsyncClient that simulates upstream proxy responses."""
    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def build_request(self, method, url, content=None, headers=None):
        return {"method": method, "url": url, "content": content, "headers": headers}

    async def send(self, request, stream=False):
        resp = FakeUpstreamResponse()
        return resp

    async def aclose(self):
        pass


class FakeUpstreamStreamResponse:
    status_code = 200
    headers = {
        "content-type": "text/event-stream",
        "content-length": "999",
        "transfer-encoding": "chunked",
        "x-upstream": "stream-ok",
    }
    closed = False

    async def aiter_raw(self):
        yield b"data: upstream chunk 1\n\n"
        yield b"data: upstream chunk 2\n\n"

    async def aclose(self):
        self.closed = True


class FakeUpstreamManagerStreaming:
    def __init__(self, upstreams=None):
        self._upstreams = upstreams or []
        self._models = [FakeUpstreamModel("llama3.1", "ollama", "http://127.0.0.1:11434")]
        self.last_stream = None
        self.last_streaming_ok = None

    async def models(self):
        return self._models

    def find_model(self, model_id):
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    async def proxy(self, upstream, path, body, headers, streaming_ok=True):
        self.last_streaming_ok = streaming_ok
        self.last_stream = FakeUpstreamStreamResponse()
        return self.last_stream

    def upstreams_status(self):
        return []


def test_non_streaming_proxy_strips_hop_by_hop_headers(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
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
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
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


def test_upstream_model_proxy(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClientUpstream)
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"upstream": True}
    assert response.headers["x-upstream"] == "upstream-ok"


def test_list_models_includes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()["data"]
    ids = {m["id"] for m in data}
    assert "qwen" in ids
    assert "llama3.1" in ids
    upstream = next(m for m in data if m["id"] == "llama3.1")
    assert upstream["owned_by"] == "upstream:ollama"


def test_admin_status_includes_upstreams(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/admin/status")

    assert response.status_code == 200
    status = response.json()
    assert "upstreams" in status
    assert len(status["upstreams"]) == 1
    assert status["upstreams"][0]["name"] == "ollama"


def test_admin_load_rejects_upstream(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManagerWithModels)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        response = client.post("/admin/load/llama3.1")

    assert response.status_code == 400
    assert "Upstream model" in response.json()["detail"]


def test_upstream_streaming_proxy_forwards_sse_and_closes_upstream(monkeypatch):
    import arc_llama.server as server_mod

    mgr = FakeUpstreamManagerStreaming()
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", lambda upstreams=None: mgr)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "llama3.1", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: upstream chunk 1\n\ndata: upstream chunk 2\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-upstream"] == "stream-ok"
    assert "transfer-encoding" not in response.headers
    assert "content-length" not in response.headers
    assert mgr.last_streaming_ok is True
    assert mgr.last_stream.closed is True


# ---------------------------------------------------------------------------
# /admin/models/{name}/edit — perf recipe fields
# ---------------------------------------------------------------------------

class FakeRouterWithRebuild(FakeRouter):
    async def rebuild_model(self, name):
        return True, False


def _edit_app(monkeypatch, tmp_path):
    from conftest import make_config

    import arc_llama.server as server_mod

    # Keep cfg.save() away from the real ~/.config.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = make_config(tmp_path)
    return create_app(cfg), cfg


def test_admin_edit_accepts_flash_attn_and_batch_size(monkeypatch, tmp_path):
    app, cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/admin/models/qwen/edit",
            json={"flash_attn": "on", "batch_size": 2048, "ubatch_size": 1024},
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body["changed"]) == {"flash_attn", "batch_size", "ubatch_size"}
    model = next(m for m in cfg.models if m.name == "qwen")
    assert model.recipe["flash_attn"] == "on"
    assert model.recipe["batch_size"] == 2048
    assert model.recipe["ubatch_size"] == 1024


def test_admin_edit_flash_attn_null_clears(monkeypatch, tmp_path):
    app, cfg = _edit_app(monkeypatch, tmp_path)
    model = next(m for m in cfg.models if m.name == "qwen")
    model.recipe["flash_attn"] = "on"
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"flash_attn": None})
    assert response.status_code == 200
    assert "flash_attn" not in model.recipe


def test_admin_edit_rejects_bad_flash_attn(monkeypatch, tmp_path):
    app, _cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"flash_attn": "yes"})
    assert response.status_code == 400


def test_admin_edit_rejects_bad_batch_size(monkeypatch, tmp_path):
    app, _cfg = _edit_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"batch_size": 0})
    assert response.status_code == 400
