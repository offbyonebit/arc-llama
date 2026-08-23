from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from arc_llama.config import (
    Config,
    GPUConfig,
    MCPServerConfig,
    ModelConfig,
    PathsConfig,
    ProfileConfig,
    ServerConfig,
    TuneConfig,
)
from arc_llama.server import create_app


class FakeServerPlan:
    backend_url = "http://fake-upstream"


class FakeBackend:
    plan = FakeServerPlan()
    is_running = True
    ready = True


class FakeRouter:
    last_activity = 0.0
    inflight = 0

    # Per-model in-flight accounting, mirroring Router. server.py calls these
    # on every proxied request that resolves to a local model.
    def acquire_model(self, name):
        self.model_inflight[name] = self.model_inflight.get(name, 0) + 1

    def release_model(self, name):
        current = self.model_inflight.get(name, 0)
        if current <= 1:
            self.model_inflight.pop(name, None)
        else:
            self.model_inflight[name] = current - 1

    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self.model_inflight = {}
        self.model = ModelConfig(
            name="qwen",
            path="/models/qwen.gguf",
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            display_name="Qwen",
            aliases=["qwen.gguf"],
        )
        self._servers = {"qwen": FakeBackend()}
        self.metrics = {
            "loads": 5,
            "stops": 2,
            "load_errors": 1,
            "last_load_at": 1234.0,
            "last_error": None,
        }

    def all_models(self):
        return [self.model]

    def all_audio_models(self):
        return list(self.cfg.audio_models)

    async def ensure_active(self, query, *, acquire: bool = False):
        if query not in {"qwen", "qwen.gguf"}:
            raise KeyError(query)
        # Use a model from cfg if it exists there, so the request path sees
        # the same object the Autotuner is watching.
        for m in self.cfg.models:
            if m.name == "qwen":
                if acquire:
                    self.acquire_model(m.name)
                return m, FakeBackend()
        if acquire:
            self.acquire_model(self.model.name)
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
        return FakeAsyncClientUpstream(), resp

    def upstreams_status(self):
        return [
            {
                "name": "ollama",
                "url": "http://127.0.0.1:11434",
                "model_count": 1,
                "last_fetch": 123.0,
            }
        ]


class FakeAsyncClientUpstream:
    """httpx.AsyncClient that simulates upstream proxy responses."""

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def build_request(self, method, url, content=None, headers=None):
        return {"method": method, "url": url, "content": content, "headers": headers}

    async def send(self, request, stream=False):
        resp = FakeUpstreamResponse()
        return resp

    async def aclose(self):
        self.closed = True


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
        self.last_client = None
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
        self.last_client = FakeAsyncClientUpstream()
        return self.last_client, self.last_stream

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
    app = create_app(Config())

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
    app = create_app(Config())

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
            json={
                "model": "llama3.1",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: upstream chunk 1\n\ndata: upstream chunk 2\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")


def test_health_includes_loaded_models_and_uptime(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app()

    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["loaded_model_count"] == 1
    assert "qwen" in data["loaded_models"]
    assert data["uptime_seconds"] >= 0


def test_session_token_served_to_loopback_peer(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    app = create_app(cfg)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/admin/session-token")
    assert r.status_code == 200
    assert r.json()["admin_token"] == "secret"


def test_session_token_refused_for_non_loopback_peer(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token="secret"))
    app = create_app(cfg)

    with TestClient(app, client=("192.168.1.50", 12345)) as client:
        r = client.get("/admin/session-token")
    assert r.status_code == 403


def test_admin_metrics_returns_counters_and_gpus(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    cfg = Config(gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage")])
    app = create_app(cfg)

    with TestClient(app) as client:
        r = client.get("/admin/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["loads"] == 5
    assert data["stops"] == 2
    assert data["load_errors"] == 1
    assert data["active_models"] == ["qwen"]
    assert any(g["pci_slot"] == "0000:03:00.0" for g in data["gpus"])


class CapturingMCPClientManager:
    started_servers = []

    def __init__(self, servers):
        self.servers = servers

    async def start(self):
        CapturingMCPClientManager.started_servers = list(self.servers)

    async def stop(self):
        pass


def test_server_lifespan_uses_active_profile_mcp_servers(monkeypatch):
    import arc_llama.server as server_mod
    from arc_llama.config import Config

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod, "MCPClientManager", CapturingMCPClientManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)

    cfg = Config()
    cfg.mcp_servers = [
        MCPServerConfig(name="fs", command="npx"),
        MCPServerConfig(name="gh", command="npx"),
    ]
    cfg.profiles = [ProfileConfig(name="work", mcp_servers=["fs"])]
    cfg.agent.profile = "work"

    app = create_app(cfg)
    with TestClient(app):
        pass

    assert [s.name for s in CapturingMCPClientManager.started_servers] == ["fs"]


class AgentFakeAsyncClient:
    """httpx.AsyncClient stand-in that lets /v1/agent complete without a real LLM."""

    def __init__(self, timeout=None, base_url=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "done"}}]
        }
        return resp

    async def aclose(self):
        pass


def _app_with_admin_token(monkeypatch, token: str):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", AgentFakeAsyncClient)
    cfg = Config(server=ServerConfig(admin_token=token))
    return create_app(cfg)


def test_admin_status_requires_token_when_configured(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.get("/admin/status").status_code == 401
        assert (
            client.get("/admin/status", headers={"Authorization": "Bearer wrong"}).status_code
            == 403
        )
        assert (
            client.get("/admin/status", headers={"Authorization": "Bearer secret"}).status_code
            == 200
        )


def test_admin_load_requires_token_when_configured(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/admin/load/qwen").status_code == 401
        assert (
            client.post("/admin/load/qwen", headers={"Authorization": "Bearer secret"}).status_code
            == 200
        )


def test_agent_auto_confirm_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        r = client.post(
            "/v1/agent",
            json={
                "model": "qwen",
                "task": "hello",
                "auto_confirm": True,
            },
        )
        assert r.status_code == 401

        r = client.post(
            "/v1/agent",
            json={
                "model": "qwen",
                "task": "hello",
                "auto_confirm": True,
            },
            headers={"Authorization": "Bearer secret"},
        )
        assert r.status_code == 200


def test_agent_without_auto_confirm_allows_unauthenticated_request(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        r = client.post(
            "/v1/agent",
            json={
                "model": "qwen",
                "task": "hello",
                "auto_confirm": False,
            },
        )
        assert r.status_code == 200


def test_agent_confirm_endpoint_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/v1/agent/run-1/confirm", json={"approved": True}).status_code == 401
        assert (
            client.post(
                "/v1/agent/run-1/confirm",
                json={"approved": True},
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 404
        )  # run not found, but auth passed


def test_agent_plan_endpoint_requires_admin_token(monkeypatch):
    app = _app_with_admin_token(monkeypatch, "secret")

    with TestClient(app) as client:
        assert client.post("/v1/agent/run-1/plan", json={"approved": True}).status_code == 401
        assert (
            client.post(
                "/v1/agent/run-1/plan",
                json={"approved": True},
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 404
        )  # run not found, but auth passed


# ---------------------------------------------------------------------------
# /admin/models/{name}/edit — perf recipe fields
# ---------------------------------------------------------------------------


class FakeRouterWithRebuild(FakeRouter):
    def _build_servers(self):
        pass

    async def rebuild_model(self, name):
        return True, False


def _edit_app(monkeypatch, tmp_path):
    import arc_llama.server as server_mod

    # Keep cfg.save() away from the real ~/.config.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    cfg = Config(
        server=ServerConfig(admin_token=None),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[
            ModelConfig(
                name="qwen",
                path="/models/qwen.gguf",
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": 8192},
            )
        ],
    )
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


def test_admin_edit_persists_to_custom_config_path(monkeypatch, tmp_path):
    """Regression: /admin/models/{name}/edit must honour --config."""
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    custom_path = tmp_path / "custom.toml"
    default_path = tmp_path / "default" / "config.toml"
    monkeypatch.setattr("arc_llama.config.default_config_path", lambda: default_path)

    cfg = Config(
        server=ServerConfig(admin_token=None),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[
            ModelConfig(
                name="qwen",
                path="/models/qwen.gguf",
                port=18080,
                gpu_pci_slot="0000:03:00.0",
                recipe={"ctx": 8192},
            )
        ],
        tune=TuneConfig(auto=False),
    )
    app = create_app(cfg, config_path=custom_path)

    with TestClient(app) as client:
        response = client.post("/admin/models/qwen/edit", json={"ctx": 4096})

    assert response.status_code == 200
    assert custom_path.exists()
    assert not default_path.exists()
    model = next(m for m in cfg.models if m.name == "qwen")
    assert model.recipe["ctx"] == 4096


def test_admin_scan_persists_to_custom_config_path(monkeypatch, tmp_path):
    """Regression: /admin/scan must honour --config when persisting discoveries."""
    import arc_llama.server as server_mod
    from arc_llama import models as models_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouterWithRebuild)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    custom_path = tmp_path / "custom.toml"
    default_path = tmp_path / "default" / "config.toml"
    monkeypatch.setattr("arc_llama.config.default_config_path", lambda: default_path)

    cfg = Config(
        server=ServerConfig(admin_token=None),
        paths=PathsConfig(models_dir=str(tmp_path / "models")),
        gpus=[GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage", vram_mb=24576)],
        models=[],
        tune=TuneConfig(auto=False),
    )
    app = create_app(cfg, config_path=custom_path)

    fake_model = ModelConfig(
        name="new",
        path=str(tmp_path / "new.gguf"),
        port=18081,
        gpu_pci_slot="0000:03:00.0",
    )

    monkeypatch.setattr(models_mod, "discover_ggufs", lambda c: [fake_model])

    def fake_register(c, found):
        c.models.extend(found)
        return found

    monkeypatch.setattr(models_mod, "register_discovered", fake_register)

    with TestClient(app) as client:
        response = client.post("/admin/scan")

    assert response.status_code == 200
    assert custom_path.exists()
    assert not default_path.exists()
    assert any(m.name == "new" for m in cfg.models)


def test_local_request_bumps_tuner_use_count(monkeypatch, tmp_path):
    """A real /v1/chat/completions request must call Autotuner.bump_use."""
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)

    cfg = Config(
        server=ServerConfig(admin_token=None),
        paths=PathsConfig(models_dir=str(tmp_path / "models")),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24 * 1024,
            ),
        ],
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "qwen.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
            ),
        ],
        # Keep auto=False so the real lifespan does not spin up a background
        # tuner and overwrite our fake one.
        tune=TuneConfig(auto=False, idle_seconds=120, min_uses=1),
    )

    bumped: list[str] = []

    class FakeTuner:
        def bump_use(self, name: str) -> None:
            bumped.append(name)

        async def stop(self) -> None:
            pass

    app = create_app(cfg)
    # Bypass lifespan so our fake tuner survives and does not need a full loop.
    app.state.tuner = FakeTuner()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert "qwen" in bumped


def test_custom_config_path_used_by_autotune_save(monkeypatch, tmp_path):
    """create_app(config_path=...) must make the autotuner save to that path."""
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)

    custom_path = tmp_path / "custom.toml"
    cfg = Config(
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24 * 1024,
            ),
        ],
        models=[
            ModelConfig(
                name="qwen",
                path=str(tmp_path / "qwen.gguf"),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
            ),
        ],
        tune=TuneConfig(auto=True, idle_seconds=120, min_uses=1),
    )

    app = create_app(cfg, config_path=custom_path)

    saved = []

    class FakeTuner:
        def __init__(self, on_save):
            self.on_save = on_save

        async def stop(self) -> None:
            pass

    import arc_llama.autotune as autotune_mod

    async def fake_start_autotuner(cfg, router, *, version, on_save=None):
        saved.append(on_save)
        return FakeTuner(on_save)

    monkeypatch.setattr(autotune_mod, "start_autotuner", fake_start_autotuner)

    # Trigger lifespan to wire the tuner. The fake tuner has a no-op stop()
    # so shutdown does not fail.
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert saved
    on_save = saved[0]
    assert on_save is not None
    # Invoking on_save should write to the custom path.
    on_save()
    assert custom_path.exists()


def test_admin_tune_abort_uses_public_method(monkeypatch):
    """DELETE /admin/tune must call Autotuner.abort_sweep(), not _abort_event."""
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)

    calls: list[bool] = []

    class FakeTuner:
        running_model = "qwen"

        def abort_sweep(self) -> bool:
            calls.append(True)
            return True

        async def stop(self) -> None:
            pass

    cfg = Config(
        server=ServerConfig(admin_token="secret"),
        gpus=[
            GPUConfig(
                pci_slot="0000:03:00.0",
                sycl_index=0,
                arch="battlemage",
                vram_mb=24 * 1024,
            ),
        ],
        models=[
            ModelConfig(
                name="qwen",
                path="/models/qwen.gguf",
                port=18080,
                gpu_pci_slot="0000:03:00.0",
            ),
        ],
        tune=TuneConfig(auto=False),
    )
    app = create_app(cfg)
    app.state.tuner = FakeTuner()

    with TestClient(app) as client:
        response = client.delete("/admin/tune", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["aborted"] is True
    assert calls


# ---------------------------------------------------------------------------
# inflight must span the whole request lifetime (AUTOTUNE_ROUND3 defect A)
#
# These drive the real app through TestClient against a fake backend whose
# response is slow, and assert router.inflight > 0 while the response is
# still being produced. Calling ensure_active directly would prove nothing:
# the defect shipped precisely because the counter only covered ensure_active.
# ---------------------------------------------------------------------------


def _wait_drained(router, timeout: float = 5.0) -> None:
    import time as _time

    deadline = _time.monotonic() + timeout
    while router.inflight != 0 and _time.monotonic() < deadline:
        _time.sleep(0.01)


def test_inflight_covers_non_streaming_generation(monkeypatch):
    import threading

    import arc_llama.server as server_mod

    entered = threading.Event()
    release = threading.Event()

    class SlowClient:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, content=None, headers=None):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return FakeResponse()

        async def aclose(self):
            pass

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", SlowClient)
    app = create_app(Config(tune=TuneConfig(auto=False)))

    with TestClient(app) as client:
        result: dict = {}

        def do_request():
            result["r"] = client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )

        t = threading.Thread(target=do_request)
        t.start()
        try:
            # The fake backend has the request and is "generating": the
            # request holds the GPU, so the counter must be up.
            assert entered.wait(timeout=5)
            assert app.state.router.inflight > 0
        finally:
            release.set()
            t.join(timeout=10)

    assert result["r"].status_code == 200
    _wait_drained(app.state.router)
    assert app.state.router.inflight == 0


def test_inflight_covers_streaming_generation(monkeypatch):
    import threading

    import arc_llama.server as server_mod

    entered = threading.Event()
    release = threading.Event()

    class SlowStream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        closed = False

        async def aiter_raw(self):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            yield b"data: one\n\n"

        async def aclose(self):
            self.closed = True

    class SlowStreamClient:
        def __init__(self, timeout=None):
            pass

        def build_request(self, method, url, content=None, headers=None):
            return {"method": method, "url": url, "content": content, "headers": headers}

        async def send(self, request, stream=False):
            assert stream is True
            return SlowStream()

        async def aclose(self):
            pass

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", SlowStreamClient)
    app = create_app(Config(tune=TuneConfig(auto=False)))

    with TestClient(app) as client:
        result: dict = {}

        def do_request():
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "qwen",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as r:
                result["status"] = r.status_code
                result["body"] = b"".join(r.iter_bytes())

        t = threading.Thread(target=do_request)
        t.start()
        try:
            # The streamed body has started but not finished: still inflight.
            assert entered.wait(timeout=5)
            assert app.state.router.inflight > 0
        finally:
            release.set()
            t.join(timeout=10)

    assert result["status"] == 200
    assert result["body"] == b"data: one\n\n"
    # The decrement lives in the close_upstream BackgroundTask, which runs
    # after the body is fully sent.
    _wait_drained(app.state.router)
    assert app.state.router.inflight == 0


def test_inflight_decremented_when_load_fails(monkeypatch):
    """A failed load must not leak the in-flight count."""
    import arc_llama.server as server_mod

    class FailingRouter(FakeRouter):
        async def ensure_active(self, query, *, acquire: bool = False):
            raise RuntimeError("llama-server did not become healthy")

    monkeypatch.setattr(server_mod, "Router", FailingRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app(Config(tune=TuneConfig(auto=False)))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 503
    _wait_drained(app.state.router)
    assert app.state.router.inflight == 0


class ColdStartBackend(FakeBackend):
    """Process alive, health check not yet passed — a cold start in progress."""

    ready = False


def test_health_does_not_report_cold_start_as_loaded(monkeypatch):
    """A subprocess that exists but has not passed its health check is not
    'loaded': during a cold start or a crash-respawn the port is not serving,
    and dashboards or scripts gating on this field would act on a lie."""
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    # Explicit token-less config: the autouse isolated-config fixture generates
    # an admin token, which would 401 the /admin/status call below.
    app = create_app(Config(server=ServerConfig(admin_token=None)))

    with TestClient(app) as client:
        app.state.router._servers["qwen"] = ColdStartBackend()
        health = client.get("/health").json()
        assert health["loaded_models"] == [], "cold-starting model reported as loaded"
        status = client.get("/admin/status").json()
        entry = next(m for m in status["models"] if m["name"] == "qwen")
        assert entry["loaded"] is False
