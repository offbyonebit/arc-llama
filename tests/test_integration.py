"""Tests for the Connect-a-frontend integration flow.

Covers the pure payload builders in arc_llama.integration, the loopback
Ollama probe, and the read-only /admin/integration endpoint: it must
reflect the configured host/port, expose no credentials, and never mutate
registered upstreams.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from helpers import ConfigRouter as FakeRouter
from httpx import AsyncClient as _HttpxAsyncClient

from arc_llama.config import Config, ModelConfig, ServerConfig, UpstreamConfig
from arc_llama.integration import (
    DEFAULT_OLLAMA_URL,
    client_host,
    format_base_url,
    integration_payload,
    probe_ollama,
)
from arc_llama.server import create_app


def _cfg(
    gguf: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 11437,
    admin_token: str | None = None,
    upstreams: list[UpstreamConfig] | None = None,
) -> Config:
    return Config(
        server=ServerConfig(host=host, port=port, admin_token=admin_token),
        models=[
            ModelConfig(
                name="qwen",
                path=str(gguf),
                port=18080,
                gpu_pci_slot="0000:03:00.0",
            )
        ],
        upstreams=upstreams or [],
    )


class FakeUpstreamManager:
    def __init__(self, upstreams):
        self.upstreams = upstreams

    async def models(self):
        return []

    def find_model(self, model_id):
        return None

    def upstreams_status(self):
        return [
            {"name": u.name, "url": u.url, "model_count": 0, "last_fetch": None}
            for u in self.upstreams
        ]


def _fake_ollama_unreachable(**kwargs):
    class FakeAsyncContext:
        async def __aenter__(self):
            class Client:
                async def get(self, path):
                    raise ConnectionError("refused")

            return Client()

        async def __aexit__(self, *exc):
            return None

    return FakeAsyncContext()


def _fake_ollama_ok(version):
    def factory(**kwargs):
        response = type(
            "Response",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self: {"version": version}},
        )()

        class FakeAsyncContext:
            async def __aenter__(self):
                class Client:
                    async def get(self, path):
                        return response

                return Client()

            async def __aexit__(self, *exc):
                return None

        return FakeAsyncContext()

    return factory


class TestClientHost:
    def test_loopback_passes_through(self):
        assert client_host("127.0.0.1") == "127.0.0.1"

    def test_lan_host_passes_through(self):
        assert client_host("192.168.1.10") == "192.168.1.10"

    def test_ipv6_host_is_matched_exactly(self):
        assert client_host("::1") == "::1"

    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
    def test_wildcard_binds_become_loopback(self, wildcard):
        assert client_host(wildcard) == "127.0.0.1"


class TestFormatBaseUrl:
    def test_http(self):
        assert format_base_url("127.0.0.1", 11437) == "http://127.0.0.1:11437/v1"

    def test_ipv6_is_bracketed(self):
        assert format_base_url("::1", 11437) == "http://[::1]:11437/v1"

    def test_lan_host(self):
        assert format_base_url("192.168.1.50", 8000) == "http://192.168.1.50:8000/v1"


class TestProbeOllama:
    async def test_unreachable_ollama_reports_not_reachable(self, monkeypatch):
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_unreachable)
        result = await probe_ollama()
        assert result == {"url": DEFAULT_OLLAMA_URL, "reachable": False, "version": None}

    async def test_reachable_ollama_reports_version(self, monkeypatch):
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_ok("0.5.2"))
        result = await probe_ollama()
        assert result == {
            "url": DEFAULT_OLLAMA_URL,
            "reachable": True,
            "version": "0.5.2",
        }

    async def test_ollama_without_version_still_reachable(self, monkeypatch):
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_ok(None))
        result = await probe_ollama()
        assert result["reachable"] is True
        assert result["version"] is None


def integration_module():
    import arc_llama.integration as module

    return module


class TestIntegrationPayload:
    def _base_fields(self, **kwargs):
        gguf = kwargs.pop("_gguf")
        payload = integration_payload(_cfg(gguf, **kwargs), ollama={"reachable": False})
        return payload

    def test_base_url_reflects_configured_host_and_port(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", host="127.0.0.1", port=11437)
        assert payload["base_url"] == "http://127.0.0.1:11437/v1"
        assert payload["server"]["port"] == 11437
        assert payload["server"]["bind_all"] is False

    def test_wildcard_bind_reports_loopback_and_lan_note(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", host="0.0.0.0", port=11437)
        assert payload["base_url"] == "http://127.0.0.1:11437/v1"
        assert payload["server"]["configured_host"] == "0.0.0.0"
        assert payload["server"]["bind_all"] is True
        assert payload["lan_note"]
        assert "LAN" in payload["lan_note"] or "replace" in payload["lan_note"].lower()

    def test_loopback_bind_has_no_lan_note(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", host="127.0.0.1")
        assert payload["lan_note"] is None

    def test_no_credentials_in_payload(self, tmp_path: Path):
        cfg = _cfg(tmp_path / "m.gguf", admin_token="super-secret-token")
        payload = integration_payload(cfg, ollama={"reachable": False})
        serialized = repr(payload)
        assert "super-secret-token" not in serialized
        assert "admin_token" not in serialized

    def test_payload_exposes_api_key_guidance_not_a_key(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf")
        guidance = payload["api_key_guidance"]
        assert "On localhost" in guidance and "any non-empty" in guidance
        assert "For LAN callers" in guidance
        assert "arc-llama keys create <name>" in guidance
        assert "validated once any key exists" in guidance
        assert "arc-llama serve --lan" in guidance
        assert "does not validate" not in guidance

    def test_upstreams_listed_without_change(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
        cfg = _cfg(tmp_path / "m.gguf", upstreams=upstreams)
        payload = integration_payload(cfg, ollama={"reachable": False})
        assert payload["upstreams"] == [{"name": "lmstudio", "url": "http://127.0.0.1:1234"}]
        assert cfg.upstreams == upstreams

    def test_ollama_upstream_command_is_copyable(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf")
        assert payload["ollama"]["upstream_add_command"] == (
            f"arc-llama upstream add ollama {DEFAULT_OLLAMA_URL}"
        )

    def test_registered_ollama_upstream_by_name_is_detected(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="ollama", url="http://127.0.0.1:11434")]
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", upstreams=upstreams)
        assert payload["ollama"]["already_registered"] is True

    def test_registered_ollama_upstream_by_url_is_detected(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="my-ollama", url="http://127.0.0.1:11434/")]
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", upstreams=upstreams)
        assert payload["ollama"]["already_registered"] is True

    def test_other_upstreams_do_not_mark_ollama_registered(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
        payload = self._base_fields(_gguf=tmp_path / "m.gguf", upstreams=upstreams)
        assert payload["ollama"]["already_registered"] is False

    def test_curl_examples_are_platform_specific(self, tmp_path: Path):
        payload = self._base_fields(_gguf=tmp_path / "m.gguf")
        assert payload["base_url"] in payload["curl_example"]
        assert payload["base_url"] in payload["curl_example_windows"]
        assert payload["curl_example"] != payload["curl_example_windows"]
        assert '""' in payload["curl_example_windows"] or '"' in payload["curl_example_windows"]

    def test_empty_registry_falls_back_to_model_placeholder(self, tmp_path: Path):
        cfg = Config(server=ServerConfig(host="127.0.0.1", port=11437))
        payload = integration_payload(cfg, ollama={"reachable": False})
        assert payload["model_id"] == "MODEL_ID"


def _make_test_app(cfg: Config):
    import arc_llama.server as server_mod

    with (
        patch.object(server_mod, "Router", FakeRouter),
        patch.object(server_mod, "UpstreamManager", FakeUpstreamManager),
    ):
        return create_app(cfg, plugins=[])


class TestAdminIntegrationEndpoint:
    def _probe_stub(self, monkeypatch):
        """Point the loopback probe at an unreachable fake so offline runs are deterministic."""
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_unreachable)

    async def test_endpoint_reports_configured_server(self, tmp_path: Path, monkeypatch):
        import httpx

        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", port=11437))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/admin/integration")
        assert response.status_code == 200
        data = response.json()
        assert data["base_url"] == "http://127.0.0.1:11437/v1"
        assert data["server"]["port"] == 11437
        assert data["ollama"]["reachable"] is False
        assert data["ollama"]["upstream_add_command"].startswith("arc-llama upstream add")

    async def test_endpoint_requires_admin_token_when_configured(self, tmp_path: Path, monkeypatch):
        import httpx

        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token="sekrit"))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/admin/integration")).status_code == 401
                ok = await client.get(
                    "/admin/integration", headers={"Authorization": "Bearer sekrit"}
                )
                assert ok.status_code == 200

    async def test_endpoint_without_token_is_open_like_other_admin_reads(
        self, tmp_path: Path, monkeypatch
    ):
        import httpx

        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token=None))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/admin/integration")).status_code == 200

    async def test_endpoint_leaks_no_credentials(self, tmp_path: Path, monkeypatch):
        import httpx

        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token="super-secret-token"))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                data = (
                    await client.get(
                        "/admin/integration",
                        headers={"Authorization": "Bearer super-secret-token"},
                    )
                ).json()
        assert "super-secret-token" not in repr(data)

    async def test_endpoint_lists_registered_upstreams(self, tmp_path: Path, monkeypatch):
        import httpx

        self._probe_stub(monkeypatch)
        upstreams = [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
        app = _make_test_app(_cfg(tmp_path / "m.gguf", upstreams=upstreams))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                data = (await client.get("/admin/integration")).json()
        assert data["upstreams"] == [{"name": "lmstudio", "url": "http://127.0.0.1:1234"}]
        assert data["ollama"]["already_registered"] is False

    async def test_reachable_ollama_reported_in_endpoint(self, tmp_path: Path, monkeypatch):
        import httpx

        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_ok("0.6.1"))
        app = _make_test_app(_cfg(tmp_path / "m.gguf"))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with _HttpxAsyncClient(transport=transport, base_url="http://test") as client:
                data = (await client.get("/admin/integration")).json()
        assert data["ollama"]["reachable"] is True
        assert data["ollama"]["version"] == "0.6.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.20", "fd7a:115c:a1e0::123"])
def test_all_frontend_recipes_use_loaded_model_url_and_placeholder(tmp_path, host):
    cfg = _cfg(tmp_path / "first.gguf", host=host, admin_token="never-copy-admin")
    loaded = ModelConfig("loaded-second", str(tmp_path / "second.gguf"), 18081, "gpu", recipe={"ctx": 8192})
    cfg.models.append(loaded)
    payload = integration_payload(cfg, loaded_model=loaded)
    assert payload["model_id"] == "loaded-second"
    assert "loaded" in payload["recipe_note"]
    recipes = payload["recipes"]
    assert [recipe["name"] for recipe in recipes] == ["Continue", "Cline", "Zed", "SillyTavern", "AnythingLLM", "Jan"]
    for recipe in recipes:
        assert payload["base_url"] in recipe["snippet"]
        assert "loaded-second" in recipe["snippet"] and "YOUR_ARC_API_KEY" in recipe["snippet"]
        assert "never-copy-admin" not in repr(recipe)
        assert recipe["docs_url"].startswith("https://") and recipe["steps"]
    zed = next(recipe for recipe in recipes if recipe["id"] == "zed")
    import json

    zed_cfg = json.loads(zed["snippet"].split("\n", 1)[1])
    provider = zed_cfg["language_models"]["openai_compatible"]["arc-llama"]
    assert provider["available_models"][0]["max_tokens"] == 8192
    assert provider["available_models"][0]["capabilities"]["max_tokens_parameter"] is True
    assert {"tools", "images", "parallel_tool_calls", "prompt_cache_key"} <= provider["available_models"][0]["capabilities"].keys()
    assert "api_key" not in provider
    assert "provider UI/keychain" in " ".join(zed["steps"])


def test_wildcard_recipes_use_lan_request_ip_without_dns(tmp_path):
    cfg = _cfg(tmp_path / "m.gguf", host="0.0.0.0")
    payload = integration_payload(cfg, request_host="192.168.1.50")
    assert payload["base_url"] == "http://192.168.1.50:11437/v1"
    assert all(payload["base_url"] in recipe["snippet"] for recipe in payload["recipes"])
    assert integration_payload(cfg, request_host="0.0.0.0")["base_url"] == "http://127.0.0.1:11437/v1"
    assert integration_payload(cfg, request_host='bad-host\"\nunsafe')["base_url"] == "http://127.0.0.1:11437/v1"
    assert integration_payload(cfg, request_host="arc-host.local")["base_url"] == "http://arc-host.local:11437/v1"
    cfg.server.host = "192.168.1.8"
    assert integration_payload(cfg, request_host="192.168.1.50")["base_url"] == "http://192.168.1.8:11437/v1"


def test_recipe_model_fallback_and_safe_yaml_strings(tmp_path):
    import json

    cfg = _cfg(tmp_path / "m.gguf")
    cfg.models[0].name = 'model: "quoted"\nline'
    payload = integration_payload(cfg)
    assert "No chat model is loaded" in payload["recipe_note"]
    yaml = payload["recipes"][0]["snippet"]
    assert json.dumps(cfg.models[0].name) in yaml
    zed = payload["recipes"][2]
    assert zed["format"] == "text" and "Context window: fill" in zed["snippet"]
    empty = integration_payload(Config())
    assert "replace MODEL_ID" in empty["recipe_note"]


async def test_endpoint_prefers_ready_chat_model_and_lan_host(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import httpx

    import arc_llama.server as server_mod

    monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_unreachable)
    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    cfg = _cfg(tmp_path / "first.gguf", host="0.0.0.0")
    cfg.tune.auto = False
    cfg.paths.state_dir = str(tmp_path / "state")
    loaded = ModelConfig("current-chat", str(tmp_path / "loaded.gguf"), 18081, "gpu")
    cfg.models.append(loaded)
    app = _make_test_app(cfg)
    async with app.router.lifespan_context(app):
        await app.state.started.wait()
        app.state.router._servers = {"qwen": SimpleNamespace(is_running=True, ready=False),
                                     "current-chat": SimpleNamespace(is_running=True, ready=True)}
        async with _HttpxAsyncClient(transport=httpx.ASGITransport(app), base_url="http://192.168.1.50:11437") as client:
            data = (await client.get("/admin/integration")).json()
            assert data["model_id"] == "current-chat"
            assert data["base_url"] == "http://192.168.1.50:11437/v1"
            assert all("current-chat" in recipe["snippet"] for recipe in data["recipes"])


def test_recipe_fallback_skips_reranker_models(tmp_path):
    cfg = _cfg(tmp_path / "rerank.gguf")
    cfg.models[0].recipe = {"reranking": True}
    assert integration_payload(cfg)["model_id"] == "MODEL_ID"
    cfg.models.append(ModelConfig("chat-model", str(tmp_path / "chat.gguf"), 18081, "gpu"))
    assert integration_payload(cfg)["model_id"] == "chat-model"
