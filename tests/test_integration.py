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
from fastapi.testclient import TestClient

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


class FakeRouter:
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self._servers = {}

    def all_models(self):
        return list(self.cfg.models)

    async def shutdown(self):
        return None


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
        assert "non-empty" in payload["api_key_guidance"]
        assert "key" not in payload["api_key_guidance"].lower() or "valid" in payload["api_key_guidance"].lower()

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
        payload = self._base_fields(
            _gguf=tmp_path / "m.gguf", upstreams=upstreams
        )
        assert payload["ollama"]["already_registered"] is True

    def test_registered_ollama_upstream_by_url_is_detected(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="my-ollama", url="http://127.0.0.1:11434/")]
        payload = self._base_fields(
            _gguf=tmp_path / "m.gguf", upstreams=upstreams
        )
        assert payload["ollama"]["already_registered"] is True

    def test_other_upstreams_do_not_mark_ollama_registered(self, tmp_path: Path):
        upstreams = [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
        payload = self._base_fields(
            _gguf=tmp_path / "m.gguf", upstreams=upstreams
        )
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

    with patch.object(server_mod, "Router", FakeRouter), patch.object(
        server_mod, "UpstreamManager", FakeUpstreamManager
    ):
        return create_app(cfg, plugins=[])


class TestAdminIntegrationEndpoint:
    def _probe_stub(self, monkeypatch):
        """Point the loopback probe at an unreachable fake so offline runs are deterministic."""
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_unreachable)

    def test_endpoint_reports_configured_server(self, tmp_path: Path, monkeypatch):
        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", port=11437))
        with TestClient(app) as client:
            response = client.get("/admin/integration")
        assert response.status_code == 200
        data = response.json()
        assert data["base_url"] == "http://127.0.0.1:11437/v1"
        assert data["server"]["port"] == 11437
        assert data["ollama"]["reachable"] is False
        assert data["ollama"]["upstream_add_command"].startswith("arc-llama upstream add")

    def test_endpoint_requires_admin_token_when_configured(self, tmp_path: Path, monkeypatch):
        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token="sekrit"))
        with TestClient(app) as client:
            assert client.get("/admin/integration").status_code == 401
            ok = client.get(
                "/admin/integration", headers={"Authorization": "Bearer sekrit"}
            )
            assert ok.status_code == 200

    def test_endpoint_without_token_is_open_like_other_admin_reads(self, tmp_path: Path, monkeypatch):
        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token=None))
        with TestClient(app) as client:
            assert client.get("/admin/integration").status_code == 200

    def test_endpoint_leaks_no_credentials(self, tmp_path: Path, monkeypatch):
        self._probe_stub(monkeypatch)
        app = _make_test_app(_cfg(tmp_path / "m.gguf", admin_token="super-secret-token"))
        with TestClient(app) as client:
            data = client.get(
                "/admin/integration", headers={"Authorization": "Bearer super-secret-token"}
            ).json()
        assert "super-secret-token" not in repr(data)

    def test_endpoint_lists_registered_upstreams(self, tmp_path: Path, monkeypatch):
        self._probe_stub(monkeypatch)
        upstreams = [UpstreamConfig(name="lmstudio", url="http://127.0.0.1:1234")]
        app = _make_test_app(_cfg(tmp_path / "m.gguf", upstreams=upstreams))
        with TestClient(app) as client:
            data = client.get("/admin/integration").json()
        assert data["upstreams"] == [{"name": "lmstudio", "url": "http://127.0.0.1:1234"}]
        assert data["ollama"]["already_registered"] is False

    def test_reachable_ollama_reported_in_endpoint(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("arc_llama.integration.httpx.AsyncClient", _fake_ollama_ok("0.6.1"))
        app = _make_test_app(_cfg(tmp_path / "m.gguf"))
        with TestClient(app) as client:
            data = client.get("/admin/integration").json()
        assert data["ollama"]["reachable"] is True
        assert data["ollama"]["version"] == "0.6.1"
