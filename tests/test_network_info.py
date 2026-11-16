from __future__ import annotations

import json
import socket
import struct
from types import SimpleNamespace

import httpx
import pytest
from helpers import ConfigRouter

from arc_llama import network_info as network
from arc_llama.config import Config, ServerConfig, TuneConfig

ADDRESSES = [("lo", "127.0.0.1"), ("Ethernet", "192.168.1.8"),
             ("Wi-Fi", "192.168.1.9"), ("tailscale0", "100.101.102.103"),
             ("tailscale0", "fd7a:115c:a1e0::123"), ("lo", "::1"), ("eth0", "fe80::1")]


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", []), ("::1", []), ("localhost", []),
    ("192.168.1.8", ["http://192.168.1.8:12345/chat"]),
    ("0.0.0.0", ["http://192.168.1.8:12345/chat", "http://192.168.1.9:12345/chat", "http://100.101.102.103:12345/chat"]),
    ("::", ["http://[fd7a:115c:a1e0::123]:12345/chat"]),
])
def test_addresses_match_bound_interface_and_family(host, expected):
    payload = network.network_payload(ServerConfig(host=host, port=12345, admin_token="secret"), ADDRESSES)
    assert [item["url"] for item in payload["addresses"]] == expected
    assert "secret" not in json.dumps(payload)
    if host in {"127.0.0.1", "::1", "localhost"}:
        assert payload["enabled"] is False and "--lan" in payload["hint"]
    if host == "0.0.0.0":
        assert payload["addresses"][-1]["kind"] == "tailscale"


def test_unavailable_or_invalid_addresses_are_quiet_and_unique():
    server = ServerConfig(host="192.168.1.8")
    assert network.network_payload(server, []) == {"enabled": True, "addresses": [], "hint": "No address was found for this bind interface. Check its local adapter configuration."}
    values = [("eth", "invalid"), ("eth", "224.0.0.1"), ("eth", "192.168.1.8"), ("alias", "192.168.1.8")]
    assert len(network.network_payload(server, values)["addresses"]) == 1


def test_windows_reads_adapter_configuration_without_network_calls(monkeypatch):
    monkeypatch.setattr(network.sys, "platform", "win32")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=json.dumps([
            {"InterfaceAlias": "Ethernet", "IPAddress": "192.168.1.8"},
            {"InterfaceAlias": "Tailscale", "IPAddress": "100.101.102.103"},
        ]).encode())

    monkeypatch.setattr(network.subprocess, "run", run)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS must not be used"))
    assert network.local_addresses() == [("Ethernet", "192.168.1.8"), ("Tailscale", "100.101.102.103")]
    argv, kwargs = calls[0]
    assert argv[:4] == ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive"]
    assert "Get-NetIPAddress" in argv[-1]
    assert kwargs["timeout"] == 3 and kwargs["check"] is True


@pytest.mark.skipif(network.sys.platform == "win32", reason="Linux interface ioctl")
def test_linux_reads_local_ioctl_and_proc_without_connect_or_dns(monkeypatch):
    import fcntl

    monkeypatch.setattr(network.sys, "platform", "linux")

    class LocalSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def fileno(self):
            return 42

        def connect(self, *args):
            pytest.fail("A route probe must not be used")

    monkeypatch.setattr(socket, "socket", lambda *args: LocalSocket())
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS must not be used"))
    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "tailscale0")])
    monkeypatch.setattr(fcntl, "ioctl", lambda fd, command, data: b"\0" * 20 + socket.inet_aton("100.101.102.103"))
    monkeypatch.setattr(network.Path, "read_text", lambda _: "fd7a115ca1e000000000000000000123 01 40 00 00 tailscale0\n")
    assert network.local_addresses() == [("tailscale0", "100.101.102.103"), ("tailscale0", "fd7a:115c:a1e0::123")]


@pytest.mark.parametrize("output", [b"not json", b"null", b"{}"])
def test_bad_windows_adapter_output_is_quiet(monkeypatch, output):
    monkeypatch.setattr(network.sys, "platform", "win32")
    monkeypatch.setattr(network.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output))
    assert network.local_addresses() == []


async def test_network_endpoint_auth_no_keys_and_manifest(tmp_path, monkeypatch):
    from test_server import FakeUpstreamManager

    import arc_llama.server as server

    monkeypatch.setattr(server, "Router", ConfigRouter)
    monkeypatch.setattr(server, "UpstreamManager", FakeUpstreamManager)
    calls = []
    monkeypatch.setattr(network, "local_addresses", lambda: calls.append(True) or ADDRESSES)
    cfg = Config(server=ServerConfig(host="0.0.0.0", admin_token="private-admin-value"), tune=TuneConfig(auto=False))
    cfg.paths.state_dir = str(tmp_path)
    app = server.create_app(cfg, plugins=[])
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            assert (await client.get("/admin/network")).status_code == 401
            response = await client.get("/admin/network", headers={"Authorization": "Bearer private-admin-value"})
            assert response.status_code == 200 and len(response.json()["addresses"]) == 3
            assert "private-admin-value" not in response.text
            for item in response.json()["addresses"]:
                assert "?" not in item["url"] and "#" not in item["url"] and "@" not in item["url"]
            cfg.server.host = "127.0.0.1"
            result = await client.get("/admin/network", headers={"Authorization": "Bearer private-admin-value"})
            assert result.json()["addresses"] == [] and len(calls) == 1
            manifest = await client.get("/manifest.webmanifest")
            assert manifest.headers["content-type"].startswith("application/manifest+json")
            assert manifest.json()["start_url"] == "/chat" and manifest.json()["display"] == "standalone"
            for icon in manifest.json()["icons"]:
                image = await client.get(icon["src"])
                assert image.status_code == 200 and image.headers["content-type"] == "image/png"
                width, height = struct.unpack(">II", image.content[16:24])
                assert icon["sizes"] == f"{width}x{height}"
            for page in ("/", "/chat", "/work"):
                html = (await client.get(page)).text
                assert 'rel="manifest"' in html and 'rel="apple-touch-icon"' in html
                assert "private-admin-value" not in html


def test_lan_cli_reuses_local_address_discovery(monkeypatch):
    from arc_llama.cli import _lan_addresses

    monkeypatch.setattr(network, "local_addresses", lambda: ADDRESSES)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("DNS must not be used"))
    assert _lan_addresses() == ["192.168.1.8", "192.168.1.9", "100.101.102.103"]
