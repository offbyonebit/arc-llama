"""Read-only local discovery for the bundled UI's "Connect a frontend" panel.

The panel helps a user point a chat frontend (Open WebUI, Ollama, or any
OpenAI-compatible client) at the running arc-llama. Everything the backend
supplies is computed locally from config plus a loopback probe of the
default Ollama address:

* the configured server bind point and the base URL a client would paste,
* whether a local Ollama answers on its default port,
* the upstreams already registered.

No credentials are read or returned, nothing is mutated, and no external
accounts are contacted — the frontend flow is copy-only by design, so the
user registers the connection in their own client (or runs the copyable
upstream command themselves) rather than letting arc-llama touch anything.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from arc_llama.config import Config

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_NAME = "ollama"
OLLAMA_PROBE_TIMEOUT = 2.0
_BIND_ALL_HOSTS = {"0.0.0.0", "::", ""}


async def probe_ollama(
    url: str = DEFAULT_OLLAMA_URL, timeout: float = OLLAMA_PROBE_TIMEOUT
) -> dict[str, Any]:
    """Return whether an Ollama server answers at ``url`` with its version.

    Probe target is the Ollama-native ``/api/version`` endpoint on the
    loopback default; unreachable is a normal state, so failures collapse
    to ``{"reachable": False}`` and never raise.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/api/version")
            response.raise_for_status()
            data = response.json()
    except Exception:
        return {"url": url, "reachable": False, "version": None}
    version = data.get("version") if isinstance(data, dict) else None
    return {"url": url, "reachable": True, "version": str(version) if version else None}


def client_host(server_host: str) -> str:
    """Host a client running on this machine uses to reach the server.

    ``0.0.0.0``/``::`` are bind-everywhere wildcards, not connectable
    addresses on every platform, so the panel shows loopback instead and
    keeps the configured value for the remote-access hint.
    """
    if server_host in _BIND_ALL_HOSTS:
        return "127.0.0.1"
    return server_host


def format_base_url(host: str, port: int) -> str:
    """Build the OpenAI-compatible ``http://host:port/v1`` base URL."""
    bracketed = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{bracketed}:{port}/v1"


def integration_payload(cfg: Config, ollama: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the Connect-a-frontend payload from config and probe result.

    Read-only by construction: derived from ``cfg`` plus the
    ``probe_ollama`` result, never touching secrets or mutating state.
    """
    ollama = ollama or {"url": DEFAULT_OLLAMA_URL, "reachable": False, "version": None}
    ollama_url = str(ollama.get("url") or DEFAULT_OLLAMA_URL).rstrip("/")
    configured_host = cfg.server.host
    bind_all = configured_host in _BIND_ALL_HOSTS
    host = client_host(configured_host)
    base_url = format_base_url(host, cfg.server.port)
    model_id = cfg.models[0].name if cfg.models else "MODEL_ID"

    body = json.dumps(
        {"model": model_id, "messages": [{"role": "user", "content": "Hello"}]},
        separators=(",", ":"),
    )
    body_windows = body.replace('"', '""')
    curl_example = (
        f'curl -s {base_url}/chat/completions '
        f'-H "Content-Type: application/json" -d \'{body}\''
    )
    curl_example_windows = (
        f'curl -s {base_url}/chat/completions '
        f'-H "Content-Type: application/json" -d "{body_windows}"'
    )

    upstreams = [{"name": u.name, "url": u.url} for u in cfg.upstreams]
    already_registered = any(
        u.name == DEFAULT_OLLAMA_NAME or u.url.rstrip("/") == ollama_url
        for u in cfg.upstreams
    )
    lan_note = (
        f"arc-llama listens on all interfaces. From another machine, replace {host} "
        "in the URL with this computer's LAN address."
        if bind_all
        else None
    )

    return {
        "server": {
            "configured_host": configured_host,
            "host": host,
            "port": cfg.server.port,
            "bind_all": bind_all,
        },
        "base_url": base_url,
        "api_key_guidance": "Any non-empty string works (arc-llama does not validate client API keys).",
        "lan_note": lan_note,
        "model_id": model_id,
        "ollama": {
            "name": DEFAULT_OLLAMA_NAME,
            "url": ollama_url,
            "reachable": bool(ollama.get("reachable")),
            "version": ollama.get("version"),
            "upstream_add_command": f"arc-llama upstream add {DEFAULT_OLLAMA_NAME} {ollama_url}",
            "already_registered": already_registered,
        },
        "curl_example": curl_example,
        "curl_example_windows": curl_example_windows,
        "upstreams": upstreams,
    }
