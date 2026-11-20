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

import ipaddress
import json
import re
from typing import Any

import httpx

from arc_llama.config import Config, ModelConfig

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


def integration_payload(
    cfg: Config, ollama: dict[str, Any] | None = None, *,
    loaded_model: ModelConfig | None = None, request_host: str | None = None,
) -> dict[str, Any]:
    """Build the Connect-a-frontend payload from config and probe result.

    Read-only by construction: derived from ``cfg`` plus the
    ``probe_ollama`` result, never touching secrets or mutating state.
    """
    ollama = ollama or {"url": DEFAULT_OLLAMA_URL, "reachable": False, "version": None}
    ollama_url = str(ollama.get("url") or DEFAULT_OLLAMA_URL).rstrip("/")
    configured_host = cfg.server.host
    bind_all = configured_host in _BIND_ALL_HOSTS
    host = client_host(configured_host)
    if bind_all and request_host:
        try:
            address = ipaddress.ip_address(request_host)
            family = 6 if configured_host == "::" else 4
            if address.version == family and not address.is_unspecified and not address.is_multicast:
                host = str(address)
        except ValueError:
            if len(request_host) <= 253 and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", request_host):
                host = request_host
    base_url = format_base_url(host, cfg.server.port)
    selected = loaded_model or next((m for m in cfg.models if not (m.recipe or {}).get("reranking")), None)
    model_id = selected.name if selected else "MODEL_ID"
    context = (selected.recipe or {}).get("ctx") if selected else None
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        context = None

    body = json.dumps(
        {"model": model_id, "messages": [{"role": "user", "content": "Hello"}]},
        separators=(",", ":"),
    )
    body_windows = body.replace('"', '""')
    curl_example = (
        f"curl -s {base_url}/chat/completions -H \"Content-Type: application/json\" -d '{body}'"
    )
    curl_example_windows = (
        f"curl -s {base_url}/chat/completions "
        f'-H "Content-Type: application/json" -d "{body_windows}"'
    )

    upstreams = [{"name": u.name, "url": u.url} for u in cfg.upstreams]
    already_registered = any(
        u.name == DEFAULT_OLLAMA_NAME or u.url.rstrip("/") == ollama_url for u in cfg.upstreams
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
        "api_key_guidance": (
            "On localhost, clients can use any non-empty API key string. "
            "For LAN callers, create a key with `arc-llama keys create <name>` and use it as the API key; "
            "remote keys are validated once any key exists (`arc-llama serve --lan` creates one automatically)."
        ),
        "lan_note": lan_note,
        "model_id": model_id,
        "recipe_note": (
            f"Using loaded model {model_id}."
            if loaded_model else f"No chat model is loaded; examples use configured model {model_id}."
            if selected else "Load or register a chat model, then refresh these recipes to replace MODEL_ID."
        ),
        "recipes": frontend_recipes(base_url, model_id, context),
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


def frontend_recipes(base_url: str, model: str, context: int | None) -> list[dict[str, Any]]:
    """Copyable examples; app-specific instructions and fields have one home."""
    key = "YOUR_ARC_API_KEY"
    quote = json.dumps
    continue_yaml = (
        "name: Arc Llama\nversion: 1.0.0\nschema: v1\nmodels:\n"
        f"  - name: {quote(model)}\n    provider: openai\n    model: {quote(model)}\n"
        f"    apiBase: {quote(base_url)}\n    apiKey: {quote(key)}"
    )
    zed_model: dict[str, Any] = {
        "name": model, "display_name": model,
        "capabilities": {
            "tools": True, "images": False, "parallel_tool_calls": False,
            "prompt_cache_key": False, "chat_completions": True,
            "interleaved_reasoning": False, "max_tokens_parameter": True,
        },
    }
    if context is not None:
        zed_model["max_tokens"] = context
    zed_json = f"// Enter API key {key} in Agent Settings, not this file.\n" + json.dumps({
        "language_models": {"openai_compatible": {"arc-llama": {
            "api_url": base_url, "available_models": [zed_model],
        }}},
    }, indent=2)
    if context is None:
        zed_json = (
            f"Provider name: arc-llama\nAPI URL: {base_url}\nModel ID: {model}\nAPI key: {key}\n"
            "Context window: fill from this model's configured context"
        )
    context_hint = f"Use context window {context}." if context else "Set the context window to this model's configured context; do not guess a larger value."
    return [
        {
            "id": "continue", "name": "Continue", "format": "yaml",
            "snippet_label": "config.yaml (merge the model into your existing config)",
            "steps": [
                "Open the Continue sidebar. Above the chat input, open the Agent selector and use its gear icon to edit the local config.yaml.",
                "Add this OpenAI provider model, save, then select it for Chat.",
            ],
            "snippet": continue_yaml,
            "docs_url": "https://docs.continue.dev/customize/model-providers/top-level/openai",
        },
        {
            "id": "cline", "name": "Cline", "format": "text",
            "snippet_label": "Settings values (paste each field; not an import file)",
            "steps": [
                "Open Cline's gear/settings panel and choose OpenAI Compatible as API Provider.",
                "Fill Base URL, API Key and Model/Model ID below, save, then start with a short chat.",
                context_hint + " Enable image/tool capabilities only when this model supports them.",
            ],
            "snippet": f"API Provider: OpenAI Compatible\nBase URL: {base_url}\nAPI Key: {key}\nModel ID: {model}",
            "docs_url": "https://docs.cline.bot/provider-config/openai-compatible",
        },
        {
            "id": "zed", "name": "Zed", "format": "jsonc" if context else "text",
            "snippet_label": ("settings.json (merge this language_models block)" if context
                              else "Add Provider values (paste each field; not an import file)"),
            "steps": [
                "Run agent: open settings, then Add Provider under LLM Providers. Use an OpenAI-compatible provider named arc-llama.",
                ("Merge this block into settings.json, or enter its API URL and model in Add Provider. " if context
                 else "Use these form values in Add Provider and fill the required context window. ")
                + "Enter the key placeholder's replacement in the provider UI/keychain, never settings.json.",
                context_hint + " Select the model in the Agent panel. Keep tools enabled only if this model supports them; enable images only for a vision model.",
            ],
            "snippet": zed_json,
            "docs_url": "https://zed.dev/docs/ai/use-api-access#openai-compatible-endpoints",
        },
        {
            "id": "sillytavern", "name": "SillyTavern", "format": "text",
            "snippet_label": "API Connections values (paste each field; not an import file)",
            "steps": [
                "Open API Connections. Select Chat Completion, then Custom (OpenAI-compatible) as Chat Completion Source.",
                "Paste the endpoint and key, choose the exact model from the list or enter its ID, then use Test Message. Do not append /chat/completions to the URL.",
            ],
            "snippet": f"API: Chat Completion\nChat Completion Source: Custom (OpenAI-compatible)\nCustom endpoint URL: {base_url}\nAPI key: {key}\nModel ID: {model}",
            "docs_url": "https://docs.sillytavern.app/usage/api-connections/openai/#custom-openai-compatible-endpoint",
        },
        {
            "id": "anythingllm", "name": "AnythingLLM", "format": "text",
            "snippet_label": "LLM provider values (paste each field; not an import file)",
            "steps": [
                "Open Settings, find the LLM provider selector, and choose OpenAI (Generic).",
                "Fill Base URL and API Key, then choose/type the exact Selected Model below and save.",
                context_hint + " Model context window and Max Tokens are different: the latter limits output. Test a short prompt in an empty workspace first.",
            ],
            "snippet": f"LLM Provider: OpenAI (Generic)\nBase URL: {base_url}\nAPI Key: {key}\nSelected Model: {model}",
            "docs_url": "https://docs.anythingllm.com/setup/llm-configuration/cloud/openai-generic",
        },
        {
            "id": "jan", "name": "Jan", "format": "text",
            "snippet_label": "Add Custom Provider values (paste each field; not an import file)",
            "steps": [
                "Open Settings > Model Providers > Add Provider. Choose OpenAI-compatible.",
                "Enter Provider name, Base URL and API key below, then Create. In Models, select the discovered model or use + to add its exact ID.",
                "Choose that model in a thread. Custom provider capabilities are manual; enable only the tools/vision features the model supports.",
            ],
            "snippet": f"API format: OpenAI-compatible\nProvider name: Arc Llama\nBase URL: {base_url}\nAPI key: {key}\nModel ID: {model}",
            "docs_url": "https://docs.jan.ai/desktop/remote-models/custom-endpoint",
        },
    ]
