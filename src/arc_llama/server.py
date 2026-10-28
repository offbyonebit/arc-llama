"""OpenAI-compatible HTTP server.

Mounts on `cfg.server.host:cfg.server.port` and forwards requests to whichever
llama-server backend the router decides is the right one for the model id in
the request body.

Also exposes a small admin surface used by the bundled web UI and the TUI:

    GET  /admin/status        — full snapshot (gpus, models, who's loaded)
    GET  /admin/integration   — read-only connection guidance for frontends
    GET  /admin/plugins       — catalog of installed plugins (UI panel)
    POST /admin/load/{name}   — preload a model without sending a chat request
    POST /admin/stop/{name}   — stop one model's llama-server
    POST /admin/stop-all      — stop every running llama-server

The web UI itself is a single static page mounted at `/` when the static dir
ships with the install.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send

from arc_llama import __version__
from arc_llama.agent import run_agent
from arc_llama.agent.changes import ChangeTracker, sweep_orphans
from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.harness import (
    HARNESSES,
    OpenCodeError,
    clean_history,
    compose_prompt,
    experimental_enabled,
    filter_commands,
    find_binary,
    find_hermes_binary,
    list_capabilities,
    list_commands,
    run_hermes,
    run_opencode,
)
from arc_llama.agent.harness_history import (
    NativeHistoryError,
    get_native_session,
    list_native_sessions,
)
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.agent.profiles import PROFILE_ID, ProfileStore
from arc_llama.agent.repo_map import SemanticIndex
from arc_llama.agent.sessions import SESSION_ID, SessionStore
from arc_llama.api.context_budget import register_context_budget_route
from arc_llama.api.integration import register_integration_route
from arc_llama.api.library import register_library_routes
from arc_llama.api.work import ensure_work, register_work_routes
from arc_llama.api_keys import ApiKeyStore
from arc_llama.chat_store import Chat, ChatMessage, ChatStore, validate_compaction
from arc_llama.config import Config, default_config_path, load_config
from arc_llama.failures import StartupFailureError
from arc_llama.gguf_meta import gguf_total_bytes
from arc_llama.harness_install import (
    HarnessInstallManager,
    is_managed,
    managed_installs,
    remove_harness,
)
from arc_llama.harness_install import offer as harness_offer
from arc_llama.model_compatibility import CompatibilityChecks
from arc_llama.model_library import (
    DownloadManager,
    deletable_files,
    disk_report,
    repo_options,
    search_repos,
)
from arc_llama.perf_history import METRICS as PERF_METRICS
from arc_llama.perf_history import PerfHistory
from arc_llama.plugin_api import PLUGIN_API_VERSION, _event_bus
from arc_llama.plugins import (
    PluginDiscovery,
    build_catalog,
    load_plugins,
    register_plugins,
    shutdown_plugins,
    startup_plugins,
)
from arc_llama.resources import ResourceLeaseManager
from arc_llama.router import Router, model_vram_fit_info
from arc_llama.skills import load_skills
from arc_llama.stream_metrics import StreamMetricsObserver, generation_rate
from arc_llama.upstream import UpstreamManager

log = logging.getLogger("arc_llama.server")

_REASONING_FORMATS = {"auto", "none", "deepseek", "deepseek-legacy"}


async def _iter_sse_body(response: httpx.Response, *, require_done: bool = False):
    """Forward bytes unchanged; recognize only an SSE terminator line."""
    tail = b""
    async for chunk in response.aiter_raw():
        yield chunk
        combined = tail + chunk
        if any(line.strip() == b"data: [DONE]" for line in combined.splitlines()):
            return
        tail = combined[-64:]
    if require_done:
        raise httpx.ReadError("Inference stream ended without a completion marker")


class _LocalStreamingResponse(StreamingResponse):
    """Also finalize when the client disappears before body iteration starts."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Local stream finalizers are idempotent; the normal body/background
            # paths may already have settled this request.
            if self.background is not None:
                await self.background()


def _strip_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower()
        not in (
            "transfer-encoding",
            "content-encoding",
            "content-length",
            "connection",
        )
    }


def _local_request_body(body: dict[str, Any], target_path: str) -> bytes:
    """Apply Arc Llama's stable local chat defaults and encode the request.

    llama.cpp's generic ``deepseek`` default does not recognize every model's
    channel markers. Its ``auto`` parser selects from the active chat template
    and returns reasoning through ``reasoning_content`` for streaming and
    non-streaming requests. Explicit supported values remain available for
    clients that need llama.cpp's raw or legacy behavior.
    """
    if target_path == "/v1/chat/completions":
        requested = body.get("reasoning_format")
        if requested is None:
            body = {**body, "reasoning_format": "auto"}
        elif not isinstance(requested, str) or requested not in _REASONING_FORMATS:
            allowed = ", ".join(sorted(_REASONING_FORMATS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported reasoning_format {requested!r}. Choose one of: {allowed}.",
            )
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _cors_origins() -> list[str]:
    """Return explicitly allowed browser origins for the OpenAI API.

    The compose example uses Open WebUI on localhost:3000. Deployments with a
    different browser origin can opt in through ``ARC_LLAMA_CORS_ORIGINS`` as a
    comma-separated list; an empty value disables cross-origin browser access.
    """
    configured = os.environ.get("ARC_LLAMA_CORS_ORIGINS")
    if configured is not None:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


def _origin_host(origin: str) -> str | None:
    try:
        from urllib.parse import urlparse

        return urlparse(origin).hostname
    except Exception:
        return None


def _is_cross_origin(request: Request) -> bool:
    """True when the request carries an Origin header from a non-loopback host.

    Browsers attach an ``Origin`` header to cross-origin requests. A same-origin
    request (the bundled UI) or a non-browser client (curl, the TUI) sends no
    ``Origin``, so those are never treated as cross-origin. This lets us refuse
    cross-origin access to the token-leaking ``/admin/session-token`` endpoint
    without breaking local development or the OpenAI-compatible ``/v1`` surface.
    """
    origin = request.headers.get("Origin")
    if not origin:
        return False
    host = _origin_host(origin)
    return host is not None and host not in _LOOPBACK_HOSTS


async def _require_admin(request: Request) -> None:
    """Require the configured admin token in the Authorization header.

    When ``cfg.server.admin_token`` is unset the dependency is a no-op so
    existing single-user deployments keep working. If a token is configured,
    callers must supply ``Authorization: Bearer <token>``.
    """
    cfg: Config = request.app.state.cfg
    token = cfg.server.admin_token
    if not token:
        return

    auth = request.headers.get("Authorization", "")
    scheme, _, provided = auth.partition(" ")
    if scheme.lower() != "bearer" or not provided:
        raise HTTPException(status_code=401, detail="Admin token required")
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Invalid admin token")


_CLIENT_PROTECTED_PREFIXES = ("/v1/", "/api/")


def _bearer(request: Request) -> str:
    scheme, _, provided = request.headers.get("Authorization", "").partition(" ")
    return provided.strip() if scheme.lower() == "bearer" else ""


def client_allowed(request: Request) -> bool:
    """Whether a caller may use the inference API (``/v1/*``, ``/api/*``).

    Loopback callers always may. Remote callers may while no API key exists
    (the behaviour before keys were introduced) and afterwards only with a
    valid key or the admin token.
    """
    peer = request.client.host if request.client else ""
    if peer in _LOOPBACK_HOSTS:
        return True
    store: ApiKeyStore | None = getattr(request.app.state, "api_keys", None)
    if store is None or not store:
        return True
    provided = _bearer(request)
    if not provided:
        return False
    token = request.app.state.cfg.server.admin_token
    if token and secrets.compare_digest(provided, token):
        return True
    return store.verify(provided) is not None


def create_app(
    cfg: Config | None = None,
    config_path: Path | None = None,
    plugins: list[Any] | None = None,
) -> FastAPI:
    cfg = cfg or load_config()
    state_dir = None
    if cfg.paths.state_dir:
        state_dir = Path(cfg.paths.state_dir).expanduser()

    # Discover and instantiate plugins lazily. ``plugins`` is an explicit
    # override for tests; when None we discover from installed entry points.
    # Discovery happens here (app creation), not at import time, so optional
    # plugin dependencies are never pulled in just by importing arc_llama.
    if plugins is None:
        # Discovery records keep failed and disabled plugins visible in the
        # admin catalog without ever running their code.
        discovery = PluginDiscovery()
        app_plugins: list[Any] = load_plugins(discovery=discovery)
    else:
        discovery = PluginDiscovery()
        app_plugins = list(plugins)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = Router(cfg, log_dir=state_dir)
        app.state.upstream_mgr = UpstreamManager(cfg.upstreams)
        app.state.cfg = cfg
        app.state.compatibility_checks = CompatibilityChecks()
        app.state.started_at = time.time()
        # Exclusive GPU arbitration for plugin tasks (vision, ...). Built
        # after the router — it delegates all llama-server process
        # management to it — and before plugin startup so plugins can grab
        # it from app.state in their startup hooks.
        app.state.resources = ResourceLeaseManager(app.state.router)
        app.state.router.events = _event_bus(app)
        history = PerfHistory.for_state_dir(state_dir) if state_dir else None
        app.state.perf_history = history
        router_timings = getattr(app.state.router, "timings", None)
        if history is not None and router_timings is not None:
            router_timings.history = history
        pending_confirmations: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}
        pending_plan_approvals: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}
        app.state.pending_confirmations = pending_confirmations
        app.state.pending_plan_approvals = pending_plan_approvals
        app.state.work_runs = {}
        app.state.active_harness_runs = collections.Counter()
        app.state.harness_installs = HarnessInstallManager(cfg, config_path or default_config_path())
        app.state.work_profiles = ProfileStore((state_dir / "work-profiles.json") if state_dir else Path(".arc_llama_work_profiles.json"))
        app.state.work_sessions = SessionStore((state_dir / "work-sessions") if state_dir else Path(".arc_llama_work_sessions"))
        app.state.work_undo_dir = (state_dir / "work-undo") if state_dir else Path(".arc_llama_work_undo")
        if state_dir:
            swept = sweep_orphans(app.state.work_undo_dir)
            if swept:
                log.info("removed %d leftover work-undo folder(s) from earlier runs", swept)
        if state_dir:
            app.state.chat_store = ChatStore(state_dir / "chats")
            app.state.checkpoint_store = CheckpointStore(state_dir / "checkpoints")
            app.state.semantic_index = SemanticIndex(state_dir / "semantic_index")
        else:
            app.state.chat_store = ChatStore(Path(".arc_llama_chats"))
            app.state.checkpoint_store = CheckpointStore(Path(".arc_llama_checkpoints"))
            app.state.semantic_index = SemanticIndex(Path(".arc_llama_semantic_index"))
        app.state.work = None
        if state_dir and (state_dir / "work" / "work.sqlite3").exists():
            await ensure_work(app)
        load_skills(cfg.paths.skills_dir)
        app.state.mcp_manager = MCPClientManager(cfg.active_mcp_servers())
        tuner: Any | None = None
        if getattr(cfg, "tune", None) and cfg.tune.auto:
            from arc_llama.autotune import start_autotuner

            save_path = config_path or default_config_path()

            def _save_cfg() -> None:
                cfg.save(save_path)

            tuner = await start_autotuner(
                cfg, app.state.router, version=__version__, on_save=_save_cfg
            )
            app.state.tuner = tuner
        try:
            await app.state.mcp_manager.start()
            await startup_plugins(app_plugins, app)
            yield
        finally:
            for tracker in app.state.work_runs.values():
                tracker.close()
            if getattr(app.state, "work", None) is not None:
                await app.state.work.shutdown()
            if app.state.api_keys is not None:
                app.state.api_keys.flush()
            if app.state.perf_history is not None:
                app.state.perf_history.flush()
            await app.state.compatibility_checks.shutdown()
            first_answer = getattr(app.state, "first_answer", None)
            if first_answer is not None:
                await first_answer.shutdown()
            downloads = getattr(app.state, "downloads", None)
            if downloads is not None:
                await downloads.shutdown()
            await shutdown_plugins(app_plugins, app)
            if tuner is not None:
                await tuner.stop()
            await app.state.mcp_manager.stop()
            await app.state.router.shutdown()

    app = FastAPI(title="arc-llama", version=__version__, lifespan=lifespan)
    app.state.plugin_discovery = discovery

    # Register plugin routes before the static mount so plugin paths are not
    # shadowed by the catch-all web UI. A plugin failure here is isolated.
    # Status is recorded per app (not per lifespan) because register() runs
    # now, before the lifespan starts; /admin/plugins reads it later.
    app.state.plugin_status = {}
    register_plugins(app, app_plugins)

    app.state.cfg = cfg
    app.state.api_keys = (
        ApiKeyStore.for_state_dir(cfg.paths.state_dir) if cfg.paths.state_dir else None
    )

    @app.middleware("http")
    async def require_client_key(request: Request, call_next):
        if request.url.path.startswith(_CLIENT_PROTECTED_PREFIXES) and not client_allowed(request):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": (
                            "An API key is required for remote access. Send "
                            "'Authorization: Bearer <key>'; create one with "
                            "'arc-llama keys create NAME'."
                        ),
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        return await call_next(request)

    @app.get("/admin/api-keys")
    async def list_api_keys(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        store: ApiKeyStore | None = request.app.state.api_keys
        return {"keys": store.list() if store is not None else []}

    @app.post("/admin/api-keys")
    async def create_api_key(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        store: ApiKeyStore | None = request.app.state.api_keys
        if store is None:
            raise HTTPException(status_code=503, detail="No state directory for API keys")
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        name = body.get("name") if isinstance(body, dict) else None
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="name must be a string")
        try:
            key, plaintext = store.create(name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {**key.public(), "key": plaintext}

    @app.delete("/admin/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        store: ApiKeyStore | None = request.app.state.api_keys
        if store is None or not store.revoke(key_id):
            raise HTTPException(status_code=404, detail=f"Unknown API key: {key_id!r}")
        return {"revoked": key_id}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Liveness probe for the arc-llama router itself."""
        rt: Router = request.app.state.router
        uptime = time.time() - request.app.state.started_at
        # "Loaded" means the health check passed, not merely that a process
        # exists: during a cold start (tens of seconds) or a crash-respawn the
        # subprocess is alive but the port is not serving, and reporting that
        # as loaded misleads dashboards and scripts that gate on it.
        loaded = [
            m.name
            for m in rt.all_models()
            if rt._servers.get(m.name)
            and rt._servers[m.name].is_running
            and rt._servers[m.name].ready
        ]
        return {
            "status": "ok",
            "uptime_seconds": round(uptime, 2),
            "loaded_models": loaded,
            "loaded_model_count": len(loaded),
        }

    @app.get("/admin/session-token")
    async def admin_session_token(request: Request) -> dict[str, str | None]:
        """Hand the bundled first-party web UI its own admin token.

        Deliberately not gated by ``_require_admin`` -- it exists so the
        static page served by this same process can bootstrap itself without
        the user copy-pasting a token. Instead it's gated on the *connection*
        being loopback: the browser talking to the bundled UI on the same
        machine always looks like a loopback peer to us, so this stays
        zero-friction for the default deployment. But if the server is bound
        to a LAN/public address, a remote caller's TCP peer address won't be
        loopback, so they get refused here and have to obtain the token
        out-of-band (CLI startup output, config file) like any other
        non-local caller -- this endpoint must never become a way to fetch
        the secret over the network it's meant to guard.
        """
        peer = request.client.host if request.client else ""
        if peer not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            raise HTTPException(status_code=403, detail="Loopback connections only")
        # A cross-origin browser request (e.g. a malicious page on another host
        # that the user happens to visit) would otherwise be able to read the
        # admin token back via CORS and then drive the admin surface. Refuse
        # any request that carries a non-loopback Origin header.
        if _is_cross_origin(request):
            raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed")
        c: Config = request.app.state.cfg
        return {"admin_token": c.server.admin_token}

    @app.get("/admin/metrics")
    async def admin_metrics(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        """Operational counters and current GPU/model state."""
        rt: Router = request.app.state.router
        c: Config = request.app.state.cfg
        uptime = time.time() - request.app.state.started_at
        # "Loaded" means the health check passed, not merely that a process
        # exists: during a cold start (tens of seconds) or a crash-respawn the
        # subprocess is alive but the port is not serving, and reporting that
        # as loaded misleads dashboards and scripts that gate on it.
        loaded = [
            m.name
            for m in rt.all_models()
            if rt._servers.get(m.name)
            and rt._servers[m.name].is_running
            and rt._servers[m.name].ready
        ]
        # Test routers may predate the timings member; omit the block rather
        # than fail the whole endpoint when it is missing.
        timings = getattr(rt, "timings", None)
        timings_snapshot = timings.snapshot() if timings is not None else None
        return {
            "uptime_seconds": round(uptime, 2),
            "loads": rt.metrics["loads"],
            "stops": rt.metrics["stops"],
            "load_errors": rt.metrics["load_errors"],
            "last_load_at": rt.metrics["last_load_at"],
            "last_error": rt.metrics["last_error"],
            "active_models": loaded,
            "timings": timings_snapshot,
            "autotune": {
                "auto": c.tune.auto,
                "models": [
                    {
                        "name": m.name,
                        "tune_state": m.tune_state,
                        "tuned_at": m.tuned_at,
                        "tune_error": m.tune_error,
                        "before_after": getattr(
                            getattr(request.app.state, "tuner", None), "last_results", {}
                        ).get(m.name),
                    }
                    for m in rt.all_models()
                ],
            },
            "gpus": [
                {
                    "pci_slot": g.pci_slot,
                    "name": g.name,
                    "arch": g.arch,
                    "vram_mb": g.vram_mb,
                    "enabled": g.enabled,
                }
                for g in c.gpus
            ],
        }

    @app.get("/admin/metrics/history")
    async def admin_metrics_history(
        request: Request,
        model: str | None = None,
        metric: str | None = None,
        days: float = Query(30, gt=0, le=90),
        _auth: None = Depends(_require_admin),
    ) -> dict[str, Any]:
        """Hourly medians of real-traffic measurements, for trend charts."""
        history: PerfHistory | None = getattr(request.app.state, "perf_history", None)
        if metric is not None and metric not in PERF_METRICS:
            raise HTTPException(
                status_code=400, detail=f"metric must be one of {list(PERF_METRICS)}"
            )
        points = history.query(model, metric, days) if history is not None else []
        return {"bucket_seconds": 3600, "points": points}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        data = []
        # Local models
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            try:
                created = int(os.path.getmtime(m.path))
            except OSError:
                created = 0
            data.append(
                {
                    "id": m.name,
                    "object": "model",
                    "owned_by": "arc-llama",
                    "created": created,
                    "metadata": {
                        "display_name": m.display_name,
                        "path": m.path,
                        "model_file_mb": getattr(m, "model_file_mb", None),
                        "gpu_pci_slot": m.gpu_pci_slot,
                        "loaded": bool(srv and srv.is_running and srv.ready),
                        "aliases": list(m.aliases),
                        "capabilities": model_capabilities(m),
                        "ctx": (m.recipe or {}).get("ctx"),
                    },
                }
            )
            for alias in m.aliases:
                if alias != m.name:
                    data.append(
                        {
                            "id": alias,
                            "object": "model",
                            "owned_by": "arc-llama-alias",
                            "created": created,
                            "metadata": {"canonical": m.name},
                        }
                    )
        # Upstream models
        try:
            upstream_models = await mgr.models()
        except Exception:
            upstream_models = []
        for u in upstream_models:
            data.append(
                {
                    "id": u.id,
                    "object": "model",
                    "owned_by": f"upstream:{u.upstream_name}",
                    "created": 0,
                    "metadata": {
                        "upstream": u.upstream_name,
                        **u.metadata,
                    },
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat_or_completions(request: Request):
        return await _proxy_post(request, request.url.path)

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        """Rank documents against a query with a reranker model.

        The target model needs ``reranking = true`` in its recipe, which
        starts its llama-server with ``--reranking``.
        """
        return await _proxy_post(request, "/v1/rerank", streaming_ok=False)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)

    @app.get("/api/tags")
    async def ollama_tags(request: Request) -> dict:
        """Return the registry in Ollama's model-list shape."""
        models = await list_models(request)
        return {
            "models": [
                {
                    "name": item["id"],
                    "model": item["id"],
                    "modified_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(item.get("created", 0))
                    ),
                    "size": int(item.get("metadata", {}).get("model_file_mb", 0) or 0)
                    * 1024
                    * 1024,
                    "digest": "",
                    "details": {"family": item.get("owned_by", "arc-llama")},
                }
                for item in models["data"]
            ]
        }

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        body = await _read_json_body(request)
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400, detail="Ollama chat requests require a messages array"
            )
        payload = _ollama_to_openai(body, messages=messages)
        request._body = json.dumps(payload).encode("utf-8")  # type: ignore[attr-defined]
        response = await _proxy_post(request, "/v1/chat/completions")
        return _openai_response_as_ollama(response, body.get("model", ""), generate=False)

    @app.post("/api/generate")
    async def ollama_generate(request: Request):
        body = await _read_json_body(request)
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            raise HTTPException(
                status_code=400, detail="Ollama generate requests require a string prompt"
            )
        payload = _ollama_to_openai(body, messages=[{"role": "user", "content": prompt}])
        payload.pop("messages")
        payload["prompt"] = prompt
        request._body = json.dumps(payload).encode("utf-8")  # type: ignore[attr-defined]
        response = await _proxy_post(request, "/v1/completions")
        return _openai_response_as_ollama(response, body.get("model", ""), generate=True)

    @app.post("/v1/agent")
    async def agent_endpoint(request: Request):
        """Run the local coding agent and stream tool-execution events.

        Request body:
            {
                "model": "model-id",
                "task": "user task",
                "auto_confirm": false,
                "plan_mode": false,
                "max_turns": 30,
                "root": "/optional/project/root",  // defaults to agent.root in config
                "profile": "optional-profile-name", // must match the server's active profile
                "harness": "builtin" | "opencode" | "hermes",  // experimental; default "builtin"
                "local_only": true,                 // opencode only: never reach cloud providers
                "track_changes": false              // emit a final "changes" event and allow undo
            }

        Returns a text/event-stream of JSON objects:
            {"type": "status", "message": "..."}
            {"type": "plan", "content": "..."}
            {"type": "assistant", "content": "..."}
            {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
            {"type": "tool_result", "id": "...", "name": "...", "content": "...", "error": false}
            {"type": "confirm_required", "id": "...", "run_id": "...", "tool": "...", "arguments": {...}}
            {"type": "error", "message": "..."}
            {"type": "done"}
        """
        body = await _read_json_body(request)

        model = body.get("model")
        task = body.get("task")
        if (
            not isinstance(model, str)
            or not model.strip()
            or not isinstance(task, str)
            or not task.strip()
        ):
            raise HTTPException(
                status_code=400, detail="'model' and 'task' must be nonempty strings"
            )
        for key in ("root", "folder", "profile"):
            if body.get(key) is not None and not isinstance(body[key], str):
                raise HTTPException(status_code=400, detail=f"{key} must be a string")
        for key in ("auto_confirm", "plan_mode", "track_changes"):
            if key in body and not isinstance(body[key], bool):
                raise HTTPException(status_code=400, detail=f"{key} must be a boolean")
        max_turns = body.get("max_turns", 30)
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise HTTPException(status_code=400, detail="max_turns must be a positive integer")

        work_session_id = body.get("work_session_id")
        if work_session_id is not None and (not isinstance(work_session_id, str) or not SESSION_ID.fullmatch(work_session_id)):
            raise HTTPException(status_code=400, detail="invalid work_session_id")
        history = clean_history(body.get("history"))
        if history is None:
            raise HTTPException(status_code=400, detail="history must be a list of {role, content} messages")
        harness = body.get("harness", "builtin")
        if harness not in HARNESSES:
            raise HTTPException(status_code=400, detail=f"harness must be one of {list(HARNESSES)}")
        local_only = body.get("local_only", True)
        if not isinstance(local_only, bool):
            raise HTTPException(status_code=400, detail="local_only must be a boolean")
        if harness == "hermes" and (body.get("auto_confirm", False) or not local_only):
            raise HTTPException(status_code=400, detail="Hermes uses native approvals and requires local inference; auto-approve is unavailable")
        work_type = body.get("work_type", "code")
        if work_type not in WORK_TYPES:
            raise HTTPException(status_code=400, detail=f"work_type must be one of {list(WORK_TYPES)}")
        if work_type == "documents":
            if harness != "builtin":
                raise HTTPException(status_code=400, detail="Documents work uses the built-in agent")
            doc_model = cfg.find_model(model)
            has_vision = doc_model is not None and "vision" in model_capabilities(doc_model)
            if doc_model is None:
                try:
                    upstream = await request.app.state.upstream_mgr.models()
                except Exception:
                    upstream = []
                has_vision = any(
                    u.id == model and "vision" in (u.metadata.get("capabilities") or []) for u in upstream
                )
            if not has_vision:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Documents work needs a model that can read images, and {model!r} cannot. "
                        "Choose a vision model (one with a vision projector)."
                    ),
                )
        tool_profile: dict[str, Any] | None = None
        profile_id = body.get("tool_profile_id")
        if profile_id is not None:
            if harness != "opencode":
                raise HTTPException(status_code=400, detail="Tool profiles need the OpenCode harness")
            tool_profile = request.app.state.work_profiles.get(profile_id) if isinstance(profile_id, str) else None
            if tool_profile is None:
                raise HTTPException(status_code=400, detail="That tool profile no longer exists")
        harness_command = body.get("command")
        command_arguments = body.get("arguments", "")
        if harness_command is not None:
            if harness != "opencode":
                raise HTTPException(status_code=400, detail="Slash commands need the OpenCode harness")
            if not isinstance(harness_command, str) or not re.fullmatch(r"[\w.:-]{1,100}", harness_command):
                raise HTTPException(status_code=400, detail="command must be a command name")
            if not isinstance(command_arguments, str) or len(command_arguments) > 20000:
                raise HTTPException(status_code=400, detail="arguments must be a string")
        if harness != "builtin":
            if not experimental_enabled():
                raise HTTPException(
                    status_code=403,
                    detail="External harnesses are experimental. Set ARC_LLAMA_EXPERIMENTAL_AGENT=1 to enable them.",
                )
            # A harness can read the project and run tools; keep it behind the admin token when one is set.
            await _require_admin(request)

        auto_confirm = bool(body.get("auto_confirm", False))
        if auto_confirm:
            await _require_admin(request)
        plan_mode = bool(body.get("plan_mode", False))
        folder = body.get("folder") or ""
        root_path = body.get("root") or cfg.agent.root
        root = Path(root_path).expanduser()
        if body.get("root") and not root.is_absolute():
            root = Path.home() / root      # a typed relative path means "inside the home folder", not the server's cwd
        root = root.resolve()
        if (harness != "builtin" or body.get("track_changes", False)) and not root.is_dir():
            near = root
            while near != near.parent and not near.is_dir():
                near = near.parent
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Project folder not found on the server: {root}",
                    "closest": str(near) if near.is_dir() else None,
                },
            )
        prompt = compose_prompt(task, history, str(root))
        requested_profile = body.get("profile")
        active_profile = cfg.agent.profile
        if requested_profile is not None and requested_profile != active_profile:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Server is running profile {active_profile!r}; "
                    f"requested profile {requested_profile!r} is not active. "
                    "Start a server with the requested profile or omit the field."
                ),
            )

        run_id = str(uuid.uuid4())
        pending = request.app.state.pending_confirmations
        confirm_event = asyncio.Event()
        confirm_result: dict[str, bool] = {"approved": False}
        pending[run_id] = (confirm_event, confirm_result)

        pending_plans = request.app.state.pending_plan_approvals
        plan_event = asyncio.Event()
        plan_result: dict[str, bool] = {"approved": False}
        if plan_mode:
            pending_plans[run_id] = (plan_event, plan_result)

        base_url = f"http://{cfg.server.host}:{cfg.server.port}"
        chat_store: ChatStore = request.app.state.chat_store
        checkpoint_store: CheckpointStore = request.app.state.checkpoint_store
        semantic_index: SemanticIndex = request.app.state.semantic_index

        # Create a chat to hold the agent run transcript.
        agent_chat_id: str | None = None
        try:
            title = task.strip().split("\n")[0][:80] or "Agent task"
            agent_chat = chat_store.create(str(uuid.uuid4()), title, folder=folder)
            agent_chat_id = agent_chat.id
        except Exception as e:
            log.warning("Could not create agent chat: %s", e)

        async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
            await confirm_event.wait()
            return confirm_result["approved"]

        async def plan_callback(plan_text: str) -> bool:
            await plan_event.wait()
            return plan_result["approved"]

        tracker: ChangeTracker | None = None
        if body.get("track_changes", False):
            tracker = ChangeTracker(root, request.app.state.work_undo_dir / run_id)

        async def event_stream() -> AsyncIterator[str]:
            transcript: list[ChatMessage] = [ChatMessage(role="user", content=task)]
            kept = False
            if tracker is not None:
                await asyncio.to_thread(tracker.begin)
            if harness == "opencode":
                runner = run_opencode(
                    task=prompt,
                    model=model,
                    base_url=base_url,
                    root=root,
                    auto_confirm=auto_confirm,
                    confirm_callback=confirm_callback,
                    local_only=local_only,
                    command=harness_command,
                    arguments=command_arguments,
                    tool_profile=tool_profile,
                    session_title=task,
                    binary=cfg.agent.opencode_bin,
                )
            elif harness == "hermes":
                runner = run_hermes(
                    task=prompt, model=model, base_url=base_url, root=root,
                    auto_confirm=auto_confirm, confirm_callback=confirm_callback,
                    local_only=local_only, binary=cfg.agent.hermes_bin,
                )
            else:
                runner = run_agent(
                    task=prompt,
                    model=model,
                    base_url=base_url,
                    root=root,
                    auto_confirm=auto_confirm,
                    confirm_callback=confirm_callback,
                    plan_mode=plan_mode,
                    plan_callback=plan_callback,
                    run_id=run_id,
                    checkpoint_store=checkpoint_store,
                    max_turns=max_turns,
                    report_context=True,
                    chat_store=chat_store,
                    extra={"semantic_index": semantic_index},
                    work_type=work_type,
                )
            streamed_message: ChatMessage | None = None
            active_harness_runs = request.app.state.active_harness_runs
            active_harness_runs[harness] += 1
            try:
                async for event in runner:
                    if event.get("type") == "harness_session" and work_session_id:
                        try:
                            await asyncio.to_thread(request.app.state.work_sessions.link_harness_session, work_session_id, event.get("session_id"))
                        except (OSError, ValueError):
                            log.warning("Could not link generated harness session to its Work conversation")
                    if event.get("type") == "confirm_required":
                        # Each destructive tool needs its own approval. Reset
                        # before publishing the event so a fast reply is not lost.
                        confirm_event.clear()
                        confirm_result["approved"] = False
                        event = {**event, "run_id": run_id}
                    if event.get("type") == "plan":
                        event = {**event, "run_id": run_id}
                    if event.get("type") == "checkpoint":
                        event = {**event, "run_id": run_id}
                    yield f"data: {json.dumps(event)}\n\n"

                    if event.get("type") == "assistant_delta" and event.get("content"):
                        if streamed_message is None:
                            streamed_message = ChatMessage(role="assistant", content="")
                            transcript.append(streamed_message)
                        streamed_message.content += event["content"]
                    elif event.get("type") == "assistant" and event.get("content"):
                        streamed_message = None
                        transcript.append(ChatMessage(role="assistant", content=event["content"]))
                    elif event.get("type") == "tool_call":
                        streamed_message = None
                    elif event.get("type") == "tool_result":
                        streamed_message = None
                        label = event.get("name", "tool")
                        content = event.get("content", "")
                        transcript.append(ChatMessage(role="tool", content=f"{label}:\n{content}"))
                if tracker is not None:
                    summary = await asyncio.to_thread(tracker.finish)
                    if summary["files"]:
                        runs = request.app.state.work_runs
                        runs[run_id] = tracker
                        kept = True
                        for old in list(runs)[:-8]:        # keep only the most recent runs undoable
                            runs.pop(old).discard()
                    yield f"data: {json.dumps({'type': 'changes', 'run_id': run_id, **summary})}\n\n"
            finally:
                # A browser disconnect must close the adapter before discarding
                # its change tracker, rather than leaving a harness running.
                with anyio.CancelScope(shield=True):
                    close_runner = getattr(runner, "aclose", None)
                    if close_runner is not None:
                        await close_runner()
                active_harness_runs[harness] -= 1
                if tracker is not None and not kept:
                    tracker.discard()
                pending.pop(run_id, None)
                pending_plans.pop(run_id, None)
                if agent_chat_id is not None:
                    try:
                        chat = chat_store.get(agent_chat_id)
                        if chat is not None:
                            chat.messages.extend(transcript)
                            chat_store.save(chat)
                    except Exception as e:
                        log.warning("Could not save agent transcript: %s", e)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/v1/agent/commands")
    async def agent_commands(
        request: Request, harness: str = "opencode", root: str = "", model: str = "arc-llama", profile_id: str = "",
        _auth: None = Depends(_require_admin),
    ) -> dict[str, Any]:
        """Slash commands the chosen harness offers for a project folder (empty for the built-in agent)."""
        if harness not in HARNESSES:
            raise HTTPException(status_code=400, detail=f"harness must be one of {list(HARNESSES)}")
        if harness in ("builtin", "hermes"):
            return {"commands": []}
        if not experimental_enabled():
            raise HTTPException(status_code=403, detail="External harnesses are experimental. Set ARC_LLAMA_EXPERIMENTAL_AGENT=1 to enable them.")
        folder = Path(root).expanduser() if root else Path.home()
        if not folder.is_absolute():
            folder = Path.home() / folder
        folder = folder.resolve()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Project folder not found on the server: {folder}")
        try:
            profile = request.app.state.work_profiles.get(profile_id) if profile_id else None
            return {"commands": filter_commands(await list_commands(folder, model, binary=cfg.agent.opencode_bin), profile)}
        except OpenCodeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/v1/agent/capabilities")
    async def agent_capabilities(root: str = "", _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        """What a tool profile can switch on or off for a project: MCP servers, skills and web tools."""
        if not experimental_enabled():
            raise HTTPException(status_code=403, detail="External harnesses are experimental. Set ARC_LLAMA_EXPERIMENTAL_AGENT=1 to enable them.")
        folder = Path(root).expanduser() if root else Path.home()
        if not folder.is_absolute():
            folder = Path.home() / folder
        folder = folder.resolve()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Project folder not found on the server: {folder}")
        try:
            return await list_capabilities(folder, binary=cfg.agent.opencode_bin)
        except OpenCodeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/v1/agent/profiles")
    async def agent_profiles(request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        return {"profiles": await asyncio.to_thread(request.app.state.work_profiles.list)}

    @app.put("/v1/agent/profiles/{profile_id}")
    async def agent_profile_put(profile_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        try:
            data = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Body must be JSON") from exc
        try:
            return await asyncio.to_thread(request.app.state.work_profiles.save, profile_id, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/agent/profiles/{profile_id}")
    async def agent_profile_delete(profile_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        if not PROFILE_ID.fullmatch(profile_id):
            raise HTTPException(status_code=400, detail="invalid profile id")
        return {"deleted": await asyncio.to_thread(request.app.state.work_profiles.delete, profile_id)}

    @app.get("/v1/agent/sessions")
    async def agent_sessions(request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        """Saved Work conversations and native harness history, newest first."""
        sessions = await asyncio.to_thread(request.app.state.work_sessions.list)
        warnings: list[str] = []
        if experimental_enabled():
            native, warnings = await asyncio.to_thread(list_native_sessions)
            saved_ids = {item["id"] for item in sessions}
            represented_ids = set(saved_ids)
            by_id = {item["id"]: item for item in native}
            for item in sessions:
                linked = [by_id[sid] for sid in item.get("harness_sessions", []) if sid in by_id and sid not in represented_ids]
                if linked:
                    item["harness_history"] = sorted(linked, key=lambda entry: entry["updated"], reverse=True)
                    represented_ids.update(entry["id"] for entry in linked)
            sessions.extend(item for item in native if item["id"] not in represented_ids)
        return {"sessions": sorted(sessions, key=lambda item: item["updated"], reverse=True), "warnings": warnings}

    @app.get("/v1/agent/sessions/{session_id}")
    async def agent_session_get(session_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        try:
            record = await asyncio.to_thread(request.app.state.work_sessions.get, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record is None and experimental_enabled():
            try:
                record = await asyncio.to_thread(get_native_session, session_id)
            except NativeHistoryError as exc:
                raise HTTPException(status_code=413 if "too large" in str(exc) or "too many" in str(exc) else 503, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="No such session")
        return record

    @app.put("/v1/agent/sessions/{session_id}")
    async def agent_session_put(session_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        try:
            data = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Body must be JSON") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        try:
            return await asyncio.to_thread(request.app.state.work_sessions.save, session_id, data)
        except ValueError as exc:
            raise HTTPException(status_code=400 if "invalid" in str(exc) else 413, detail=str(exc)) from exc

    @app.delete("/v1/agent/sessions/{session_id}")
    async def agent_session_delete(session_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        try:
            removed = await asyncio.to_thread(request.app.state.work_sessions.delete, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted": removed}

    def _harness_status(name: str) -> dict[str, Any]:
        """Where the harness Work will use comes from, plus what the app can install."""
        configured = getattr(cfg.agent, f"{name}_bin")
        env_value = os.environ.get("ARC_LLAMA_OPENCODE_BIN" if name == "opencode" else "ARC_LLAMA_HERMES_BIN")
        found = (find_binary if name == "opencode" else find_hermes_binary)(configured)
        if found is None:
            source = "missing"
        elif configured and is_managed(cfg, found):
            source = "managed"
        elif configured:
            source = "config"
        elif env_value:
            source = "env"
        else:
            source = "path"
        managed = managed_installs(cfg, name)
        return {
            "available": found is not None,
            "source": source,
            "path": found,
            "managed": [{k: m.get(k) for k in ("version", "tested", "path", "installed_at")} for m in managed],
            "install": harness_offer(cfg, name),
        }

    @app.get("/admin/harnesses/jobs")
    async def harness_install_jobs(request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        manager: HarnessInstallManager = request.app.state.harness_installs
        return {"jobs": [job.public() for job in manager.jobs.values()]}

    @app.post("/admin/harnesses/{name}/install")
    async def harness_install_start(name: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        if not experimental_enabled():
            raise HTTPException(status_code=403, detail="External harnesses are experimental. Set ARC_LLAMA_EXPERIMENTAL_AGENT=1 to enable them.")
        try:
            body = await request.json()
        except ValueError:
            body = {}
        version = body.get("version", "pinned") if isinstance(body, dict) else "pinned"
        manager: HarnessInstallManager = request.app.state.harness_installs
        try:
            job = manager.submit(name, version)
        except ValueError as exc:
            raise HTTPException(status_code=409 if "already running" in str(exc) else 400, detail=str(exc)) from exc
        return job.public()

    @app.post("/admin/harnesses/jobs/{job_id}/cancel")
    async def harness_install_cancel(job_id: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        if not request.app.state.harness_installs.cancel(job_id):
            raise HTTPException(status_code=404, detail="No running install with that id")
        return {"cancelling": job_id}

    @app.delete("/admin/harnesses/{name}/{version}")
    async def harness_remove(name: str, version: str, request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        if name not in ("opencode", "hermes"):
            raise HTTPException(status_code=400, detail="Unknown harness")
        if request.app.state.active_harness_runs[name] > 0:
            raise HTTPException(status_code=409, detail=f"A Work task is using {name}. Wait for it to finish.")
        if request.app.state.harness_installs.active() is not None:
            raise HTTPException(status_code=409, detail="An agent install is running. Wait for it to finish.")
        removed = await asyncio.to_thread(remove_harness, cfg, name, version, config_path=config_path or default_config_path())
        if not removed:
            raise HTTPException(status_code=404, detail="That version is not installed by Arc Llama")
        return {"removed": version}

    @app.get("/v1/agent/info")
    async def agent_info(request: Request, _auth: None = Depends(_require_admin)) -> dict[str, Any]:
        """What the Work view can offer on this server."""
        experimental = experimental_enabled()
        return {
            "experimental": experimental,
            "root": str(Path(cfg.agent.root).expanduser().resolve()),
            "home": str(Path.home()),
            # Present only on servers whose adapter keeps destructive commands asking under auto-approve.
            "auto_approve": True,
            "work_types": [
                {"id": "code", "label": "Code", "needs_vision": False},
                {"id": "documents", "label": "Documents", "needs_vision": True},
            ],
            "vision_models": sorted(
                {m.name for m in request.app.state.router.all_models() if "vision" in model_capabilities(m)}
                | {u.id for u in await _upstream_models(request) if "vision" in (u.metadata.get("capabilities") or [])}
            ),
            "harnesses": [
                {"id": "builtin", "label": "Built-in", "available": True},
                {"id": "opencode", "label": "OpenCode", **_harness_status("opencode")},
                {"id": "hermes", "label": "Hermes", **_harness_status("hermes"),
                 "auto_approve": False,
                 "approval_note": "Hermes asks before file edits and commands flagged by its own checks. Auto-approve is unavailable."},
            ],
        }

    @app.get("/v1/agent/{run_id}/changes/{path:path}")
    async def agent_change_diff(
        run_id: str, path: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        tracker: ChangeTracker | None = request.app.state.work_runs.get(run_id)
        if tracker is None:
            raise HTTPException(status_code=404, detail="Run has no recorded changes")
        diff = await asyncio.to_thread(tracker.diff_text, path)
        if diff is None:
            raise HTTPException(status_code=404, detail="No text diff available for that file")
        return {"path": path, "diff": diff}

    @app.post("/v1/agent/{run_id}/undo")
    async def agent_undo(
        run_id: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        """Undo the file changes recorded for a run. Files edited since are left alone."""
        runs = request.app.state.work_runs
        tracker: ChangeTracker | None = runs.get(run_id)
        if tracker is None:
            raise HTTPException(status_code=404, detail="Run has no recorded changes")
        result = await asyncio.to_thread(tracker.undo)
        if result["restored"] or not tracker.undoable:
            runs.pop(run_id, None)
            tracker.discard()
        return result

    @app.post("/v1/agent/{run_id}/confirm")
    async def confirm_agent_run(
        run_id: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, bool]:
        """Approve or deny a pending agent tool confirmation.

        Request body: {"approved": true|false}
        """
        body = await _read_json_body(request)
        if not isinstance(body.get("approved", False), bool):
            raise HTTPException(status_code=400, detail="approved must be a boolean")

        entry = request.app.state.pending_confirmations.get(run_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail="Run not found or not awaiting confirmation"
            )

        event, result = entry
        result["approved"] = bool(body.get("approved", False))
        event.set()
        return {"ok": True}

    @app.post("/v1/agent/{run_id}/plan")
    async def approve_agent_plan(
        run_id: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, bool]:
        """Approve or deny the plan for a planning-mode agent run.

        Request body: {"approved": true|false}
        """
        body = await _read_json_body(request)
        if not isinstance(body.get("approved", False), bool):
            raise HTTPException(status_code=400, detail="approved must be a boolean")

        entry = request.app.state.pending_plan_approvals.get(run_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail="Run not found or not awaiting plan approval"
            )

        event, result = entry
        result["approved"] = bool(body.get("approved", False))
        event.set()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Chat history persistence
    # ------------------------------------------------------------------

    async def _chat_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        return body

    def _chat_messages(body: dict[str, Any]) -> list[ChatMessage]:
        messages = body["messages"]
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be an array")
        try:
            return [ChatMessage.from_dict(message) for message in messages]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _persist_chat(store: ChatStore, chat: Chat) -> None:
        try:
            store.save(chat)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=503,
                detail="Could not save chat. Check disk space and storage permissions, then retry.",
            ) from exc

    @app.get("/v1/chats")
    async def list_chats(request: Request, folder: str | None = Query(None)) -> dict[str, Any]:
        """Return a list of chat summaries ordered by most recently updated first.

        Pass ``?folder=...`` to filter; omit for all folders. Pass
        ``?folder=`` for the root/legacy folder only.
        """
        store: ChatStore = request.app.state.chat_store
        try:
            summaries = store.list_summaries(folder=folder)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"object": "list", "data": summaries}

    @app.get("/v1/chats/folders")
    async def list_chat_folders(request: Request) -> dict[str, Any]:
        """Return all folders with chat counts."""
        store: ChatStore = request.app.state.chat_store
        return {"object": "list", "data": store.list_folders()}

    @app.post("/v1/chats")
    async def create_chat(request: Request) -> dict[str, Any]:
        """Create a new chat.

        Body: {"id": "optional-id", "title": "optional title", "folder": "optional folder"}
        If no id is provided a UUID is generated. If no folder is provided,
        the chat is placed in the ``default`` folder.
        """
        body = await _chat_body(request)

        chat_id = body.get("id") or str(uuid.uuid4())
        title = body.get("title") or "Untitled chat"
        folder = body.get("folder") if body.get("folder") is not None else ""
        if not isinstance(folder, str):
            raise HTTPException(status_code=400, detail="folder must be a string")
        store: ChatStore = request.app.state.chat_store
        try:
            chat = store.create(chat_id, title, folder=folder)
        except FileExistsError:
            raise HTTPException(status_code=409, detail=f"Chat already exists: {chat_id}") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=503,
                detail="Could not create chat. Check disk space and storage permissions, then retry.",
            ) from exc
        return chat.to_dict()

    @app.post("/v1/chats/search")
    async def search_chats(request: Request) -> dict[str, Any]:
        """Search chat titles and messages.

        Body: {"query": "string", "limit": 20}
        Returns matching chats with the indices of matching messages.
        """
        body = await _chat_body(request)
        query = body.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise HTTPException(status_code=400, detail="query must be a nonempty string")
        try:
            value = body.get("limit", 20)
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError
            limit = int(value)
            if limit < 1:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="limit must be a positive integer") from exc
        store: ChatStore = request.app.state.chat_store
        results = store.search(query, limit=limit)
        return {
            "object": "list",
            "data": [
                {
                    "chat": chat.summary(),
                    "matching_message_indices": indices,
                }
                for chat, indices in results
            ],
        }

    @app.get("/v1/chats/export")
    async def export_chats(request: Request) -> dict[str, Any]:
        """Export every chat as a portable JSON document."""
        store: ChatStore = request.app.state.chat_store
        return {"version": 1, "exported_at": time.time(), "chats": store.export_all()}

    @app.post("/v1/chats/import")
    async def import_chats(request: Request) -> dict[str, Any]:
        """Import chats from an export document.

        Body: {"chats": [...], "overwrite": false}
        Existing chats are skipped unless ``overwrite`` is true.
        """
        body = await _chat_body(request)
        chats = body.get("chats")
        if not isinstance(chats, list):
            raise HTTPException(status_code=400, detail="'chats' must be an array")
        store: ChatStore = request.app.state.chat_store
        overwrite = body.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise HTTPException(status_code=400, detail="overwrite must be a boolean")
        result = store.import_chats(chats, overwrite=overwrite)
        return {
            "imported": result["imported"],
            "skipped": result["skipped"],
            "errors": result["errors"],
            "error_details": result["error_details"],
        }

    @app.get("/v1/chats/{chat_id}")
    async def get_chat(chat_id: str, request: Request) -> dict[str, Any]:
        """Return a full chat including all messages."""
        store: ChatStore = request.app.state.chat_store
        chat = store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat.to_dict()

    @app.put("/v1/chats/{chat_id}")
    async def update_chat(chat_id: str, request: Request) -> dict[str, Any]:
        """Replace an entire chat (title and/or messages)."""
        body = await _chat_body(request)

        store: ChatStore = request.app.state.chat_store
        chat = store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        if "title" in body:
            chat.title = str(body["title"])
        if "messages" in body:
            chat.messages = _chat_messages(body)
            # A replacement transcript invalidates an old summary unless the
            # caller explicitly supplies the matching metadata.
            chat.compaction = None
        if "compaction" in body:
            try:
                chat.compaction = validate_compaction(body["compaction"], len(chat.messages))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        _persist_chat(store, chat)
        return chat.to_dict()

    @app.patch("/v1/chats/{chat_id}")
    async def patch_chat(chat_id: str, request: Request) -> dict[str, Any]:
        """Append messages or update a chat's title/folder without replacing everything.

        Body:
            {
                "title": "new title",          // optional
                "folder": "new folder",        // optional; moves the chat
                "messages": [                  // optional; appended to existing
                    {"role": "user", "content": "..."},
                    ...
                ]
            }
        """
        body = await _chat_body(request)

        store: ChatStore = request.app.state.chat_store
        chat = store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        if "title" in body:
            chat.title = str(body["title"])
        if "folder" in body and body["folder"] is not None:
            chat.folder = str(body["folder"])
        if "messages" in body:
            chat.messages.extend(_chat_messages(body))
        _persist_chat(store, chat)
        return chat.to_dict()

    @app.delete("/v1/chats/{chat_id}")
    async def delete_chat(chat_id: str, request: Request) -> dict[str, Any]:
        """Delete a chat permanently."""
        store: ChatStore = request.app.state.chat_store
        if not store.delete(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"deleted": True}

    # ------------------------------------------------------------------
    # Admin (used by the web UI / TUI)
    # ------------------------------------------------------------------

    @app.get("/admin/status")
    async def admin_status(request: Request, _auth: None = Depends(_require_admin)) -> dict:
        rt: Router = request.app.state.router
        c: Config = request.app.state.cfg
        # Reuse one VRAM-estimate cache across this snapshot (and across calls:
        # it lives on app.state) so repeated status polls never re-scan GGUFs.
        vram_cache: dict[str, tuple[float, int | None]] | None = getattr(
            request.app.state, "vram_estimate_cache", None
        )
        if vram_cache is None:
            vram_cache = {}
            request.app.state.vram_estimate_cache = vram_cache
        from arc_llama.model_compatibility import model_runtime_identity
        from arc_llama.model_library import file_readiness

        identities: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[str, str] | None] = {}
        models = []
        for m in rt.all_models():
            runtime_key = (c.model_llama_server(m), tuple(sorted(m.env.items())))
            if runtime_key not in identities:
                identities[runtime_key] = await asyncio.to_thread(model_runtime_identity, c, m)
            identity = identities[runtime_key]
            srv = rt._servers.get(m.name)
            r = m.recipe or {}
            running = bool(srv and srv.is_running)
            loaded = bool(srv and srv.is_running and srv.ready)
            model_file_mb = (
                (gguf_total_bytes(m.path) + 1_048_575) // 1_048_576
                if Path(m.path).exists()
                else None
            )
            fit_info = await asyncio.to_thread(
                model_vram_fit_info, m, c.find_gpu(m.gpu_pci_slot), vram_cache
            )
            if fit_info and not c.server.single_resident:
                fit_info["headroom_mb"] = None
                fit_info["fit"] = None
                fit_info["detail"] = (
                    "Per-model estimate; available memory depends on co-resident models."
                )
            files = await asyncio.to_thread(file_readiness, m)
            models.append(
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "path": m.path,
                    "llama_server": m.llama_server,
                    "effective_llama_server": runtime_key[0],
                    "env": dict(m.env),
                    "runtime_compatibility_identity": identity[1][:12] if identity else None,
                    "model_file_mb": model_file_mb,
                    "file_readiness": files,
                    "gpu_pci_slot": m.gpu_pci_slot,
                    "port": m.port,
                    "loaded": loaded,
                    "state": (
                        "draining"
                        if m.name in getattr(rt, "_stopping", set())
                        else "ready"
                        if loaded
                        else "loading"
                        if running
                        else "idle"
                    ),
                    "pid": getattr(getattr(srv, "process", None), "pid", None) if running else None,
                    "ctx": r.get("ctx"),
                    "cache_type_k": r.get("cache_type_k"),
                    "cache_type_v": r.get("cache_type_v"),
                    "flash_attn": r.get("flash_attn"),
                    "ubatch_size": r.get("ubatch_size"),
                    "batch_size": r.get("batch_size"),
                    "kv_class": m.kv_class,
                    "aliases": list(m.aliases),
                    "capabilities": model_capabilities(m),
                    "mmproj": r.get("mmproj"),
                    "tensor_split": r.get("tensor_split"),
                    "tune_state": m.tune_state,
                    "tuned_at": m.tuned_at,
                    "tune_error": m.tune_error,
                    "tune_fingerprint": m.tune_fingerprint,
                    "vram_estimate": fit_info,
                }
            )
        gpus = [
            {
                "pci_slot": g.pci_slot,
                "sycl_index": g.sycl_index,
                "arch": g.arch,
                "vram_mb": g.vram_mb,
                "name": g.name,
                "enabled": g.enabled,
                "backend": g.backend,
            }
            for g in c.gpus
        ]
        mgr: UpstreamManager = request.app.state.upstream_mgr
        from arc_llama.model_compatibility import runtime_identity

        compatibility_identity = await asyncio.to_thread(runtime_identity, c.paths.llama_server)
        return {
            "runtime_compatibility_identity": compatibility_identity[1][:12]
            if compatibility_identity
            else None,
            "server": {
                "host": c.server.host,
                "port": c.server.port,
                "single_resident": c.server.single_resident,
                "auto_tune": c.tune.auto,
                "switch_drain_seconds": c.server.switch_drain_seconds,
                "switch_interrupt_policy": c.server.switch_interrupt_policy,
            },
            "gpus": gpus,
            "models": models,
            "upstreams": mgr.upstreams_status(),
        }

    register_work_routes(app, require_admin=_require_admin, read_json_body=_read_json_body)
    from arc_llama.api.onboarding import register_onboarding_routes
    register_onboarding_routes(app, require_admin=_require_admin,
                               read_json_body=_read_json_body,
                               config_path=config_path or default_config_path())
    register_context_budget_route(app)
    register_integration_route(app, require_admin=_require_admin)

    @app.get("/admin/plugins")
    async def admin_plugins(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        """Catalog installed plugins for the dashboard's Plugins panel.

        Read-only and reflective: it reports what was discovered at app
        creation, a stable per-plugin status key, and whatever metadata each
        plugin chooses to publish through its optional ``info()`` hook.
        Plugins written against the original contract simply appear with
        name and status only. Discovery records for plugins that failed to
        import or were disabled appear alongside with their error, without
        any plugin code running here.
        """
        statuses: dict[str, str] = getattr(request.app.state, "plugin_status", {})
        discovery: PluginDiscovery | None = getattr(request.app.state, "plugin_discovery", None)
        extra = discovery.catalog() if discovery is not None else []
        return {
            "api_version": PLUGIN_API_VERSION,
            "plugins": build_catalog(app_plugins, statuses, extra=extra),
        }

    def _ui_layout_path() -> Path:
        base = state_dir or Path(".arc_llama_state")
        base.mkdir(parents=True, exist_ok=True)
        return base / "ui-layout.json"

    def _ui_actions() -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for plugin in build_catalog(app_plugins, getattr(app.state, "plugin_status", {})):
            for action in plugin.get("ui", {}).get("actions", []):
                item = dict(action)
                item["plugin"] = plugin["name"]
                actions.append(item)
        return actions

    @app.get("/admin/ui/layout")
    async def get_ui_layout(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        actions = _ui_actions()
        ids = {a["id"] for a in actions}
        path = _ui_layout_path()
        try:
            saved = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            saved = {}
        layout = saved.get("layout", {}) if isinstance(saved, dict) else {}
        result: dict[str, Any] = {"layout": {}, "actions": actions}
        for placement in ("toolbar", "plugins", "chat"):
            raw = layout.get(placement, []) if isinstance(layout, dict) else []
            result["layout"][placement] = [x for x in raw if x in ids]
        used = set(result["layout"]["toolbar"] + result["layout"]["plugins"])
        for action in actions:
            if action["id"] not in used:
                result["layout"][action.get("placement", "plugins")].append(action["id"])
        result["hidden"] = [
            x for x in (saved.get("hidden", []) if isinstance(saved, dict) else []) if x in ids
        ]
        return result

    @app.put("/admin/ui/layout")
    async def put_ui_layout(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        body = await request.json()
        actions = _ui_actions()
        ids = {a["id"] for a in actions}
        layout = body.get("layout", {}) if isinstance(body, dict) else {}
        hidden = body.get("hidden", []) if isinstance(body, dict) else []
        if not isinstance(layout, dict) or not isinstance(hidden, list):
            raise HTTPException(status_code=400, detail="Invalid UI layout")
        values = [x for p in ("toolbar", "plugins", "chat") for x in layout.get(p, [])]
        if any(not isinstance(x, str) or x not in ids for x in values + hidden) or len(
            values
        ) != len(set(values)):
            raise HTTPException(status_code=400, detail="Unknown or duplicate UI action")
        payload = {
            "layout": {p: list(layout.get(p, [])) for p in ("toolbar", "plugins", "chat")},
            "hidden": hidden,
        }
        path = _ui_layout_path()
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return await get_ui_layout(request)

    @app.post("/admin/load/{name}")
    async def admin_load(name: str, request: Request, _auth: None = Depends(_require_admin)) -> Any:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(
                status_code=400, detail=f"Upstream model cannot be loaded locally: {name!r}"
            )
        try:
            model, srv = await rt.ensure_active(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}") from None
        except StartupFailureError as e:
            # Details come along: the UI renders them behind an expandable
            # diagnostics region, and the payload is redacted + bounded by
            # the failure object itself.
            return JSONResponse(status_code=e.http_status, content=e.to_dict(include_details=True))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        # ensure_active only returns once wait_ready has passed, so ready is
        # the accurate claim here; is_running would also report a server that
        # crashed between readiness and this line.
        return {"name": model.name, "loaded": bool(srv.is_running and srv.ready)}

    @app.post("/admin/stop/{name}")
    async def admin_stop(
        name: str, request: Request, force: bool = False, _auth: None = Depends(_require_admin)
    ) -> Any:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(
                status_code=400, detail=f"Upstream model cannot be stopped locally: {name!r}"
            )
        if name not in {m.name for m in rt.all_models()}:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        try:
            was_running = await rt.stop_one(name, force=force)
        except StartupFailureError as exc:
            return JSONResponse(
                status_code=exc.http_status, content=exc.to_dict(include_details=True)
            )
        return {"name": name, "was_running": was_running, "loaded": False}

    @app.post("/admin/stop-all")
    async def admin_stop_all(
        request: Request, force: bool = False, _auth: None = Depends(_require_admin)
    ) -> Any:
        rt: Router = request.app.state.router
        try:
            stopped = await rt.stop_all(force=force)
        except StartupFailureError as exc:
            return JSONResponse(
                status_code=exc.http_status, content=exc.to_dict(include_details=True)
            )
        return {"stopped": stopped}

    @app.get("/admin/tune/status")
    async def admin_tune_status(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        """Return the current auto-tuner state and per-model tune status."""
        rt: Router = request.app.state.router
        c: Config = request.app.state.cfg
        tuner = getattr(request.app.state, "tuner", None)
        return {
            "auto_tune": c.tune.auto,
            "idle_seconds": c.tune.idle_seconds,
            "sweep_running": bool(tuner and tuner.is_sweep_running),
            "sweep_model": getattr(tuner, "running_model", None),
            "sweep_stage": getattr(tuner, "running_stage", None),
            "models": [
                {
                    "name": m.name,
                    "tune_state": m.tune_state,
                    "tuned_at": m.tuned_at,
                    "tune_error": m.tune_error,
                    "tune_fingerprint": m.tune_fingerprint,
                }
                for m in rt.all_models()
            ],
        }

    @app.post("/admin/tune/{name}")
    async def admin_tune_queue(
        name: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, Any]:
        """Queue a model for immediate tuning when the next idle window arrives."""
        c: Config = request.app.state.cfg
        m = c.find_model(name)
        if m is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        tuner = getattr(request.app.state, "tuner", None)
        if tuner is None:
            raise HTTPException(status_code=503, detail="Auto-tuner is not running")
        ok = tuner.queue_now(name)
        return {"queued": ok, "name": name}

    @app.delete("/admin/tune")
    async def admin_tune_abort(
        request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, bool]:
        """Abort the currently running sweep."""
        tuner = getattr(request.app.state, "tuner", None)
        if tuner is None:
            return {"aborted": False}
        aborted = tuner.abort_sweep()
        return {"aborted": aborted}

    @app.post("/admin/parse-pdf")
    async def admin_parse_pdf(
        request: Request,
        file: UploadFile,
        _auth: None = Depends(_require_admin),
    ) -> dict:
        """Extract text from an uploaded PDF using pypdf.

        Returns the original filename and the concatenated text of all pages.
        The endpoint is lazy about importing pypdf so the rest of the server
        starts fine even when the optional dependency is missing.
        """
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail="PDF parsing is not available; install pypdf: pip install pypdf",
            ) from e

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        try:
            content = await file.read()
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="PDF must be under 50 MB")
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}") from e
        finally:
            await file.close()

        return {"filename": file.filename, "text": text}

    @app.post("/admin/models/{name}/edit")
    async def admin_edit_model(
        name: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict:
        """Update a model's recipe in-place.

        Body is a partial recipe dict — only provided fields change. Recognised
        fields: `ctx`, `cache_type_k`, `cache_type_v`, `parallel`, `kv_class`,
        `spec_type`, `spec_draft_name`, `spec_draft_n_max`, `spec_draft_p_min`,
        `spec_draft_model`, `spec_draft_ngl`, `ubatch_size`,
        `batch_size`, `flash_attn` (null clears), `llama_server` (empty inherits
        the global executable), `env` (model-only string variables; empty clears),
        `n_cpu_moe` (null or 0 clears), `override_tensor` (list of regex
        patterns, null clears). When `override_tensor` is set, `n_cpu_moe` is
        cleared and vice versa: the two flags are alternative means to the
        same end and applying both would double-count the offload.
        If the model is currently loaded, the server is stopped first; callers
        decide whether to reload it afterwards via /admin/load.
        """
        from arc_llama.config import validate_model_runtime
        from arc_llama.gguf_meta import validate_override_patterns, weight_tensor_table
        from arc_llama.recipes import FLASH_ATTN_VALUES, KVCacheType

        c: Config = request.app.state.cfg
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(
                status_code=400, detail=f"Upstream model cannot be edited locally: {name!r}"
            )
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        model = next((m for m in c.models if m.name == name), None)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        valid_kv = {kv.value for kv in KVCacheType}
        valid_classes = {
            "default",
            "moe_a3b",
            "qwen3_dense",
            "qwen3_27b_dense",
            "qwen2_5",
            "gemma_swa",
            "phi4",
            "llama3",
            "deepseek_r1_distill",
        }
        recipe = dict(model.recipe or {})
        changed: list[str] = []
        runtime_binary = body.get("llama_server", model.llama_server)
        runtime_env = body.get("env", model.env)
        try:
            validate_model_runtime(runtime_binary, runtime_env)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if "llama_server" in body:
            changed.append("llama_server")
        if "env" in body:
            changed.append("env")
        if "ctx" in body:
            try:
                ctx = int(body["ctx"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="ctx must be an integer") from None
            if not (256 <= ctx <= 1_048_576):
                raise HTTPException(status_code=400, detail="ctx must be 256..1048576")
            recipe["ctx"] = ctx
            changed.append("ctx")
        for fld in ("cache_type_k", "cache_type_v"):
            if fld in body:
                v = str(body[fld])
                if v not in valid_kv:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{fld} must be one of {sorted(valid_kv)}",
                    )
                recipe[fld] = v
                changed.append(fld)
        if "parallel" in body:
            try:
                par = int(body["parallel"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="parallel must be an integer") from None
            if not (1 <= par <= 32):
                raise HTTPException(status_code=400, detail="parallel must be 1..32")
            recipe["parallel"] = par
            changed.append("parallel")
        if "kv_class" in body:
            v = str(body["kv_class"])
            if v not in valid_classes:
                raise HTTPException(
                    status_code=400,
                    detail=f"kv_class must be one of {sorted(valid_classes)}",
                )
            model.kv_class = v
            changed.append("kv_class")
        if "spec_type" in body:
            v = body["spec_type"]
            if v is None or v == "":
                recipe.pop("spec_type", None)
            elif (
                isinstance(v, str) and len(v) <= 64 and all(ch.isalnum() or ch in "-_" for ch in v)
            ):
                # llama.cpp adds speculation strategies faster than arc-llama
                # releases. Keep this future-compatible while rejecting values
                # that cannot be one argv atom.
                recipe["spec_type"] = v
            else:
                raise HTTPException(
                    status_code=400,
                    detail="spec_type must be a short strategy name or null",
                )
            changed.append("spec_type")
        if "spec_draft_name" in body:
            v = body["spec_draft_name"]
            if v is None or v == "":
                recipe.pop("spec_draft_name", None)
            else:
                draft_name = str(v)
                if draft_name == name or c.find_model(draft_name) is None:
                    raise HTTPException(
                        status_code=400,
                        detail="spec_draft_name must name another registered model",
                    )
                recipe["spec_draft_name"] = draft_name
            changed.append("spec_draft_name")
        if "spec_draft_n_max" in body:
            v = body["spec_draft_n_max"]
            if v is None:
                recipe.pop("spec_draft_n_max", None)
            else:
                try:
                    draft_n = int(v)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400, detail="spec_draft_n_max must be an integer or null"
                    ) from None
                if not (1 <= draft_n <= 16):
                    raise HTTPException(status_code=400, detail="spec_draft_n_max must be 1..16")
                recipe["spec_draft_n_max"] = draft_n
            changed.append("spec_draft_n_max")
        if "spec_draft_p_min" in body:
            from arc_llama.speculation import confidence_value

            v = body["spec_draft_p_min"]
            if v is None:
                recipe.pop("spec_draft_p_min", None)
            else:
                try:
                    recipe["spec_draft_p_min"] = confidence_value(v)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
            changed.append("spec_draft_p_min")
        if "spec_draft_model" in body:
            v = body["spec_draft_model"]
            if v is None:
                recipe.pop("spec_draft_model", None)
            elif isinstance(v, str) and Path(v).suffix.lower() == ".gguf" and Path(v).is_file():
                recipe["spec_draft_model"] = v
            else:
                raise HTTPException(
                    status_code=400,
                    detail="spec_draft_model must be an existing .gguf path or null",
                )
            changed.append("spec_draft_model")
        if "spec_draft_ngl" in body:
            v = body["spec_draft_ngl"]
            if v is None:
                recipe.pop("spec_draft_ngl", None)
            elif isinstance(v, int) and not isinstance(v, bool) and -1 <= v <= 999:
                recipe["spec_draft_ngl"] = v
            else:
                raise HTTPException(
                    status_code=400, detail="spec_draft_ngl must be -1..999 or null"
                )
            changed.append("spec_draft_ngl")
        if "speculation_result" in body:
            v = body["speculation_result"]
            if v is None or v == "":
                recipe.pop("speculation_result", None)
            elif isinstance(v, str):
                recipe["speculation_result"] = v[:512]
            else:
                raise HTTPException(
                    status_code=400, detail="speculation_result must be a string or null"
                )
            changed.append("speculation_result")
        if "ubatch_size" in body:
            try:
                ub = int(body["ubatch_size"])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="ubatch_size must be an integer"
                ) from None
            if not (1 <= ub <= 4096):
                raise HTTPException(status_code=400, detail="ubatch_size must be 1..4096")
            recipe["ubatch_size"] = ub
            changed.append("ubatch_size")
        if "batch_size" in body:
            try:
                b = int(body["batch_size"])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="batch_size must be an integer"
                ) from None
            if not (1 <= b <= 8192):
                raise HTTPException(status_code=400, detail="batch_size must be 1..8192")
            recipe["batch_size"] = b
            changed.append("batch_size")
        if "flash_attn" in body:
            v = body["flash_attn"]
            if v is None:
                recipe.pop("flash_attn", None)
            elif str(v) in FLASH_ATTN_VALUES:
                recipe["flash_attn"] = str(v)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"flash_attn must be one of {list(FLASH_ATTN_VALUES)} or null",
                )
            changed.append("flash_attn")
        if "n_cpu_moe" in body:
            v = body["n_cpu_moe"]
            if v is None:
                recipe.pop("n_cpu_moe", None)
            else:
                try:
                    n = int(v)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400, detail="n_cpu_moe must be an integer or null"
                    ) from None
                if not (0 <= n <= 1024):
                    raise HTTPException(
                        status_code=400, detail="n_cpu_moe must be 0..1024 (layers)"
                    )
                if n == 0:
                    recipe.pop("n_cpu_moe", None)
                else:
                    recipe["n_cpu_moe"] = n
                    recipe.pop("override_tensor", None)
            changed.append("n_cpu_moe")
        if "override_tensor" in body:
            v = body["override_tensor"]
            if v is None:
                recipe.pop("override_tensor", None)
            else:
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise HTTPException(
                        status_code=400,
                        detail="override_tensor must be a list of strings or null",
                    )
                table = weight_tensor_table(model.path)
                ok, err = validate_override_patterns(table, v)
                if not ok:
                    raise HTTPException(status_code=400, detail=err)
                recipe["override_tensor"] = list(v)
                recipe.pop("n_cpu_moe", None)
            changed.append("override_tensor")
        if "mmproj" in body:
            v = body["mmproj"]
            if v is None or v == "":
                recipe.pop("mmproj", None)
                recipe.pop("mmproj_offload", None)
            else:
                mmproj_path = Path(str(v)).expanduser()
                if mmproj_path.suffix.lower() != ".gguf" or not mmproj_path.is_file():
                    raise HTTPException(
                        status_code=400, detail="mmproj must be the path of an existing .gguf file"
                    )
                recipe["mmproj"] = str(mmproj_path)
            changed.append("mmproj")
        for flag in ("mmproj_offload", "reranking"):
            if flag in body:
                v = body[flag]
                if not isinstance(v, bool):
                    raise HTTPException(status_code=400, detail=f"{flag} must be a boolean")
                default = flag == "mmproj_offload"
                if v == default:
                    recipe.pop(flag, None)
                else:
                    recipe[flag] = v
                changed.append(flag)
        if "tensor_split" in body:
            v = body["tensor_split"]
            if v is None or v == []:
                recipe.pop("tensor_split", None)
                recipe.pop("split_mode", None)
                recipe.pop("split_gpus", None)
            else:
                if (
                    not isinstance(v, list)
                    or not 2 <= len(v) <= 8
                    or not all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0
                        for x in v
                    )
                    or not any(x > 0 for x in v)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="tensor_split must be 2..8 non-negative numbers, or null",
                    )
                recipe["tensor_split"] = [float(x) for x in v]
            changed.append("tensor_split")
        if "split_gpus" in body:
            v = body["split_gpus"]
            if v is None or v == []:
                recipe.pop("split_gpus", None)
            else:
                known = {g.pci_slot for g in c.gpus}
                if (
                    not isinstance(v, list)
                    or not all(isinstance(x, str) and x in known for x in v)
                    or len(set(v)) != len(v)
                    or model.gpu_pci_slot not in v
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="split_gpus must list distinct configured GPU slots, "
                        "including the model's own GPU",
                    )
                recipe["split_gpus"] = list(v)
            changed.append("split_gpus")
        if "split_mode" in body:
            from arc_llama.recipes import SPLIT_MODES

            v = body["split_mode"]
            if v is None:
                recipe.pop("split_mode", None)
            elif v in SPLIT_MODES:
                recipe["split_mode"] = v
            else:
                raise HTTPException(
                    status_code=400, detail=f"split_mode must be one of {list(SPLIT_MODES)} or null"
                )
            changed.append("split_mode")
        if not changed:
            raise HTTPException(status_code=400, detail="no recognised fields to edit")
        previous_recipe = model.recipe
        previous_runtime = model.llama_server, model.env
        model.recipe = recipe
        model.llama_server, model.env = runtime_binary, dict(runtime_env)
        try:
            c.save(config_path or default_config_path())
        except OSError as e:
            # Previously this was logged and the edit continued, so the caller
            # got a 200 listing the fields it "changed" while the config on
            # disk still held the old recipe. The running server was then
            # rebuilt to match the unsaved version, so memory and disk
            # disagreed until the next restart quietly reverted the edit. Undo
            # and fail loudly instead: a rejected edit is recoverable, an edit
            # that silently un-applies later is not.
            model.recipe = previous_recipe
            model.llama_server, model.env = previous_runtime
            log.warning("edit %s: persist failed: %s", name, e)
            raise HTTPException(
                status_code=500,
                detail=f"Could not persist config; edit rolled back: {e}",
            ) from e
        rebuilt, was_running = await rt.rebuild_model(name)
        return {
            "name": name,
            "changed": changed,
            "recipe": recipe,
            "llama_server": model.llama_server,
            "env": model.env,
            "kv_class": model.kv_class,
            "stopped_running_instance": was_running,
            "rebuilt": rebuilt,
        }

    @app.post("/admin/scan")
    async def admin_scan(request: Request, _auth: None = Depends(_require_admin)) -> dict:
        """Re-walk scan paths for new GGUFs and auto-register them.

        Mutates the in-memory Config and persists it to disk so subsequent
        restarts see the same registry. New models become loadable via
        `/admin/load/{name}` immediately — the router rebuilds its server map.
        """
        from arc_llama.models import discover_ggufs, register_discovered

        c: Config = request.app.state.cfg
        rt: Router = request.app.state.router
        try:
            found = discover_ggufs(c)
            added = register_discovered(c, found)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if added:
            try:
                c.save(config_path or default_config_path())
            except OSError as e:
                log.warning("scan: persist failed: %s", e)
            # Rebuild the router's server map so new entries are immediately
            # visible to /admin/load. We don't disturb running servers.
            rt._build_servers()  # type: ignore[attr-defined]
        return {
            "found": len(found),
            "added": [m.name for m in added],
        }

    register_library_routes(
        app,
        require_admin=_require_admin,
        read_json_body=lambda request: _read_json_body(request),
        config_path=config_path,
        search_repos_fn=lambda q, **kwargs: search_repos(q, **kwargs),
        repo_options_fn=lambda repo, vram: repo_options(repo, vram),
        disk_report_fn=lambda config, last_used: disk_report(config, last_used),
        deletable_files_fn=lambda config, model: deletable_files(config, model),
        download_manager_factory=lambda config, register: DownloadManager(config, register),
    )

    # ------------------------------------------------------------------
    # Static web UI (optional; only mounted if the static dir is present)
    # ------------------------------------------------------------------
    static_dir = Path(__file__).parent / "static"

    @app.get("/work")
    async def work_page() -> Response:
        return FileResponse(static_dir / "chat.html")

    @app.get("/workbench")
    async def workbench_page() -> Response:
        return FileResponse(static_dir / "work.html")

    @app.get("/chat")
    async def chat_page() -> Response:
        return FileResponse(static_dir / "chat.html")

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


async def _read_json_body(request: Request) -> dict[str, Any]:
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


def _ollama_to_openai(body: dict[str, Any], *, messages: list[Any]) -> dict[str, Any]:
    raw_options = body.get("options")
    options: dict[str, Any] = raw_options if isinstance(raw_options, dict) else {}
    payload: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": bool(body.get("stream", True)),
    }
    for ollama_key, openai_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop", "stop"),
    ):
        if ollama_key in options:
            payload[openai_key] = options[ollama_key]
    if "num_predict" in options:
        payload["max_tokens"] = options["num_predict"]
    return payload


def _openai_response_as_ollama(response: Response, model: str, *, generate: bool):
    if response.status_code >= 400:
        try:
            body = json.loads(bytes(response.body))
        except (TypeError, json.JSONDecodeError):
            body = {}
        # Structured startup failures carry their readable message under
        # "error.message"; flat errors keep using "detail". Either way the
        # Ollama-compat client sees one plain string, never nested JSON.
        if isinstance(body, dict):
            error_obj = body.get("error")
            if isinstance(error_obj, dict) and isinstance(error_obj.get("message"), str):
                detail = error_obj["message"]
            else:
                detail = str(body.get("detail", "Upstream request failed"))
        else:
            detail = "Upstream request failed"
        return JSONResponse({"error": str(detail)}, status_code=response.status_code)
    if isinstance(response, StreamingResponse):

        async def stream() -> AsyncIterator[bytes]:
            buffer = b""
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                buffer += chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                while b"\n\n" in buffer:
                    raw, buffer = buffer.split(b"\n\n", 1)
                    for line in raw.splitlines():
                        if not line.startswith(b"data:"):
                            continue
                        text = line[5:].strip()
                        if text == b"[DONE]":
                            continue
                        try:
                            obj = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        choice = (obj.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        value = delta.get("content") if not generate else choice.get("text", "")
                        if value:
                            yield (
                                json.dumps(
                                    {
                                        "model": model,
                                        "created_at": time.strftime(
                                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                        ),
                                        "message": {"role": "assistant", "content": value},
                                        "response": value if generate else "",
                                        "done": False,
                                    }
                                )
                                + "\n"
                            ).encode()
                        if choice.get("finish_reason"):
                            yield (
                                json.dumps(
                                    {
                                        "model": model,
                                        "created_at": time.strftime(
                                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                        ),
                                        "done": True,
                                        "done_reason": choice["finish_reason"],
                                    }
                                )
                                + "\n"
                            ).encode()

        return StreamingResponse(stream(), media_type="application/x-ndjson")
    try:
        obj = json.loads(bytes(response.body))
    except (TypeError, json.JSONDecodeError):
        return response
    choice = (obj.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = choice.get("text", "") if generate else message.get("content", "")
    result: dict[str, Any] = {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "done": True,
        "done_reason": choice.get("finish_reason"),
    }
    if generate:
        result["response"] = content
    else:
        result["message"] = {"role": message.get("role", "assistant"), "content": content}
        if message.get("reasoning_content"):
            result["message"]["thinking"] = message["reasoning_content"]
    return JSONResponse(result, status_code=response.status_code)


_IMAGE_PART_TYPES = {"image_url", "input_image", "image"}


def request_has_images(body: dict[str, Any]) -> bool:
    """True when an OpenAI chat body carries image content parts."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                return True
    return False


WORK_TYPES = ("code", "documents")


async def _upstream_models(request: Request) -> list[Any]:
    try:
        return list(await request.app.state.upstream_mgr.models())
    except Exception:
        return []


def model_capabilities(model: Any) -> list[str]:
    """Capability tags advertised in /v1/models and used for validation."""
    recipe = getattr(model, "recipe", None) or {}
    if recipe.get("reranking"):
        return ["rerank"]
    caps = ["chat", "completion", "embedding"]
    flags = [str(f) for f in recipe.get("extra_flags") or []]
    if recipe.get("mmproj") or any(f in ("--mmproj", "-mm") or f.startswith("--mmproj=") for f in flags):
        caps.append("vision")
    return caps


def _check_local_capabilities(
    rt: Router, model_query: str, body: dict[str, Any], target_path: str
) -> None:
    cfg = getattr(rt, "cfg", None)
    model = cfg.find_model(model_query) if cfg is not None and model_query else None
    if model is None:
        return
    caps = model_capabilities(model)
    if target_path == "/v1/rerank" and "rerank" not in caps:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {model.name!r} is not a reranker. Set reranking = true in its "
                "recipe, or choose a reranker model."
            ),
        )
    if target_path == "/v1/chat/completions" and request_has_images(body) and "vision" not in caps:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {model.name!r} cannot read images: it has no vision projector "
                "(mmproj). Choose a vision-language model, or set mmproj in its recipe."
            ),
        )


async def _proxy_post(request: Request, target_path: str, streaming_ok: bool = True):
    rt: Router = request.app.state.router
    mgr: UpstreamManager = request.app.state.upstream_mgr
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    if not isinstance(body.get("model", ""), str):
        raise HTTPException(status_code=400, detail="model must be a string")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise HTTPException(status_code=400, detail="stream must be a boolean")
    model_query = body.get("model", "")

    # Check upstreams first — they are passive proxies, no llama-server to start.
    # Ensure the model list cache is warm before looking up the model id; otherwise
    # the very first chat request after startup can incorrectly 404 an upstream
    # model until /v1/models has been queried.
    await mgr.models()
    upstream_model = mgr.find_model(model_query)
    if upstream_model is not None:
        try:
            upstream_client, upstream_resp = await mgr.proxy(
                upstream_model,
                target_path,
                body_bytes,
                {"Content-Type": "application/json"},
                streaming_ok=streaming_ok,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
        want_stream = streaming_ok and bool(body.get("stream"))
        if want_stream:
            released = False

            async def close_upstream() -> None:
                """Close response then client, exactly once.

                The BackgroundTask alone was not enough: Starlette only runs
                it when streaming the body returned normally, so an upstream
                that dies mid-generation skipped it and leaked both the
                response's connection and the client's pool. The generator's
                finally covers that path; the task settles a late finalize.
                """
                nonlocal released
                if released:
                    return
                released = True
                try:
                    await upstream_resp.aclose()
                except Exception:
                    log.debug("upstream proxy: response close failed", exc_info=True)
                try:
                    await upstream_client.aclose()
                except Exception:
                    log.debug("upstream proxy: client close failed", exc_info=True)

            async def upstream_body_iter():
                try:
                    async for chunk in _iter_sse_body(upstream_resp):
                        yield chunk
                finally:
                    await close_upstream()

            return StreamingResponse(
                upstream_body_iter(),
                status_code=upstream_resp.status_code,
                headers=_strip_response_headers(dict(upstream_resp.headers)),
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(close_upstream),
            )
        try:
            content = await upstream_resp.aread()
        finally:
            try:
                await upstream_resp.aclose()
            finally:
                await upstream_client.aclose()
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=_strip_response_headers(dict(upstream_resp.headers)),
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # Reject requests a local model cannot serve before loading it, so a
    # mistaken request never evicts a working model just to fail.
    _check_local_capabilities(rt, model_query, body, target_path)

    # Local model — router manages llama-server lifecycle. The in-flight count
    # spans the ENTIRE request lifetime from here until the forwarded response
    # (including a streamed body consumed after this handler returns) is done,
    # because that whole window is time the request is using the GPU. The
    # auto-tuner's abort hook keys off this counter; if it drops to zero while
    # generation is still running, a sweep will restart the backend out from
    # under the user.
    request_entered_at = time.monotonic()
    resources = getattr(request.app.state, "resources", None)
    lease_context = (
        resources.acquire("local-inference", exclusive=False) if resources is not None else None
    )
    if lease_context is not None:
        await lease_context.__aenter__()
    rt.inflight += 1
    streaming_response_started = False
    resolved_at: float | None = None
    # Set once the request has resolved to a local model, so the router can
    # answer "is this specific model still serving?" — which _evict_for and
    # rebuild_model need to drain an incumbent instead of killing its
    # generation mid-stream. The global counter cannot answer that: the
    # evicting request itself holds it too. The acquisition itself happens
    # inside ensure_active, atomically with the readiness check.
    acquired_model: str | None = None
    try:
        try:
            model, srv = await rt.ensure_active(model_query, acquire=True)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_query!r}") from None
        except StartupFailureError as e:
            # Detailed command/log diagnostics are available only on the
            # authenticated admin load route, not the public inference API.
            return JSONResponse(status_code=e.http_status, content=e.to_dict())
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        acquired_model = model.name
        # Queue timing: how long this request waited before it even resolved
        # to a model (router lock, eviction drain, cold start join).
        resolved_at = time.monotonic()
        try:
            rt.timings.record_queue_wait(resolved_at - request_entered_at, model.name)
        except Exception:  # noqa: BLE001 - metrics must never break serving
            log.debug("queue-timing record failed", exc_info=True)
        # Tell the background tuner this model was actually used by a real request.
        # Upstream models do not reach this point, so only local models can become
        # auto-tune candidates.
        tuner = getattr(request.app.state, "tuner", None)
        if tuner is not None:
            tuner.bump_use(model.name)
        target_url = f"{srv.plan.backend_url}{target_path}"
        want_stream = streaming_ok and bool(body.get("stream"))
        body_bytes = _local_request_body(body, target_path)
        fwd_headers = {"Content-Type": "application/json"}

        async def _complete() -> None:
            """Mark the completion of request handling as router activity.

            This is called after the response body has been fully sent, so a
            long generation against an already-warm model still counts as
            activity and prevents the auto-tuner from starting.
            """
            rt.last_activity = time.time()

        if want_stream:
            policy = request.app.state.cfg.server
            first_timeout = policy.inference_first_response_seconds
            idle_timeout = policy.inference_stream_idle_seconds
            first_deadline = time.monotonic() + first_timeout if first_timeout else None
            client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
            try:
                req = client.build_request(
                    "POST",
                    target_url,
                    content=body_bytes,
                    headers=fwd_headers,
                )
                upstream = await asyncio.wait_for(
                    client.send(req, stream=True), first_timeout or None
                )
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                await client.aclose()
                raise HTTPException(
                    status_code=504,
                    detail="The model did not respond in time. Retry, or increase server.inference_first_response_seconds for long prefills.",
                ) from exc
            except httpx.HTTPError as exc:
                await client.aclose()
                raise HTTPException(
                    status_code=502,
                    detail="The model backend disconnected before responding. Retry the request.",
                ) from exc
            except BaseException:
                # Nothing was handed to Starlette, so streaming_response_started
                # stays False and the outer finally does the decrement. Close the
                # client here or it leaks: no response owns it yet.
                await client.aclose()
                raise

            released = False

            async def _release() -> None:
                """End the in-flight window: close upstream, close the client,
                drop the counter. Safe to call more than once.

                Relying on the BackgroundTask alone was not enough. Starlette
                only reaches ``await self.background()`` if streaming the body
                returned normally, so anything that makes stream_response raise
                (an upstream that dies mid-generation is the realistic one)
                skips it entirely and the counter is never decremented. Because
                autotune._tick() returns early whenever inflight > 0, a single
                leak silently disables background tuning for the rest of the
                process lifetime.
                """
                nonlocal released
                if released:
                    return
                released = True

                # Keep admission through connection cleanup. Shield normal
                # ASGI cancellation, and settle bookkeeping even if close fails.
                with anyio.CancelScope(shield=True):
                    try:
                        try:
                            await upstream.aclose()
                        finally:
                            await client.aclose()
                    except Exception:
                        log.debug("streaming proxy: connection cleanup failed", exc_info=True)
                    finally:
                        rt.inflight -= 1
                        if acquired_model is not None:
                            rt.release_model(acquired_model)
                        rt.last_activity = time.time()
                        if lease_context is not None:
                            await lease_context.__aexit__(None, None, None)

            observer = StreamMetricsObserver()
            ttft_recorded = False

            async def body_iter():
                nonlocal ttft_recorded
                timings = getattr(rt, "timings", None)
                try:
                    require_done = (
                        upstream.status_code < 400
                        and "text/event-stream" in upstream.headers.get("content-type", "")
                    )
                    iterator = _iter_sse_body(upstream, require_done=require_done).__aiter__()
                    first = True
                    while True:
                        if first:
                            timeout = (
                                max(0.001, first_deadline - time.monotonic())
                                if first_deadline is not None
                                else None
                            )
                        else:
                            timeout = idle_timeout or None
                        try:
                            chunk = await asyncio.wait_for(iterator.__anext__(), timeout)
                        except StopAsyncIteration:
                            break
                        first = False
                        if timings is not None and upstream.status_code < 400:
                            try:
                                observer.feed(chunk, time.monotonic())
                                if not ttft_recorded and observer.first_token_at is not None:
                                    timings.record_ttft(
                                        model.name, observer.first_token_at - request_entered_at
                                    )
                                    ttft_recorded = True
                            except Exception:
                                log.debug("stream metrics observation failed", exc_info=True)
                        # Forward the original bytes immediately, including malformed
                        # or oversized events that the observer ignores.
                        yield chunk
                    if timings is not None and observer.generation_tok_s is not None:
                        timings.record_generation_tok_s(model.name, observer.generation_tok_s)
                except (asyncio.TimeoutError, httpx.HTTPError):
                    error = {
                        "error": {
                            "message": "Generation was interrupted. The model stalled or disconnected; retry to continue.",
                            "type": "server_error",
                            "code": "inference_interrupted",
                        }
                    }
                    yield ("data: " + json.dumps(error) + "\n\n").encode()
                finally:
                    await _release()

            streaming_response_started = True
            return _LocalStreamingResponse(
                body_iter(),
                status_code=upstream.status_code,
                headers=_strip_response_headers(dict(upstream.headers)),
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                # Kept as a second path so a disconnect that finalizes the
                # generator late still settles promptly. _release is idempotent.
                background=BackgroundTask(_release),
            )
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(target_url, content=body_bytes, headers=fwd_headers)
        timings = getattr(rt, "timings", None)
        if timings is not None and r.status_code < 400:
            try:
                rate = generation_rate(json.loads(r.content))
                if rate is not None:
                    timings.record_generation_tok_s(model.name, rate)
            except (ValueError, TypeError, OverflowError):
                pass
        await _complete()
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_strip_response_headers(dict(r.headers)),
            media_type=r.headers.get("content-type", "application/json"),
        )
    finally:
        # Every non-streaming exit — success, failed load, forward error —
        # decrements here. The streaming path hands the decrement to
        # _release above, which runs after the body is consumed.
        if not streaming_response_started:
            rt.inflight -= 1
            if acquired_model is not None:
                rt.release_model(acquired_model)
            if lease_context is not None:
                await lease_context.__aexit__(None, None, None)
