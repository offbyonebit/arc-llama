"""OpenAI-compatible HTTP server.

Mounts on `cfg.server.host:cfg.server.port` and forwards requests to whichever
llama-server backend the router decides is the right one for the model id in
the request body.

Also exposes a small admin surface used by the bundled web UI and the TUI:

    GET  /admin/status        — full snapshot (gpus, models, who's loaded)
    POST /admin/load/{name}   — preload a model without sending a chat request
    POST /admin/stop/{name}   — stop one model's llama-server
    POST /admin/stop-all      — stop every running llama-server

The web UI itself is a single static page mounted at `/` when the static dir
ships with the install.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

# `request.form()` yields Starlette's UploadFile. fastapi.UploadFile is a
# *subclass* of it, so isinstance() against the FastAPI one silently misses
# every real upload — match the base class instead.
from starlette.datastructures import UploadFile as FormUploadFile

from arc_llama.agent import run_agent
from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.agent.repo_map import SemanticIndex
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.config import AudioModelConfig, Config, load_config
from arc_llama.router import Router
from arc_llama.skills import load_skills
from arc_llama.tts import engine_names as tts_engine_names
from arc_llama.tts import get_engine as get_tts_engine
from arc_llama.upstream import UpstreamManager

log = logging.getLogger("arc_llama.server")


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


def _loaded_model_names(rt: Router) -> list[str]:
    """Names of every backend — LLM or audio — that passed its health check.

    "Loaded" means the health check passed, not merely that a process exists:
    during a cold start (tens of seconds) or a crash-respawn the subprocess is
    alive but the port is not serving, and reporting that as loaded misleads
    dashboards and scripts that gate on it.
    """
    names = [m.name for m in rt.all_models()] + [m.name for m in rt.all_audio_models()]
    return [
        name
        for name in names
        if rt._servers.get(name) and rt._servers[name].is_running and rt._servers[name].ready
    ]


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


def create_app(cfg: Config | None = None, config_path: Path | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state_dir = None
    if cfg.paths.state_dir:
        state_dir = Path(cfg.paths.state_dir).expanduser()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = Router(cfg, log_dir=state_dir)
        app.state.upstream_mgr = UpstreamManager(cfg.upstreams)
        app.state.cfg = cfg
        app.state.started_at = time.time()
        pending_confirmations: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}
        pending_plan_approvals: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}
        app.state.pending_confirmations = pending_confirmations
        app.state.pending_plan_approvals = pending_plan_approvals
        if state_dir:
            app.state.chat_store = ChatStore(state_dir / "chats")
            app.state.checkpoint_store = CheckpointStore(state_dir / "checkpoints")
            app.state.semantic_index = SemanticIndex(state_dir / "semantic_index")
        else:
            app.state.chat_store = ChatStore(Path(".arc_llama_chats"))
            app.state.checkpoint_store = CheckpointStore(Path(".arc_llama_checkpoints"))
            app.state.semantic_index = SemanticIndex(Path(".arc_llama_semantic_index"))
        load_skills(cfg.paths.skills_dir)
        app.state.mcp_manager = MCPClientManager(cfg.active_mcp_servers())
        tuner: Any | None = None
        if getattr(cfg, "tune", None) and cfg.tune.auto:
            from arc_llama import __version__
            from arc_llama.autotune import start_autotuner
            from arc_llama.config import default_config_path

            save_path = config_path or default_config_path()

            def _save_cfg() -> None:
                cfg.save(save_path)

            tuner = await start_autotuner(
                cfg, app.state.router, version=__version__, on_save=_save_cfg
            )
            app.state.tuner = tuner
        try:
            await app.state.mcp_manager.start()
            yield
        finally:
            if tuner is not None:
                await tuner.stop()
            await app.state.mcp_manager.stop()
            await app.state.router.shutdown()

    app = FastAPI(title="arc-llama", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Liveness probe for the arc-llama router itself."""
        rt: Router = request.app.state.router
        uptime = time.time() - request.app.state.started_at
        # "Loaded" means the health check passed, not merely that a process
        # exists: during a cold start (tens of seconds) or a crash-respawn the
        # subprocess is alive but the port is not serving, and reporting that
        # as loaded misleads dashboards and scripts that gate on it.
        loaded = _loaded_model_names(rt)
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
        loaded = _loaded_model_names(rt)
        return {
            "uptime_seconds": round(uptime, 2),
            "loads": rt.metrics["loads"],
            "stops": rt.metrics["stops"],
            "load_errors": rt.metrics["load_errors"],
            "last_load_at": rt.metrics["last_load_at"],
            "last_error": rt.metrics["last_error"],
            "active_models": loaded,
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

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        data = []
        # Local models
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            data.append(
                {
                    "id": m.name,
                    "object": "model",
                    "owned_by": "arc-llama",
                    "created": 0,
                    "metadata": {
                        "display_name": m.display_name,
                        "path": m.path,
                        "gpu_pci_slot": m.gpu_pci_slot,
                        "loaded": bool(srv and srv.is_running and srv.ready),
                        "aliases": list(m.aliases),
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
                            "created": 0,
                            "metadata": {"canonical": m.name},
                        }
                    )
        # Audio models (transcription and speech backends)
        for am in rt.all_audio_models():
            srv = rt._servers.get(am.name)
            data.append(
                {
                    "id": am.name,
                    "object": "model",
                    "owned_by": "arc-llama-audio",
                    "created": 0,
                    "metadata": {
                        "display_name": am.display_name,
                        "path": am.path,
                        "gpu_pci_slot": am.gpu_pci_slot,
                        "loaded": bool(srv and srv.is_running and srv.ready),
                        "aliases": list(am.aliases),
                        "engine": am.engine,
                        "task": am.task,
                        "mode": am.mode,
                    },
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

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        """OpenAI-compatible speech-to-text, served by a llama-server backend.

        Accepts both shapes the backend accepts: a `multipart/form-data` upload
        (what Home Assistant, Open WebUI and the OpenAI SDKs send) and a JSON
        body naming a server-local path.
        """
        return await _proxy_audio_post(request, "/v1/audio/transcriptions", task="asr")

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request):
        """OpenAI-compatible text-to-speech.

        Takes OpenAI's body — `input`, `voice`, `response_format`, `speed`,
        `instructions` — and answers with the encoded audio as raw bytes, so
        the OpenAI SDKs and Home Assistant's TTS platform work unmodified.
        Which engine actually synthesises it is a property of the registered
        model, not of this route.
        """
        return await _proxy_speech_post(request)

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
                "profile": "optional-profile-name" // must match the server's active profile
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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        model = body.get("model")
        task = body.get("task")
        if not model or not task:
            raise HTTPException(status_code=400, detail="'model' and 'task' are required")

        auto_confirm = bool(body.get("auto_confirm", False))
        if auto_confirm:
            await _require_admin(request)
        plan_mode = bool(body.get("plan_mode", False))
        max_turns = int(body.get("max_turns", 30))
        folder = body.get("folder") if body.get("folder") is not None else ""
        root_path = body.get("root") or cfg.agent.root
        root = Path(root_path).expanduser().resolve()
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

        async def event_stream() -> AsyncIterator[str]:
            transcript: list[ChatMessage] = [ChatMessage(role="user", content=task)]
            try:
                async for event in run_agent(
                    task=task,
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
                    chat_store=chat_store,
                    extra={"semantic_index": semantic_index},
                ):
                    if event.get("type") == "confirm_required":
                        event = {**event, "run_id": run_id}
                    if event.get("type") == "plan":
                        event = {**event, "run_id": run_id}
                    if event.get("type") == "checkpoint":
                        event = {**event, "run_id": run_id}
                    yield f"data: {json.dumps(event)}\n\n"

                    if event.get("type") == "assistant" and event.get("content"):
                        transcript.append(ChatMessage(role="assistant", content=event["content"]))
                    elif event.get("type") == "tool_result":
                        label = event.get("name", "tool")
                        content = event.get("content", "")
                        transcript.append(ChatMessage(role="tool", content=f"{label}:\n{content}"))
            finally:
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

    @app.post("/v1/agent/{run_id}/confirm")
    async def confirm_agent_run(
        run_id: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict[str, bool]:
        """Approve or deny a pending agent tool confirmation.

        Request body: {"approved": true|false}
        """
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

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

    @app.get("/v1/chats")
    async def list_chats(request: Request, folder: str | None = Query(None)) -> dict[str, Any]:
        """Return a list of chat summaries ordered by most recently updated first.

        Pass ``?folder=...`` to filter; omit for all folders. Pass
        ``?folder=`` for the root/legacy folder only.
        """
        store: ChatStore = request.app.state.chat_store
        chats = store.list_chats(folder=folder)
        return {"object": "list", "data": [c.summary() for c in chats]}

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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        chat_id = body.get("id") or str(uuid.uuid4())
        title = body.get("title") or "Untitled chat"
        folder = body.get("folder") if body.get("folder") is not None else ""
        store: ChatStore = request.app.state.chat_store
        try:
            chat = store.create(chat_id, title, folder=folder)
        except FileExistsError:
            raise HTTPException(status_code=409, detail=f"Chat already exists: {chat_id}") from None
        return chat.to_dict()

    @app.post("/v1/chats/search")
    async def search_chats(request: Request) -> dict[str, Any]:
        """Search chat titles and messages.

        Body: {"query": "string", "limit": 20}
        Returns matching chats with the indices of matching messages.
        """
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        limit = int(body.get("limit", 20))
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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        chats = body.get("chats")
        if not isinstance(chats, list):
            raise HTTPException(status_code=400, detail="'chats' must be an array")
        store: ChatStore = request.app.state.chat_store
        result = store.import_chats(chats, overwrite=bool(body.get("overwrite", False)))
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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        store: ChatStore = request.app.state.chat_store
        chat = store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        if "title" in body:
            chat.title = str(body["title"])
        if "messages" in body and isinstance(body["messages"], list):
            chat.messages = [ChatMessage.from_dict(m) for m in body["messages"]]
        store.save(chat)
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
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        store: ChatStore = request.app.state.chat_store
        chat = store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        if "title" in body:
            chat.title = str(body["title"])
        if "folder" in body and body["folder"] is not None:
            chat.folder = str(body["folder"])
        if "messages" in body and isinstance(body["messages"], list):
            for m in body["messages"]:
                chat.messages.append(ChatMessage.from_dict(m))
        store.save(chat)
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
        models = []
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            r = m.recipe or {}
            running = bool(srv and srv.is_running)
            loaded = bool(srv and srv.is_running and srv.ready)
            models.append(
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "path": m.path,
                    "gpu_pci_slot": m.gpu_pci_slot,
                    "port": m.port,
                    "loaded": loaded,
                    "pid": getattr(getattr(srv, "process", None), "pid", None) if running else None,
                    "ctx": r.get("ctx"),
                    "cache_type_k": r.get("cache_type_k"),
                    "cache_type_v": r.get("cache_type_v"),
                    "flash_attn": r.get("flash_attn"),
                    "ubatch_size": r.get("ubatch_size"),
                    "batch_size": r.get("batch_size"),
                    "kv_class": m.kv_class,
                    "aliases": list(m.aliases),
                    "tune_state": m.tune_state,
                    "tuned_at": m.tuned_at,
                    "tune_error": m.tune_error,
                    "tune_fingerprint": m.tune_fingerprint,
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
            }
            for g in c.gpus
        ]
        audio_models = []
        for am in rt.all_audio_models():
            srv = rt._servers.get(am.name)
            running = bool(srv and srv.is_running)
            audio_models.append(
                {
                    "name": am.name,
                    "display_name": am.display_name,
                    "path": am.path,
                    "gpu_pci_slot": am.gpu_pci_slot,
                    "port": am.port,
                    "loaded": bool(srv is not None and running and srv.ready),
                    "pid": getattr(getattr(srv, "process", None), "pid", None) if running else None,
                    "engine": am.engine,
                    "mmproj": am.audio_recipe().mmproj,
                    "task": am.task,
                    "mode": am.mode,
                    "always_resident": am.always_resident,
                    "aliases": list(am.aliases),
                    "launchable": srv is not None,
                    "launch_error": getattr(rt, "audio_launch_errors", {}).get(am.name),
                }
            )
        mgr: UpstreamManager = request.app.state.upstream_mgr
        return {
            "server": {
                "host": c.server.host,
                "port": c.server.port,
                "single_resident": c.server.single_resident,
                "auto_tune": c.tune.auto,
            },
            "gpus": gpus,
            "models": models,
            "audio_models": audio_models,
            "voices": [
                {
                    "name": v.name,
                    "display_name": v.display_name,
                    "mode": "clone" if v.ref_audio else ("design" if v.instruct else "auto"),
                    "language": v.language,
                    "models": list(v.models),
                    "aliases": list(v.aliases),
                }
                for v in c.voices
            ],
            "upstreams": mgr.upstreams_status(),
        }

    @app.post("/admin/load/{name}")
    async def admin_load(
        name: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict:
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
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        # ensure_active only returns once wait_ready has passed, so ready is
        # the accurate claim here; is_running would also report a server that
        # crashed between readiness and this line.
        return {"name": model.name, "loaded": bool(srv.is_running and srv.ready)}

    @app.post("/admin/stop/{name}")
    async def admin_stop(
        name: str, request: Request, _auth: None = Depends(_require_admin)
    ) -> dict:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(
                status_code=400, detail=f"Upstream model cannot be stopped locally: {name!r}"
            )
        known = {m.name for m in rt.all_models()} | {m.name for m in rt.all_audio_models()}
        if name not in known:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        was_running = await rt.stop_one(name)
        return {"name": name, "was_running": was_running, "loaded": False}

    @app.post("/admin/stop-all")
    async def admin_stop_all(request: Request, _auth: None = Depends(_require_admin)) -> dict:
        rt: Router = request.app.state.router
        stopped = await rt.stop_all()
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
        `spec_type`, `ubatch_size`, `batch_size`, `flash_attn` (null clears),
        `n_cpu_moe` (null or 0 clears), `override_tensor` (list of regex
        patterns, null clears). When `override_tensor` is set, `n_cpu_moe` is
        cleared and vice versa: the two flags are alternative means to the
        same end and applying both would double-count the offload.
        If the model is currently loaded, the server is stopped first; callers
        decide whether to reload it afterwards via /admin/load.
        """
        from arc_llama.config import default_config_path
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
            v = str(body["spec_type"])
            recipe["spec_type"] = v
            changed.append("spec_type")
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
        if not changed:
            raise HTTPException(status_code=400, detail="no recognised fields to edit")
        previous_recipe = model.recipe
        model.recipe = recipe
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
        from arc_llama.config import default_config_path
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

    # ------------------------------------------------------------------
    # Static web UI (optional; only mounted if the static dir is present)
    # ------------------------------------------------------------------
    static_dir = Path(__file__).parent / "static"

    @app.get("/chat")
    async def chat_page() -> Response:
        return FileResponse(static_dir / "chat.html")

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


# Uploaded audio is buffered in memory to be re-encoded, so the request body
# needs its own ceiling. A minute of 16 kHz mono WAV is under 2 MB; this is
# generous for a dictation or a long meeting recording without letting a
# bogus Content-Length exhaust the host.
_MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

# A speech request is text, so the ceiling is small on purpose: it is the one
# audio route where a client can ask for unbounded GPU time with a few bytes.
_MAX_SPEECH_BODY_BYTES = 64 * 1024

# Generous, because it covers a first synthesis on a cold diffusion model as
# well as a long transcription. Not None: an unbounded wait on a wedged
# backend holds an in-flight slot open forever, which is what stops the
# auto-tuner from ever running again.
_AUDIO_REQUEST_TIMEOUT = 600.0

_NO_ASR_MODELS_DETAIL = (
    "No transcription models are registered. Add one with "
    "`arc-llama audio add <model.gguf> --mmproj mmproj-<model>.gguf`, or with "
    "`--from-hf ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0` to download a pair."
)

_NO_TTS_MODELS_DETAIL = (
    "No text-to-speech models are registered. Add one with "
    "`arc-llama audio add k2-fsa/OmniVoice --task tts --engine omnivoice`."
)

# Qwen3-ASR's native output framing. `<asr_text>` is a real token in its
# vocabulary, and the model prefixes each transcript with a detected-language
# announcement: `language English<asr_text>the actual words`. llama.cpp
# forwards that verbatim (ggml-org/llama.cpp#26749).
_ASR_TEXT_MARKER = "<asr_text>"
_ASR_AUDIO_TOKENS = ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>")


def strip_asr_markers(text: str) -> str:
    """Remove a transcription model's output framing from *text*.

    Everything up to and including the last `<asr_text>` is the model
    announcing what it is about to do ("language English"), not speech that
    anyone said. A voice assistant matching intents against the raw string
    fails on every utterance, so drop it.

    Splitting on the *last* marker rather than the first is deliberate: it
    degrades to a no-op on any model that never emits one, and a transcript
    that genuinely contained the literal token would be truncated — which
    cannot happen, because it is a reserved token the tokenizer never
    produces from audio.
    """
    if _ASR_TEXT_MARKER in text:
        text = text.rsplit(_ASR_TEXT_MARKER, 1)[1]
    for token in _ASR_AUDIO_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def _sanitize_transcription(content: bytes) -> bytes:
    """Apply ``strip_asr_markers`` to a transcription response body.

    Returns the body untouched if it is not the JSON `{"text": ...}` shape —
    an error payload or an unexpected schema is the backend's to explain, and
    rewriting it would only obscure the real failure.
    """
    try:
        body = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        return content
    cleaned = strip_asr_markers(body["text"])
    if cleaned == body["text"]:
        return content
    body["text"] = cleaned
    return json.dumps(body).encode("utf-8")


def _resolve_audio_model(cfg: Config, query: str, task: str):
    """Pick the audio model a request is asking for.

    An empty `model` resolves to the sole registered model for the task when
    there is exactly one — clients in the wild hardcode OpenAI's `whisper-1`
    or `tts-1` or omit the field entirely, and a single-model box has no
    ambiguity to protect. With several registered the request has to say which.
    """
    candidates = [m for m in cfg.audio_models if m.task == task]
    if query:
        found = cfg.find_audio_model(query)
        if found is not None:
            return found
        if len(candidates) == 1:
            log.info(
                "audio request named unknown model %r; using the only "
                "registered %s model %r. Add %r to its aliases to silence this.",
                query, task, candidates[0].name, query,
            )
            return candidates[0]
        return None
    if len(candidates) == 1:
        return candidates[0]
    return None


def _require_audio_model(
    cfg: Config, query: str, task: str, none_registered_detail: str
) -> AudioModelConfig:
    """``_resolve_audio_model``, but as an HTTP error when it cannot.

    The two audio endpoints fail the same way for the same reasons, and the
    distinction that matters to the caller — "you have none of these" (501,
    here is how to add one) versus "not that one" (404, here are the ones you
    have) — is worth keeping identical between them.
    """
    candidates = [m for m in cfg.audio_models if m.task == task]
    if not candidates:
        raise HTTPException(status_code=501, detail=none_registered_detail)
    model = _resolve_audio_model(cfg, query, task)
    if model is None:
        known = ", ".join(m.name for m in candidates)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown {task} model: {query!r}. Registered {task} models: {known}",
        )
    return model


async def _forward_audio(
    request: Request,
    model: AudioModelConfig,
    target_path: str,
    build_kwargs: Callable[[AudioModelConfig], dict[str, Any]],
    *,
    want_stream: bool = False,
    unavailable_detail: str = "",
    sanitize: Callable[[bytes], bytes] | None = None,
):
    """Start the backend serving *model* and forward one request to it.

    Kept apart from ``_proxy_post`` rather than folded into it. That path is
    JSON-only by construction — it reads `model` out of a parsed body and
    forwards the original bytes — while the audio routes have to understand
    multipart uploads and binary responses, and this function carries a set of
    hard-won in-flight/cancellation invariants that are not worth re-opening
    for a second body format.

    ``build_kwargs`` receives the *resolved* model, because the backend only
    answers to its own id: a client may address a model by any alias arc-llama
    accepts, and both the multipart `model` field and the JSON body have to be
    rewritten to the canonical one before the request goes out.
    """
    rt: Router = request.app.state.router

    rt.inflight += 1
    streaming_response_started = False
    acquired_model: str | None = None
    try:
        try:
            resolved, srv = await rt.ensure_active(model.name, acquire=True)
        except KeyError:
            raise HTTPException(
                status_code=503,
                detail=unavailable_detail
                or f"Audio model {model.name!r} is registered but not launchable.",
            ) from None
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        acquired_model = resolved.name
        if not isinstance(resolved, AudioModelConfig):
            # ensure_active resolves by name across both registries, and
            # find_any_model tries LLMs first. Registration refuses a name
            # that is already taken, so this needs a hand-edited config —
            # in which case an audio request is about to be answered by a
            # chat model, and saying so beats transcribing gibberish.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"{resolved.name!r} is registered in both [[models]] and "
                    "[[audio_models]]; rename one so audio requests reach the "
                    "audio backend."
                ),
            )
        target_url = f"{srv.plan.backend_url}{target_path}"
        request_kwargs = build_kwargs(resolved)

        if want_stream:
            # Streamed deltas are forwarded raw, so `strip_asr_markers` does
            # not apply: the framing arrives split across chunks and rewriting
            # it would mean buffering the stream, which defeats the point of
            # asking for one. Clients that need clean text should not stream.
            client = httpx.AsyncClient(timeout=None)
            try:
                req = client.build_request("POST", target_url, **request_kwargs)
                upstream = await client.send(req, stream=True)
            except BaseException:
                await client.aclose()
                raise

            released = False

            async def _release() -> None:
                """Close the upstream and end the in-flight window, once.

                Same reasoning as the streaming chat path: the counter must
                drop even when the body generator raises, because a leaked
                in-flight count disables background auto-tune permanently.
                """
                nonlocal released
                if released:
                    return
                released = True
                if rt.inflight > 0:
                    rt.inflight -= 1
                else:
                    log.warning("audio proxy: inflight already 0 at release; not decrementing")
                if acquired_model is not None:
                    rt.release_model(acquired_model)
                rt.last_activity = time.time()
                try:
                    await upstream.aclose()
                except Exception:
                    log.debug("audio proxy: upstream close failed", exc_info=True)
                try:
                    await client.aclose()
                except Exception:
                    log.debug("audio proxy: client close failed", exc_info=True)

            async def body_iter():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await _release()

            streaming_response_started = True
            return StreamingResponse(
                body_iter(),
                status_code=upstream.status_code,
                headers=_strip_response_headers(dict(upstream.headers)),
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(_release),
            )

        async with httpx.AsyncClient(timeout=_AUDIO_REQUEST_TIMEOUT) as client:
            r = await client.post(target_url, **request_kwargs)
        rt.last_activity = time.time()
        content = r.content
        if sanitize is not None and r.status_code == 200:
            content = sanitize(content)
        return Response(
            content=content,
            status_code=r.status_code,
            headers=_strip_response_headers(dict(r.headers)),
            media_type=r.headers.get("content-type", "application/json"),
        )
    finally:
        if not streaming_response_started:
            rt.inflight -= 1
            if acquired_model is not None:
                rt.release_model(acquired_model)


async def _proxy_audio_post(request: Request, target_path: str, task: str):
    """Forward a transcription request to the llama-server backend serving it.

    Multipart bodies are re-encoded rather than passed through, because the
    `model` field has to be rewritten to the backend's own id.
    """
    cfg: Config = request.app.state.cfg

    content_type = request.headers.get("content-type", "")
    is_multipart = content_type.lower().startswith("multipart/form-data")

    # Refuse an oversized upload from the declared length, before the body is
    # read. Checking only after reading would mean buffering the very payload
    # the limit exists to refuse.
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > _MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Audio upload must be under "
                        f"{_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB"
                    ),
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None

    body_bytes = b""
    form_fields: dict[str, str] = {}
    upload: tuple[str, bytes, str] | None = None

    if is_multipart:
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid multipart body: {e}") from e
        try:
            for key, value in form.multi_items():
                if isinstance(value, FormUploadFile):
                    if key != "file":
                        continue
                    content = await value.read()
                    if len(content) > _MAX_AUDIO_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Audio upload must be under "
                                f"{_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB"
                            ),
                        )
                    upload = (
                        value.filename or "audio.wav",
                        content,
                        value.content_type or "application/octet-stream",
                    )
                else:
                    form_fields[key] = str(value)
        finally:
            await form.close()
        if upload is None:
            raise HTTPException(status_code=400, detail="Missing 'file' in multipart body")
        model_query = form_fields.get("model", "")
    else:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")
        model_query = str(body.get("model", ""))

    model = _require_audio_model(cfg, model_query, task, _NO_ASR_MODELS_DETAIL)

    want_stream = str(form_fields.get("stream", "")).lower() == "true"
    if want_stream and model.mode != "streaming":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {model.name!r} is configured mode='offline'; "
                "stream=true needs a model registered with --mode streaming."
            ),
        )

    def _kwargs(resolved: AudioModelConfig) -> dict[str, Any]:
        if is_multipart:
            assert upload is not None
            return {"data": {**form_fields, "model": resolved.name}, "files": {"file": upload}}
        parsed = json.loads(body_bytes) if body_bytes else {}
        parsed["model"] = resolved.name
        return {"json": parsed}

    return await _forward_audio(
        request,
        model,
        target_path,
        _kwargs,
        want_stream=want_stream,
        unavailable_detail=(
            f"Audio model {model.name!r} is registered but not launchable. "
            "Check that paths.llama_server points at a llama.cpp build with "
            "multimodal (--mmproj) support and that the entry's mmproj exists."
        ),
        sanitize=_sanitize_transcription if model.strip_asr_markers else None,
    )


async def _proxy_speech_post(request: Request):
    """Forward an OpenAI `/v1/audio/speech` request to its TTS backend.

    The OpenAI body is translated by the model's engine rather than here, so
    the endpoint stays the same shape whichever engine is loaded — that
    translation is the whole reason engines exist, and folding a second
    request format into this function would undo it.
    """
    cfg: Config = request.app.state.cfg

    body_bytes = await request.body()
    if len(body_bytes) > _MAX_SPEECH_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Speech request must be under {_MAX_SPEECH_BODY_BYTES // 1024} KB",
        )
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="'input' must be a non-empty string")

    model = _require_audio_model(cfg, str(body.get("model", "")), "tts", _NO_TTS_MODELS_DETAIL)

    engine = get_tts_engine(model.engine)
    if engine is None:
        # Only reachable via a hand-edited config: registration refuses an
        # unknown engine, and the loader drops entries it cannot serve.
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model {model.name!r} names TTS engine {model.engine!r}, which is "
                f"not registered. Known engines: {', '.join(tts_engine_names()) or '(none)'}."
            ),
        )

    return await _forward_audio(
        request,
        model,
        engine.speech_path,
        lambda resolved: {"json": engine.build_payload(resolved, body)},
        unavailable_detail=(
            f"TTS model {model.name!r} is registered but not launchable. "
            "Check `arc-llama doctor` — a Python engine needs `paths.tts_python` "
            "pointing at an interpreter that can import it."
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
                    async for chunk in upstream_resp.aiter_raw():
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

    # Local model — router manages llama-server lifecycle. The in-flight count
    # spans the ENTIRE request lifetime from here until the forwarded response
    # (including a streamed body consumed after this handler returns) is done,
    # because that whole window is time the request is using the GPU. The
    # auto-tuner's abort hook keys off this counter; if it drops to zero while
    # generation is still running, a sweep will restart the backend out from
    # under the user.
    rt.inflight += 1
    streaming_response_started = False
    # Set once the request has resolved to a local model, so the router can
    # answer "is this specific model still serving?" — which _evict_for and
    # rebuild_model need to drain an incumbent instead of killing its
    # generation mid-stream. The global counter cannot answer that: the
    # evicting request holds it too. The acquisition itself happens inside
    # ensure_active, atomically with the readiness check.
    acquired_model: str | None = None
    try:
        try:
            model, srv = await rt.ensure_active(model_query, acquire=True)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_query!r}") from None
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        acquired_model = model.name
        # Tell the background tuner this model was actually used by a real request.
        # Upstream models do not reach this point, so only local models can become
        # auto-tune candidates.
        tuner = getattr(request.app.state, "tuner", None)
        if tuner is not None:
            tuner.bump_use(model.name)
        target_url = f"{srv.plan.backend_url}{target_path}"
        want_stream = streaming_ok and bool(body.get("stream"))
        fwd_headers = {"Content-Type": "application/json"}

        async def _complete() -> None:
            """Mark the completion of request handling as router activity.

            This is called after the response body has been fully sent, so a
            long generation against an already-warm model still counts as
            activity and prevents the auto-tuner from starting.
            """
            rt.last_activity = time.time()

        if want_stream:
            client = httpx.AsyncClient(timeout=None)
            try:
                req = client.build_request(
                    "POST",
                    target_url,
                    content=body_bytes,
                    headers=fwd_headers,
                )
                upstream = await client.send(req, stream=True)
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

                # Drop the counter before awaiting anything. There is no await
                # between the guard above and this decrement, so once we are
                # here nothing can stop it -- not a client disconnect, not a
                # cancellation delivered during shutdown. Closing the socket
                # below is best effort; a leaked connection is recovered when
                # the process exits, whereas a leaked count is permanent and
                # silently disables auto-tune for the life of the process.
                if rt.inflight > 0:
                    rt.inflight -= 1
                else:
                    # Unreachable unless a future change double-releases. Say so
                    # rather than letting the counter go negative, which would
                    # wedge _deferred_restore's "wait for zero" loop.
                    log.warning("streaming proxy: inflight already 0 at release; not decrementing")
                if acquired_model is not None:
                    rt.release_model(acquired_model)
                rt.last_activity = time.time()

                try:
                    await upstream.aclose()
                except Exception:
                    log.debug("streaming proxy: upstream close failed", exc_info=True)
                try:
                    await client.aclose()
                except Exception:
                    log.debug("streaming proxy: client close failed", exc_info=True)

            async def body_iter():
                # The decrement belongs here rather than only in the
                # BackgroundTask: this finally runs when the body is fully
                # consumed, when the upstream errors mid-stream, and when the
                # generator is finalized after a client disconnect. It must not
                # run any earlier, because dropping the count while generation
                # is still live lets the autotuner restart the backend out from
                # under the request.
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await _release()

            streaming_response_started = True
            return StreamingResponse(
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
