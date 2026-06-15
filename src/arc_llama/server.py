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
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from arc_llama.agent import run_agent
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.config import Config, load_config
from arc_llama.router import Router
from arc_llama.skills import load_skills
from arc_llama.upstream import UpstreamManager

log = logging.getLogger("arc_llama.server")


def _strip_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in (
            "transfer-encoding", "content-encoding", "content-length", "connection",
        )
    }


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state_dir = None
    if cfg.paths.state_dir:
        state_dir = Path(cfg.paths.state_dir).expanduser()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = Router(cfg, log_dir=state_dir)
        app.state.upstream_mgr = UpstreamManager(cfg.upstreams)
        app.state.cfg = cfg
        app.state.pending_confirmations: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}
        if state_dir:
            app.state.chat_store = ChatStore(state_dir / "chats")
        else:
            app.state.chat_store = ChatStore(Path(".arc_llama_chats"))
        load_skills(cfg.paths.skills_dir)
        try:
            yield
        finally:
            await app.state.router.shutdown()

    app = FastAPI(title="arc-llama", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
            data.append({
                "id": m.name,
                "object": "model",
                "owned_by": "arc-llama",
                "created": created,
                "metadata": {
                    "display_name": m.display_name,
                    "path": m.path,
                    "gpu_pci_slot": m.gpu_pci_slot,
                    "loaded": bool(srv and srv.is_running),
                    "aliases": list(m.aliases),
                },
            })
            for alias in m.aliases:
                if alias != m.name:
                    data.append({
                        "id": alias,
                        "object": "model",
                        "owned_by": "arc-llama-alias",
                        "created": created,
                        "metadata": {"canonical": m.name},
                    })
        # Upstream models
        try:
            upstream_models = await mgr.models()
        except Exception:
            upstream_models = []
        for u in upstream_models:
            data.append({
                "id": u.id,
                "object": "model",
                "owned_by": f"upstream:{u.upstream_name}",
                "created": 0,
                "metadata": {
                    "upstream": u.upstream_name,
                    **u.metadata,
                },
            })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat_or_completions(request: Request):
        return await _proxy_post(request, request.url.path)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)

    @app.post("/v1/agent")
    async def agent_endpoint(request: Request):
        """Run the local coding agent and stream tool-execution events.

        Request body:
            {
                "model": "model-id",
                "task": "user task",
                "auto_confirm": false,
                "max_turns": 30,
                "root": "/optional/project/root"
            }

        Returns a text/event-stream of JSON objects:
            {"type": "status", "message": "..."}
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
        max_turns = int(body.get("max_turns", 30))
        root = Path(body.get("root", ".")).resolve()

        run_id = str(uuid.uuid4())
        pending = request.app.state.pending_confirmations
        confirm_event = asyncio.Event()
        confirm_result: dict[str, bool] = {"approved": False}
        pending[run_id] = (confirm_event, confirm_result)

        base_url = f"http://{cfg.server.host}:{cfg.server.port}"
        chat_store: ChatStore = request.app.state.chat_store

        async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
            await confirm_event.wait()
            return confirm_result["approved"]

        async def event_stream() -> AsyncIterator[str]:
            try:
                async for event in run_agent(
                    task=task,
                    model=model,
                    base_url=base_url,
                    root=root,
                    auto_confirm=auto_confirm,
                    confirm_callback=confirm_callback,
                    max_turns=max_turns,
                    chat_store=chat_store,
                ):
                    if event.get("type") == "confirm_required":
                        event = {**event, "run_id": run_id}
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                pending.pop(run_id, None)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/v1/agent/{run_id}/confirm")
    async def confirm_agent_run(run_id: str, request: Request) -> dict[str, bool]:
        """Approve or deny a pending agent tool confirmation.

        Request body: {"approved": true|false}
        """
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        entry = request.app.state.pending_confirmations.get(run_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Run not found or not awaiting confirmation")

        event, result = entry
        result["approved"] = bool(body.get("approved", False))
        event.set()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Chat history persistence
    # ------------------------------------------------------------------

    @app.get("/v1/chats")
    async def list_chats(request: Request) -> dict[str, Any]:
        """Return a list of chat summaries ordered by most recently updated first."""
        store: ChatStore = request.app.state.chat_store
        chats = store.list_chats()
        return {"object": "list", "data": [c.summary() for c in chats]}

    @app.post("/v1/chats")
    async def create_chat(request: Request) -> dict[str, Any]:
        """Create a new chat.

        Body: {"id": "optional-id", "title": "optional title"}
        If no id is provided a UUID is generated.
        """
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        chat_id = body.get("id") or str(uuid.uuid4())
        title = body.get("title") or "Untitled chat"
        store: ChatStore = request.app.state.chat_store
        try:
            chat = store.create(chat_id, title)
        except FileExistsError:
            raise HTTPException(status_code=409, detail=f"Chat already exists: {chat_id}") from None
        return chat.to_dict()

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
        """Append messages or update a chat's title without replacing everything.

        Body:
            {
                "title": "new title",          // optional
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

    # ------------------------------------------------------------------
    # Admin (used by the web UI / TUI)
    # ------------------------------------------------------------------

    @app.get("/admin/status")
    async def admin_status(request: Request) -> dict:
        rt: Router = request.app.state.router
        c: Config = request.app.state.cfg
        models = []
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            r = m.recipe or {}
            models.append({
                "name": m.name,
                "display_name": m.display_name,
                "path": m.path,
                "gpu_pci_slot": m.gpu_pci_slot,
                "port": m.port,
                "loaded": bool(srv and srv.is_running),
                "ctx": r.get("ctx"),
                "cache_type_k": r.get("cache_type_k"),
                "cache_type_v": r.get("cache_type_v"),
                "kv_class": m.kv_class,
                "aliases": list(m.aliases),
            })
        gpus = [{
            "pci_slot": g.pci_slot,
            "sycl_index": g.sycl_index,
            "arch": g.arch,
            "vram_mb": g.vram_mb,
            "name": g.name,
            "enabled": g.enabled,
        } for g in c.gpus]
        mgr: UpstreamManager = request.app.state.upstream_mgr
        return {
            "server": {
                "host": c.server.host,
                "port": c.server.port,
                "single_resident": c.server.single_resident,
            },
            "gpus": gpus,
            "models": models,
            "upstreams": mgr.upstreams_status(),
        }

    @app.post("/admin/load/{name}")
    async def admin_load(name: str, request: Request) -> dict:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(status_code=400, detail=f"Upstream model cannot be loaded locally: {name!r}")
        try:
            model, srv = await rt.ensure_active(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}") from None
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return {"name": model.name, "loaded": srv.is_running}

    @app.post("/admin/stop/{name}")
    async def admin_stop(name: str, request: Request) -> dict:
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(status_code=400, detail=f"Upstream model cannot be stopped locally: {name!r}")
        if name not in {m.name for m in rt.all_models()}:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        was_running = await rt.stop_one(name)
        return {"name": name, "was_running": was_running, "loaded": False}

    @app.post("/admin/stop-all")
    async def admin_stop_all(request: Request) -> dict:
        rt: Router = request.app.state.router
        stopped = await rt.stop_all()
        return {"stopped": stopped}

    @app.post("/admin/parse-pdf")
    async def admin_parse_pdf(file: UploadFile) -> dict:
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
    async def admin_edit_model(name: str, request: Request) -> dict:
        """Update a model's recipe in-place.

        Body is a partial recipe dict — only provided fields change. Recognised
        fields: `ctx`, `cache_type_k`, `cache_type_v`, `parallel`, `kv_class`,
        `spec_type`, `ubatch_size`.
        If the model is currently loaded, the server is stopped first; callers
        decide whether to reload it afterwards via /admin/load.
        """
        from arc_llama.config import default_config_path
        from arc_llama.recipes import KVCacheType
        c: Config = request.app.state.cfg
        rt: Router = request.app.state.router
        mgr: UpstreamManager = request.app.state.upstream_mgr
        if mgr.find_model(name) is not None:
            raise HTTPException(status_code=400, detail=f"Upstream model cannot be edited locally: {name!r}")
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
        valid_classes = {"default", "moe_a3b", "qwen3_27b_dense", "gemma_swa"}
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
                raise HTTPException(status_code=400, detail="ubatch_size must be an integer") from None
            if not (1 <= ub <= 4096):
                raise HTTPException(status_code=400, detail="ubatch_size must be 1..4096")
            recipe["ubatch_size"] = ub
            changed.append("ubatch_size")
        if not changed:
            raise HTTPException(status_code=400, detail="no recognised fields to edit")
        model.recipe = recipe
        try:
            c.save(default_config_path())
        except OSError as e:
            log.warning("edit %s: persist failed: %s", name, e)
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
    async def admin_scan(request: Request) -> dict:
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
                c.save(default_config_path())
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
            upstream_resp = await mgr.proxy(
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
            async def close_upstream() -> None:
                await upstream_resp.aclose()
            return StreamingResponse(
                upstream_resp.aiter_raw(),
                status_code=upstream_resp.status_code,
                headers=_strip_response_headers(dict(upstream_resp.headers)),
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(close_upstream),
            )
        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=_strip_response_headers(dict(upstream_resp.headers)),
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # Local model — router manages llama-server lifecycle.
    try:
        model, srv = await rt.ensure_active(model_query)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_query!r}") from None
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    target_url = f"{srv.plan.backend_url}{target_path}"
    want_stream = streaming_ok and bool(body.get("stream"))
    fwd_headers = {"Content-Type": "application/json"}
    if want_stream:
        client = httpx.AsyncClient(timeout=None)
        req = client.build_request(
            "POST", target_url, content=body_bytes, headers=fwd_headers,
        )
        upstream = await client.send(req, stream=True)

        async def close_upstream() -> None:
            await upstream.aclose()
            await client.aclose()

        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=_strip_response_headers(dict(upstream.headers)),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            background=BackgroundTask(close_upstream),
        )
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(target_url, content=body_bytes, headers=fwd_headers)
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_strip_response_headers(dict(r.headers)),
            media_type=r.headers.get("content-type", "application/json"),
        )
