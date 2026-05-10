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

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from arc_llama.config import Config, load_config
from arc_llama.router import Router

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
        from pathlib import Path
        state_dir = Path(cfg.paths.state_dir).expanduser()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = Router(cfg, log_dir=state_dir)
        app.state.cfg = cfg
        try:
            yield
        finally:
            await app.state.router.shutdown()

    app = FastAPI(title="arc-llama", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict:
        rt: Router = request.app.state.router
        data = []
        for m in rt.all_models():
            srv = rt._servers.get(m.name)
            data.append({
                "id": m.name,
                "object": "model",
                "owned_by": "arc-llama",
                "created": 0,
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
                        "created": 0,
                        "metadata": {"canonical": m.name},
                    })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat_or_completions(request: Request):
        return await _proxy_post(request, request.url.path)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)

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
        return {
            "server": {
                "host": c.server.host,
                "port": c.server.port,
                "single_resident": c.server.single_resident,
            },
            "gpus": gpus,
            "models": models,
        }

    @app.post("/admin/load/{name}")
    async def admin_load(name: str, request: Request) -> dict:
        rt: Router = request.app.state.router
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
        if name not in {m.name for m in rt.all_models()}:
            raise HTTPException(status_code=404, detail=f"Unknown model: {name!r}")
        was_running = await rt.stop_one(name)
        return {"name": name, "was_running": was_running, "loaded": False}

    @app.post("/admin/stop-all")
    async def admin_stop_all(request: Request) -> dict:
        rt: Router = request.app.state.router
        stopped = await rt.stop_all()
        return {"stopped": stopped}

    @app.post("/admin/models/{name}/edit")
    async def admin_edit_model(name: str, request: Request) -> dict:
        """Update a model's recipe in-place.

        Body is a partial recipe dict — only provided fields change. Recognised
        fields: `ctx`, `cache_type_k`, `cache_type_v`, `parallel`, `kv_class`.
        If the model is currently loaded, the server is stopped first; callers
        decide whether to reload it afterwards via /admin/load.
        """
        from arc_llama.config import default_config_path
        from arc_llama.recipes import KVCacheType
        c: Config = request.app.state.cfg
        rt: Router = request.app.state.router
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
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


async def _proxy_post(request: Request, target_path: str, streaming_ok: bool = True):
    rt: Router = request.app.state.router
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    model_query = body.get("model", "")
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
