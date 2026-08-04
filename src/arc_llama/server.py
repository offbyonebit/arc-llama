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
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.agent.repo_map import SemanticIndex
from arc_llama.chat_store import ChatStore
from arc_llama.config import Config, load_config
from arc_llama.router import Router
from arc_llama.server_routers import admin, agent, chats, completions, health, models
from arc_llama.server_routers.common import _require_admin, _strip_response_headers
from arc_llama.server_routers.completions import _proxy_post
from arc_llama.skills import load_skills
from arc_llama.upstream import UpstreamManager

log = logging.getLogger("arc_llama.server")


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
        app.state.config_path = config_path
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

            tuner = start_autotuner(cfg, app.state.router, version=__version__, on_save=_save_cfg)
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

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(agent.router)
    app.include_router(chats.router)
    app.include_router(admin.router)

    static_dir = Path(__file__).parent / "static"

    @app.get("/chat")
    async def chat_page() -> FileResponse:
        return FileResponse(static_dir / "chat.html")

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


__all__ = [
    "create_app",
    "_require_admin",
    "_strip_response_headers",
    "_proxy_post",
    "httpx",
]
