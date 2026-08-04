"""Health, session-token and metrics endpoints."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from arc_llama.server_routers.common import require_admin

log = logging.getLogger("arc_llama.server")
router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe for the arc-llama router itself."""
    rt = request.app.state.router
    uptime = time.time() - request.app.state.started_at
    loaded = [m.name for m in rt.all_models() if rt._servers.get(m.name) and rt._servers[m.name].is_running]
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 2),
        "loaded_models": loaded,
        "loaded_model_count": len(loaded),
    }


@router.get("/admin/session-token")
async def admin_session_token(request: Request) -> dict[str, str | None]:
    """Hand the bundled first-party web UI its own admin token.

    Deliberately not gated by ``_require_admin`` -- it exists so the
    static page served by this same process can bootstrap itself without
    the user copy-pasting a token. Instead it's gated on the *connection*
    being loopback.
    """
    peer = request.client.host if request.client else ""
    if peer not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        raise HTTPException(status_code=403, detail="Loopback connections only")
    cfg = request.app.state.cfg
    return {"admin_token": cfg.server.admin_token}


@router.get("/admin/metrics", dependencies=[require_admin])
async def admin_metrics(request: Request) -> dict[str, Any]:
    """Operational counters and current GPU/model state."""
    rt = request.app.state.router
    c = request.app.state.cfg
    uptime = time.time() - request.app.state.started_at
    loaded = [m.name for m in rt.all_models() if rt._servers.get(m.name) and rt._servers[m.name].is_running]
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
