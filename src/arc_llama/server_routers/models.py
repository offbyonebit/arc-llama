"""OpenAI /v1/models endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> dict:
    rt = request.app.state.router
    mgr = request.app.state.upstream_mgr
    data = []
    # Local models
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
