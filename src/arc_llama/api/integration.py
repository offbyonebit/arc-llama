"""Read-only frontend integration guidance endpoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Request


def register_integration_route(app: FastAPI, *, require_admin: Callable[..., Any]) -> None:
    """Register the copy-only frontend integration discovery endpoint."""

    @app.get("/admin/integration")
    async def admin_integration(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        """Read-only discovery for the dashboard's Connect-a-frontend panel.

        Returns the base URL a client should paste, loopback Ollama
        reachability, and registered upstreams. The bundled UI renders this
        as copy-only guidance; nothing here transmits credentials, mutates
        config, or touches external Open WebUI accounts. ``integration`` is
        imported lazily so importing ``arc_llama.server`` stays cheap.
        """
        from arc_llama.integration import integration_payload, probe_ollama
        from arc_llama.server import model_capabilities

        cfg = request.app.state.cfg
        ollama = await probe_ollama()
        router = request.app.state.router
        loaded = next((model for model in router.all_models()
                       if (srv := router._servers.get(model.name)) is not None
                       and srv.is_running and srv.ready and "chat" in model_capabilities(model)), None)
        return integration_payload(cfg, ollama, loaded_model=loaded, request_host=request.url.hostname)
