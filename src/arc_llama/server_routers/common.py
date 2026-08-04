"""Shared server helpers and admin dependencies."""
from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request

from arc_llama.config import Config


def _strip_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in (
            "transfer-encoding", "content-encoding", "content-length", "connection",
        )
    }


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


# Re-exported dependency for router modules.
require_admin = Depends(_require_admin)

__all__ = [
    "_strip_response_headers",
    "_require_admin",
    "require_admin",
]
