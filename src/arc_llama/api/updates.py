"""Package update notice requested independently from dashboard status."""

from typing import Any

from fastapi import Depends, FastAPI, Request

from arc_llama.update_notice import check_for_update


def register_update_routes(app: FastAPI, *, require_admin: Any) -> None:
    @app.get("/admin/update-notice")
    async def update_notice(request: Request, _auth: None = Depends(require_admin)) -> dict[str, Any]:
        return {"update": await check_for_update(request.app.state.cfg)}
