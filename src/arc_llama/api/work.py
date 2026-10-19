"""Admin-only project and durable job endpoints for the Work tab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from arc_llama.workbench import files, sandbox
from arc_llama.workbench.runner import WorkManager
from arc_llama.workbench.store import WorkOwnerError


async def ensure_work(app: FastAPI) -> WorkManager:
    manager = getattr(app.state, "work", None)
    if manager is None:
        cfg = app.state.cfg
        host = cfg.server.host
        host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else "::1" if host == "::" else host
        host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        manager = WorkManager(
            Path(cfg.paths.state_dir).expanduser() / "work",
            f"http://{host}:{cfg.server.port}",
            cfg.server.admin_token,
        )
        try:
            await manager.start()
        except WorkOwnerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app.state.work = manager
    return manager


def register_work_routes(
    app: FastAPI, *, require_admin: Callable[..., Any], read_json_body: Callable[..., Any]
) -> None:
    async def manager(request: Request, _auth: None = Depends(require_admin)) -> WorkManager:
        return await ensure_work(request.app)

    def local_model(request: Request, value: Any) -> str:
        model = request.app.state.cfg.find_model(value) if isinstance(value, str) else None
        if (
            model is None
            or (model.recipe or {}).get("embedding")
            or (model.recipe or {}).get("reranking")
        ):
            raise HTTPException(
                status_code=400, detail="Choose a registered local chat model for Work."
            )
        return model.name

    @app.exception_handler(WorkError)
    async def work_error(request: Request, exc: WorkError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    async def guard(operation, *args, **kwargs):
        import inspect

        try:
            result = operation(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except KeyError as exc:
            raise WorkError("Work item not found", 404) from exc
        except ValueError as exc:
            raise WorkError(str(exc), 409) from exc
        except OSError as exc:
            raise WorkError(
                "Could not access Work storage. Check folder permissions and disk space.", 503
            ) from exc

    work_dependency = Depends(manager)

    @app.get("/admin/work")
    async def overview(work: WorkManager = work_dependency):
        return {
            "workspaces": work.store.records("workspace"),
            "jobs": [work.summary(j) for j in work.store.records("job")][:100],
            "commands_available": sandbox.available(),
            "excluded": sorted(files.EXCLUDED),
            "copy_limit_mb": 128,
        }

    @app.post("/admin/work/workspaces")
    async def create_workspace(request: Request, work: WorkManager = work_dependency):
        body = await read_json_body(request)
        body["model"] = local_model(request, body.get("model"))
        return await guard(work.workspace, body)

    @app.put("/admin/work/workspaces/{identity}")
    async def update_workspace(
        identity: str, request: Request, work: WorkManager = work_dependency
    ):
        body = await read_json_body(request)
        if "model" in body:
            body["model"] = local_model(request, body["model"])
        return await guard(work.workspace, body, identity)

    @app.post("/admin/work/jobs")
    async def create_job(request: Request, work: WorkManager = work_dependency):
        body = await read_json_body(request)
        if not isinstance(body.get("workspace_id"), str):
            raise HTTPException(status_code=400, detail="workspace_id is required")
        workspace = await guard(work.store.get, "workspace", body["workspace_id"])
        model = local_model(request, body.get("model", workspace["model"]))
        job = await guard(work.create_job, workspace["id"], body.get("task"), model)
        return work.summary(job)

    @app.get("/admin/work/jobs/{identity}")
    async def get_job(
        identity: str, after: int = Query(0, ge=0), work: WorkManager = work_dependency
    ):
        job = await guard(work.store.get, "job", identity)
        return {**work.summary(job), "events": work.store.events(identity, after)}

    @app.get("/admin/work/jobs/{identity}/changes")
    async def job_changes(identity: str, work: WorkManager = work_dependency):
        import asyncio

        job = await guard(work.store.get, "job", identity)
        if job["state"] == "settling":
            raise HTTPException(
                status_code=409,
                detail="The project copy is being replaced. Retry the review when it finishes.",
            )
        if job["state"] == "discarded":
            return {"files": [], "diff": "Project copy discarded"}
        root = await guard(work.job_dir, identity)
        receipt, diff = await guard(
            asyncio.to_thread, files.changes, root / "baseline", root / "tree"
        )
        return {"files": list(receipt), "diff": diff, "revision": files.revision(receipt)}

    @app.post("/admin/work/jobs/{identity}/{action}")
    async def job_action(
        identity: str, action: str, request: Request, work: WorkManager = work_dependency
    ):
        return await guard(work.action, identity, action, await read_json_body(request))


class WorkError(Exception):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status
