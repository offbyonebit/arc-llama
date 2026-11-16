"""Authenticated rollback using the existing runtime history and lifecycle."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from arc_llama import runtime_update


def _busy_reason(state: Any, *, own_lease: bool = False) -> str | None:
    rt = state.router
    if not own_lease and getattr(state, "runtime_rollback_running", False):
        return "A runtime rollback is already running."
    if getattr(rt, "_loading_futures", {}) or any(
        server.is_running and not server.ready for server in rt._servers.values()
    ):
        return "A model is loading. Wait for it to finish before rolling back."
    downloads = getattr(state, "downloads", None)
    if downloads and any(job.status in {"queued", "downloading", "registering", "cancelling"}
                         for job in downloads.jobs.values()):
        return "A download is running or queued. Finish or cancel it before rolling back."
    setup = getattr(state, "first_answer", None)
    if setup and setup.phase in {"checking", "runtime", "downloading", "registering", "loading"}:
        return "First-run setup is running. Finish or pause it before rolling back."
    tuner = getattr(state, "tuner", None)
    if tuner and tuner.is_sweep_running:
        return "A tune is running. Finish or stop it before rolling back."
    if getattr(rt, "inflight", 0):
        return "A model is answering. Finish or stop the request before rolling back."
    resources = getattr(state, "resources", None)
    if not own_lease and resources and (resources.exclusive_active or resources.active_leases):
        return "Another GPU operation is running. Wait for it to finish before rolling back."
    return None


def register_runtime_routes(
    app: FastAPI, *, require_admin: Callable[..., Any],
    read_json_body: Callable[[Request], Awaitable[dict[str, Any]]], config_path: Path,
) -> None:
    lock = asyncio.Lock()

    @app.get("/admin/runtime/history")
    async def history(request: Request, _auth: None = Depends(require_admin)) -> dict[str, Any]:
        state = request.app.state
        result = await asyncio.to_thread(runtime_update.runtime_history, state.cfg)
        result["busy_reason"] = _busy_reason(state)
        return result

    @app.post("/admin/runtime/rollback")
    async def rollback(request: Request, _auth: None = Depends(require_admin)) -> dict[str, Any]:
        body = await read_json_body(request)
        expected = body.get("expected_previous")
        if expected is not None and not isinstance(expected, str):
            raise HTTPException(status_code=400, detail="expected_previous must be a string")
        if lock.locked():
            raise HTTPException(status_code=409, detail="A runtime rollback is already running.")
        async with lock:
            state = request.app.state
            reason = _busy_reason(state)
            if reason:
                raise HTTPException(status_code=409, detail=reason)
            info = runtime_update.runtime_history(state.cfg)
            previous = info["previous"]
            if previous is None:
                raise HTTPException(status_code=409, detail="No earlier runtime is available on disk.")
            if expected is not None and expected != previous["path"]:
                raise HTTPException(status_code=409, detail="Runtime history changed. Refresh it and confirm again.")
            rt = state.router
            names = rt.running_models()
            restarted, errors = [], []
            state.runtime_rollback_running = True
            try:
                async with state.resources.acquire("runtime-rollback", exclusive=True):
                    reason = _busy_reason(state, own_lease=True)
                    if reason:
                        raise HTTPException(status_code=409, detail=reason)
                    # The synchronous local switch keeps config edits atomic on
                    # the event loop. Only publish live fields after it succeeds.
                    candidate = copy.deepcopy(state.cfg)
                    try:
                        restored = runtime_update.rollback_runtime(candidate, config_path)
                    except (OSError, ValueError) as exc:
                        raise HTTPException(status_code=500, detail=f"Could not restore the runtime: {exc}") from exc
                    if restored is None:
                        raise HTTPException(status_code=409, detail="The earlier runtime is no longer available.")
                    state.cfg.paths.llama_server = candidate.paths.llama_server
                    for gpu, restored_gpu in zip(state.cfg.gpus, candidate.gpus, strict=True):
                        gpu.backend = restored_gpu.backend
                    rt._servers.clear()
                    rt._build_servers()
                    for name in names:
                        try:
                            # This operation owns the exclusive lease already;
                            # the public wrapper would wait for that same lease.
                            await rt._ensure_active(name)
                            restarted.append(name)
                        except Exception as exc:  # noqa: BLE001 - switch succeeded; report reload failures separately
                            errors.append({"model": name, "message": str(exc)})
                    return {"from": restored[0], "to": restored[1], "restarted": restarted,
                            "restart_errors": errors, "history": runtime_update.runtime_history(state.cfg)}
            finally:
                state.runtime_rollback_running = False
