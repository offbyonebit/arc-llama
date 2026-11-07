"""Authenticated first-answer setup; reads never install or load anything."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from arc_llama.onboarding import FirstAnswerSetup


def register_onboarding_routes(
    app: FastAPI,
    *,
    require_admin: Callable[..., Any],
    read_json_body: Callable[[Request], Awaitable[dict[str, Any]]],
    config_path: Path,
) -> None:
    def setup(request: Request) -> FirstAnswerSetup:
        state = request.app.state
        manager = getattr(state, "first_answer", None)
        if manager is None:
            manager = FirstAnswerSetup(state.cfg, config_path, state.router)
            state.first_answer = manager
        return manager

    @app.get("/admin/onboarding")
    async def status(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        manager = setup(request)
        manager.refresh()
        return manager.public()

    @app.post("/admin/onboarding")
    async def update(
        request: Request, _auth: None = Depends(require_admin)
    ) -> dict[str, Any]:
        body = await read_json_body(request)
        manager = setup(request)
        action = body.get("action", "start")
        try:
            if action == "pause":
                manager.pause()
            elif action == "refresh":
                manager.refresh(force=True)
            elif action == "detect":
                # Recovery after a driver/permission fix, without replacing an
                # existing user's GPU choices, models, settings or unknown keys.
                if not manager.cfg.gpus:
                    from arc_llama.config import init_config_from_detection
                    from arc_llama.detect import detect_gpus

                    detected = await asyncio.to_thread(detect_gpus)
                    if not detected:
                        raise ValueError(
                            "No Intel GPU was detected. Run arc-llama doctor for driver and device-permission guidance."
                        )
                    previous = manager.cfg.gpus
                    manager.cfg.gpus = init_config_from_detection(detected).gpus
                    try:
                        manager.cfg.save(config_path)
                    except OSError:
                        manager.cfg.gpus = previous
                        raise
                manager.refresh(force=True)
            elif action == "start":
                starter = body.get("starter")
                if not isinstance(starter, str):
                    raise ValueError("Choose a starter model before beginning setup.")
                manager.start(starter)
            else:
                raise ValueError("Unknown setup action.")
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manager.public()
