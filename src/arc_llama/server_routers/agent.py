"""Local coding agent endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from arc_llama.agent import run_agent
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.server_routers.common import _require_admin, require_admin

log = logging.getLogger("arc_llama.server")
router = APIRouter()


@router.post("/v1/agent")
async def agent_endpoint(request: Request):
    """Run the local coding agent and stream tool-execution events."""
    cfg = request.app.state.cfg
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    model = body.get("model")
    task = body.get("task")
    if not model or not task:
        raise HTTPException(status_code=400, detail="'model' and 'task' are required")

    auto_confirm = bool(body.get("auto_confirm", False))
    if auto_confirm:
        await _require_admin(request)
    plan_mode = bool(body.get("plan_mode", False))
    max_turns = int(body.get("max_turns", 30))
    folder = body.get("folder") if body.get("folder") is not None else ""
    root_path = body.get("root") or cfg.agent.root
    root = Path(root_path).expanduser().resolve()
    requested_profile = body.get("profile")
    active_profile = cfg.agent.profile
    if requested_profile is not None and requested_profile != active_profile:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Server is running profile {active_profile!r}; "
                f"requested profile {requested_profile!r} is not active. "
                "Start a server with the requested profile or omit the field."
            ),
        )

    run_id = str(uuid.uuid4())
    pending = request.app.state.pending_confirmations
    confirm_event = asyncio.Event()
    confirm_result: dict[str, bool] = {"approved": False}
    pending[run_id] = (confirm_event, confirm_result)

    pending_plans = request.app.state.pending_plan_approvals
    plan_event = asyncio.Event()
    plan_result: dict[str, bool] = {"approved": False}
    if plan_mode:
        pending_plans[run_id] = (plan_event, plan_result)

    base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    chat_store: ChatStore = request.app.state.chat_store
    checkpoint_store = request.app.state.checkpoint_store
    semantic_index = request.app.state.semantic_index

    agent_chat_id: str | None = None
    try:
        title = task.strip().split("\n")[0][:80] or "Agent task"
        agent_chat = chat_store.create(str(uuid.uuid4()), title, folder=folder)
        agent_chat_id = agent_chat.id
    except Exception as e:
        log.warning("Could not create agent chat: %s", e)

    async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
        await confirm_event.wait()
        return confirm_result["approved"]

    async def plan_callback(plan_text: str) -> bool:
        await plan_event.wait()
        return plan_result["approved"]

    async def event_stream() -> AsyncIterator[str]:
        transcript: list[ChatMessage] = [ChatMessage(role="user", content=task)]
        try:
            async for event in run_agent(
                task=task,
                model=model,
                base_url=base_url,
                root=root,
                auto_confirm=auto_confirm,
                confirm_callback=confirm_callback,
                plan_mode=plan_mode,
                plan_callback=plan_callback,
                run_id=run_id,
                checkpoint_store=checkpoint_store,
                max_turns=max_turns,
                chat_store=chat_store,
                extra={"semantic_index": semantic_index},
            ):
                if event.get("type") == "confirm_required":
                    event = {**event, "run_id": run_id}
                if event.get("type") == "plan":
                    event = {**event, "run_id": run_id}
                if event.get("type") == "checkpoint":
                    event = {**event, "run_id": run_id}
                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") == "assistant" and event.get("content"):
                    transcript.append(ChatMessage(role="assistant", content=event["content"]))
                elif event.get("type") == "tool_result":
                    label = event.get("name", "tool")
                    content = event.get("content", "")
                    transcript.append(ChatMessage(role="tool", content=f"{label}:\n{content}"))
        finally:
            pending.pop(run_id, None)
            pending_plans.pop(run_id, None)
            if agent_chat_id is not None:
                try:
                    chat = chat_store.get(agent_chat_id)
                    if chat is not None:
                        chat.messages.extend(transcript)
                        chat_store.save(chat)
                except Exception as e:
                    log.warning("Could not save agent transcript: %s", e)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/v1/agent/{run_id}/confirm", dependencies=[require_admin])
async def confirm_agent_run(run_id: str, request: Request) -> dict[str, bool]:
    """Approve or deny a pending agent tool confirmation."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    entry = request.app.state.pending_confirmations.get(run_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Run not found or not awaiting confirmation")

    event, result = entry
    result["approved"] = bool(body.get("approved", False))
    event.set()
    return {"ok": True}


@router.post("/v1/agent/{run_id}/plan", dependencies=[require_admin])
async def approve_agent_plan(run_id: str, request: Request) -> dict[str, bool]:
    """Approve or deny the plan for a planning-mode agent run."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    entry = request.app.state.pending_plan_approvals.get(run_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Run not found or not awaiting plan approval")

    event, result = entry
    result["approved"] = bool(body.get("approved", False))
    event.set()
    return {"ok": True}
