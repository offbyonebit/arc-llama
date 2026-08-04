"""Chat history persistence endpoints."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from arc_llama.chat_store import ChatMessage, ChatStore

log = logging.getLogger("arc_llama.server")
router = APIRouter()


@router.get("/v1/chats")
async def list_chats(request: Request, folder: str | None = Query(None)) -> dict[str, Any]:
    """Return a list of chat summaries ordered by most recently updated first."""
    store: ChatStore = request.app.state.chat_store
    chats = store.list_chats(folder=folder)
    return {"object": "list", "data": [c.summary() for c in chats]}


@router.get("/v1/chats/folders")
async def list_chat_folders(request: Request) -> dict[str, Any]:
    """Return all folders with chat counts."""
    store: ChatStore = request.app.state.chat_store
    return {"object": "list", "data": store.list_folders()}


@router.post("/v1/chats")
async def create_chat(request: Request) -> dict[str, Any]:
    """Create a new chat."""
    import uuid

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    chat_id = body.get("id") or str(uuid.uuid4())
    title = body.get("title") or "Untitled chat"
    folder = body.get("folder") if body.get("folder") is not None else ""
    store: ChatStore = request.app.state.chat_store
    try:
        chat = store.create(chat_id, title, folder=folder)
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"Chat already exists: {chat_id}") from None
    return chat.to_dict()


@router.post("/v1/chats/search")
async def search_chats(request: Request) -> dict[str, Any]:
    """Search chat titles and messages."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    query = body.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    limit = int(body.get("limit", 20))
    store: ChatStore = request.app.state.chat_store
    results = store.search(query, limit=limit)
    return {
        "object": "list",
        "data": [
            {
                "chat": chat.summary(),
                "matching_message_indices": indices,
            }
            for chat, indices in results
        ],
    }


@router.get("/v1/chats/export")
async def export_chats(request: Request) -> dict[str, Any]:
    """Export every chat as a portable JSON document."""
    store: ChatStore = request.app.state.chat_store
    return {"version": 1, "exported_at": time.time(), "chats": store.export_all()}


@router.post("/v1/chats/import")
async def import_chats(request: Request) -> dict[str, Any]:
    """Import chats from an export document."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    chats = body.get("chats")
    if not isinstance(chats, list):
        raise HTTPException(status_code=400, detail="'chats' must be an array")
    store: ChatStore = request.app.state.chat_store
    result = store.import_chats(chats, overwrite=bool(body.get("overwrite", False)))
    return {
        "imported": result["imported"],
        "skipped": result["skipped"],
        "errors": result["errors"],
        "error_details": result["error_details"],
    }


@router.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str, request: Request) -> dict[str, Any]:
    """Return a full chat including all messages."""
    store: ChatStore = request.app.state.chat_store
    chat = store.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat.to_dict()


@router.put("/v1/chats/{chat_id}")
async def update_chat(chat_id: str, request: Request) -> dict[str, Any]:
    """Replace an entire chat (title and/or messages)."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    store: ChatStore = request.app.state.chat_store
    chat = store.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    if "title" in body:
        chat.title = str(body["title"])
    if "messages" in body and isinstance(body["messages"], list):
        chat.messages = [ChatMessage.from_dict(m) for m in body["messages"]]
    store.save(chat)
    return chat.to_dict()


@router.patch("/v1/chats/{chat_id}")
async def patch_chat(chat_id: str, request: Request) -> dict[str, Any]:
    """Append messages or update a chat's title/folder without replacing everything."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    store: ChatStore = request.app.state.chat_store
    chat = store.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    if "title" in body:
        chat.title = str(body["title"])
    if "folder" in body and body["folder"] is not None:
        chat.folder = str(body["folder"])
    if "messages" in body and isinstance(body["messages"], list):
        for m in body["messages"]:
            chat.messages.append(ChatMessage.from_dict(m))
    store.save(chat)
    return chat.to_dict()


@router.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str, request: Request) -> dict[str, Any]:
    """Delete a chat permanently."""
    store: ChatStore = request.app.state.chat_store
    if not store.delete(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True}
