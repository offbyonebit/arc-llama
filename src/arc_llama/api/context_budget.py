"""Token budgets from a resident runtime; counting never loads or evicts a model."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request


def _positive(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _estimate(messages: list[dict[str, Any]], tools: Any) -> int:
    # The fallback is explicitly approximate, including multimodal overhead.
    size = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            size += sum(
                1500 * 4
                if part.get("type") == "image_url"
                else len(str(part.get("text", "")).encode())
                for part in content
                if isinstance(part, dict)
            )
        else:
            size += len(str(content or "").encode())
        size += len(json.dumps(message.get("tool_calls", []), ensure_ascii=False).encode()) + 64
    if tools:
        size += len(json.dumps(tools, ensure_ascii=False).encode())
    return math.ceil(size / 4)


async def _native_budget(
    request: Request, model: Any, backend: Any, body: dict[str, Any]
) -> tuple[int, int]:
    rt = request.app.state.router
    resources = getattr(request.app.state, "resources", None)

    # Admission prevents counting against a backend being evicted for exclusive work.
    async def count() -> tuple[int, int]:
        if (
            not backend.is_running
            or not backend.ready
            or rt._servers.get(model.name) is not backend
        ):
            raise ValueError("not resident")
        rt.inflight += 1
        rt.acquire_model(model.name)
        try:
            async with httpx.AsyncClient(timeout=2.5, follow_redirects=False) as client:
                url = backend.plan.backend_url
                props = await client.get(url + "/props")
                props.raise_for_status()
                ctx = _positive(props.json().get("default_generation_settings", {}).get("n_ctx"))
                if ctx is None:
                    raise ValueError("runtime did not report per-slot context")
                formatted = await client.post(
                    url + "/apply-template",
                    json={
                        k: body[k]
                        for k in ("messages", "tools", "tool_choice", "chat_template_kwargs")
                        if k in body
                    },
                )
                formatted.raise_for_status()
                prompt = formatted.json().get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError("runtime did not return a template")
                tokenized = await client.post(
                    url + "/tokenize",
                    json={"content": prompt, "add_special": True, "parse_special": True},
                )
                tokenized.raise_for_status()
                tokens = tokenized.json().get("tokens")
                if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                    raise ValueError("runtime did not return token ids")
                return len(tokens), ctx
        finally:
            rt.inflight -= 1
            rt.release_model(model.name)

    if resources is None:
        return await count()
    async with resources.acquire("context-budget", exclusive=False):
        return await count()


def register_context_budget_route(app: FastAPI) -> None:
    slots = asyncio.Semaphore(2)

    @app.post("/v1/chat/budget")
    async def context_budget(request: Request) -> dict[str, Any]:
        # Bound input before decoding; image data can otherwise dominate the request.
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 2 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Budget requests must be at most 2 MiB")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise HTTPException(status_code=400, detail="model must be a string")
        messages = body.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(m, dict)
            or not isinstance(m.get("role"), str)
            or not isinstance(m.get("content", ""), (str, list, type(None)))
            for m in messages
        ):
            raise HTTPException(
                status_code=400, detail="messages must contain chat message objects"
            )
        reply = body.get("max_tokens", 1024)
        if _positive(reply) is None:
            raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")
        rt = request.app.state.router
        model = next(
            (m for m in rt.all_models() if body["model"] == m.name or body["model"] in m.aliases),
            None,
        )
        recipe = model.recipe or {} if model is not None else {}
        configured = _positive(recipe.get("ctx"))
        parallel = _positive(recipe.get("parallel")) or 1
        ctx = configured // parallel if configured else None
        tokens = _estimate(messages, body.get("tools"))
        exact = False
        reason = "Model is not resident, or is an upstream model. Count is estimated."
        backend = rt._servers.get(model.name) if model is not None else None
        # Image embeddings are projector-dependent and cannot be counted by /tokenize.
        has_media = any(
            isinstance(m.get("content"), list)
            and any(not isinstance(p, dict) or p.get("type") != "text" for p in m["content"])
            for m in messages
        )
        if has_media:
            reason = "Image/audio tokens are estimated; the runtime tokenizer counts text only."
        elif backend is not None and backend.is_running and backend.ready:

            async def assess() -> tuple[int, int]:
                async with slots:
                    return await _native_budget(request, model, backend, body)

            try:
                tokens, ctx = await asyncio.wait_for(assess(), 3.0)
                exact = True
                reason = "Counted with the resident model's tokenizer and chat template."
            except (
                asyncio.TimeoutError,
                httpx.HTTPError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ):
                reason = (
                    "Runtime counting is unavailable. Count and configured context are estimated."
                )
        total = tokens + reply
        return {
            "prompt_tokens": tokens,
            "reserved_tokens": reply,
            "total_tokens": total,
            "context_tokens": ctx,
            "exact": exact,
            "reason": reason,
            "over_budget": total > ctx if ctx is not None else None,
        }
