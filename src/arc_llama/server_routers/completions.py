"""OpenAI /v1/chat/completions, /v1/completions and /v1/embeddings proxy."""
from __future__ import annotations

import json
import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from arc_llama.server_routers.common import _strip_response_headers

log = logging.getLogger("arc_llama.server")
router = APIRouter()


async def _proxy_post(request: Request, target_path: str, streaming_ok: bool = True):
    rt = request.app.state.router
    mgr = request.app.state.upstream_mgr
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    model_query = body.get("model", "")

    # Check upstreams first — they are passive proxies, no llama-server to start.
    await mgr.models()
    upstream_model = mgr.find_model(model_query)
    if upstream_model is not None:
        try:
            upstream_resp = await mgr.proxy(
                upstream_model,
                target_path,
                body_bytes,
                {"Content-Type": "application/json"},
                streaming_ok=streaming_ok,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
        want_stream = streaming_ok and bool(body.get("stream"))
        if want_stream:
            async def close_upstream() -> None:
                await upstream_resp.aclose()
            return StreamingResponse(
                upstream_resp.aiter_raw(),
                status_code=upstream_resp.status_code,
                headers=_strip_response_headers(dict(upstream_resp.headers)),
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(close_upstream),
            )
        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=_strip_response_headers(dict(upstream_resp.headers)),
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    # Local model — router manages llama-server lifecycle.
    rt.inflight += 1
    streaming_response_started = False
    try:
        try:
            model, srv = await rt.ensure_active(model_query)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_query!r}") from None
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        tuner = getattr(request.app.state, "tuner", None)
        if tuner is not None:
            tuner.bump_use(model.name)
        target_url = f"{srv.plan.backend_url}{target_path}"
        want_stream = streaming_ok and bool(body.get("stream"))
        fwd_headers = {"Content-Type": "application/json"}

        async def _complete() -> None:
            rt.last_activity = time.time()

        if want_stream:
            client = httpx.AsyncClient(timeout=None)
            req = client.build_request(
                "POST", target_url, content=body_bytes, headers=fwd_headers,
            )
            upstream = await client.send(req, stream=True)

            async def close_upstream() -> None:
                try:
                    await upstream.aclose()
                    await client.aclose()
                finally:
                    await _complete()
                    rt.inflight -= 1

            streaming_response_started = True
            return StreamingResponse(
                upstream.aiter_raw(),
                status_code=upstream.status_code,
                headers=_strip_response_headers(dict(upstream.headers)),
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(close_upstream),
            )
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(target_url, content=body_bytes, headers=fwd_headers)
        await _complete()
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=_strip_response_headers(dict(r.headers)),
            media_type=r.headers.get("content-type", "application/json"),
        )
    finally:
        if not streaming_response_started:
            rt.inflight -= 1


@router.post("/v1/chat/completions")
@router.post("/v1/completions")
async def chat_or_completions(request: Request):
    return await _proxy_post(request, request.url.path)


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _proxy_post(request, "/v1/embeddings", streaming_ok=False)


__all__ = ["router", "_proxy_post"]
