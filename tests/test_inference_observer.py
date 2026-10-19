from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from arc_llama.agent.harness import inference as module
from arc_llama.agent.harness.inference import InferenceObserver


class Stream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_observer_counts_the_actual_prompt_and_forwards_bytes(monkeypatch):
    calls = []

    async def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/budget"):
            return httpx.Response(
                200,
                json={
                    "prompt_tokens": 5000 if body.get("tools") else 500,
                    "context_tokens": 8192,
                    "exact": True,
                },
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
            HTTPError=httpx.HTTPError,
            Timeout=httpx.Timeout,
        ),
    )
    queue = asyncio.Queue()
    logger = logging.getLogger("uvicorn.error")
    old_level, old_handlers = logger.level, list(logger.handlers)
    observer = InferenceObserver("http://model", "Task", queue)
    assert logger.level == old_level and logger.handlers == old_handlers
    try:
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "Actual project instructions"},
                {"role": "user", "content": "Task"},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "stream": True,
        }
        async with real_client(trust_env=False) as client:
            response = await client.post(observer.url + "/v1/chat/completions", json=body)
            assert response.status_code == 200
            assert response.content.endswith(b"data: [DONE]\n\n")
            assert calls[-1] == ("/v1/chat/completions", body)
            assert (
                await client.post(
                    "http://127.0.0.1:"
                    + str(observer.socket.getsockname()[1])
                    + "/wrong/v1/chat/completions",
                    json=body,
                )
            ).status_code == 404
        events = []
        while not queue.empty():
            events.append(queue.get_nowait()["event"])
        context = next(e for e in events if e["type"] == "context")
        assert (
            context["prompt_tokens"] == 5000
            and context["setup_tokens"] == 4500
            and context["exact"]
        )
        assert "Actual project instructions" not in json.dumps(events)
    finally:
        await observer.close()
    assert observer.server_task.done()


@pytest.mark.asyncio
async def test_count_failure_does_not_prevent_inference(monkeypatch):
    async def handler(request):
        if request.url.path.endswith("/budget"):
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
            HTTPError=httpx.HTTPError,
            Timeout=httpx.Timeout,
        ),
    )
    observer = InferenceObserver("http://model", "Task", asyncio.Queue())
    try:
        async with real_client(trust_env=False) as client:
            response = await client.post(
                observer.url + "/v1/chat/completions", json={"model": "m", "messages": []}
            )
            assert response.status_code == 200 and b"[DONE]" in response.content
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_close_cancels_a_model_request_waiting_for_first_token(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if request.url.path.endswith("/budget"):
            return httpx.Response(503)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
            HTTPError=httpx.HTTPError,
            Timeout=httpx.Timeout,
        ),
    )
    observer = InferenceObserver("http://model", "Task", asyncio.Queue())
    async with real_client(trust_env=False) as client:
        pending = asyncio.create_task(
            client.post(observer.url + "/v1/chat/completions", json={"model": "m", "messages": []})
        )
        await asyncio.wait_for(started.wait(), 2)
        await observer.close()
        await asyncio.wait_for(cancelled.wait(), 2)
        try:
            await asyncio.wait_for(pending, 2)
        except httpx.HTTPError:
            pass
    assert observer.server_task.done()
