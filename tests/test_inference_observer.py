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
            assert calls[-1] == ("/v1/chat/completions", {**body, "timings_per_token": True})
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


@pytest.mark.parametrize("streaming,status,rate,expected", [
    (True, 200, 42.5, 42.5), (False, 200, 25.0, 25.0),
    (True, 200, None, None), (False, 200, None, None),
    (True, 500, 42.5, None), (False, 500, 25.0, None),
    (True, 200, -1, None), (False, 200, True, None),
])
async def test_generation_speed_observation_preserves_synthetic_response(monkeypatch, streaming, status, rate, expected):
    payload = {"choices": [{"delta": {"content": "Synthetic response"}}],
               "usage": {"completion_tokens": 50}, "timings": {"predicted_per_second": rate}}
    raw = json.dumps(payload).encode()
    if streaming:
        # Multiple timing samples in a split network stream must not duplicate the counter event.
        raw = b"data: " + raw + b"\n\ndata: " + raw + b"\n\ndata: [DONE]\n\n"

    class SyntheticStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(raw), 7):
                yield raw[offset:offset + 7]

    async def handler(request):
        if request.url.path.endswith("/budget"):
            return httpx.Response(503)
        return httpx.Response(status, headers={"content-type": "text/event-stream" if streaming else "application/json"}, stream=SyntheticStream())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(module, "httpx", SimpleNamespace(
        AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        HTTPError=httpx.HTTPError, Timeout=httpx.Timeout,
    ))
    queue = asyncio.Queue()
    observer = InferenceObserver("http://synthetic-model", "Synthetic task", queue)
    try:
        async with real_client(trust_env=False) as client:
            response = await client.post(observer.url + "/v1/chat/completions", json={"model": "m", "stream": streaming})
        assert response.content == raw and response.status_code == status
        events = []
        while not queue.empty():
            events.append(queue.get_nowait()["event"])
        measured = [event for event in events if event["type"] == "performance"]
        assert measured == ([] if expected is None else [{"type": "performance", "generation_tok_s": expected}])
        assert "Synthetic response" not in json.dumps(events)
    finally:
        await observer.close()


async def test_oversized_json_does_not_block_forwarding_or_invent_speed(monkeypatch):
    monkeypatch.setattr(module, "MAX_METRICS_JSON_BYTES", 16)
    raw = json.dumps({"choices": [{"message": {"content": "Synthetic response"}}],
                      "timings": {"predicted_per_second": 30}}).encode()

    class SyntheticStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield raw

    async def handler(request):
        if request.url.path.endswith("/budget"):
            return httpx.Response(503)
        return httpx.Response(200, stream=SyntheticStream(), headers={"content-type": "application/json"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(module, "httpx", SimpleNamespace(
        AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        HTTPError=httpx.HTTPError, Timeout=httpx.Timeout,
    ))
    queue = asyncio.Queue()
    observer = InferenceObserver("http://synthetic-model", "Synthetic task", queue)
    try:
        async with real_client(trust_env=False) as client:
            response = await client.post(observer.url + "/v1/chat/completions", json={"model": "m"})
        assert response.content == raw
        while not queue.empty():
            assert queue.get_nowait()["event"]["type"] != "performance"
    finally:
        await observer.close()


async def test_streamed_generation_speed_arrives_before_request_finishes(monkeypatch):
    release = asyncio.Event()
    requests = []

    class LiveStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"Synthetic thought"}}],"timings":{"predicted_per_second":20}}\n\n'
            await release.wait()
            yield b'data: {"timings":{"predicted_per_second":40}}\n\ndata: [DONE]\n\n'

    async def handler(request):
        if request.url.path.endswith("/budget"):
            return httpx.Response(503)
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=LiveStream())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(module, "httpx", SimpleNamespace(
        AsyncClient=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        HTTPError=httpx.HTTPError, Timeout=httpx.Timeout,
    ))
    queue = asyncio.Queue()
    observer = InferenceObserver("http://synthetic-model", "Synthetic task", queue)
    try:
        async with real_client(trust_env=False) as client:
            pending = asyncio.create_task(client.post(observer.url + "/v1/chat/completions", json={"model": "m", "stream": True}))
            while True:
                event = (await asyncio.wait_for(queue.get(), 5))["event"]
                if event["type"] == "performance":
                    assert event["generation_tok_s"] == 20
                    break
            assert not pending.done()
            assert requests[0]["timings_per_token"] is True
            release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response.status_code == 200
            measured = []
            while not queue.empty():
                event = queue.get_nowait()["event"]
                if event["type"] == "performance":
                    measured.append(event["generation_tok_s"])
            assert measured[-1] == 40  # the final sample is retained despite throttling
    finally:
        release.set()
        await observer.close()
