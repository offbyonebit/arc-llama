"""Count actual OpenCode inference requests on an ephemeral loopback relay.

Only counts and activity become UI events. Prompts and credentials are never
logged or emitted. The relay closes its upstream connections when a run stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
from collections.abc import Iterator
from typing import Any

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
from uvicorn.protocols.http.h11_impl import H11Protocol


class _HTTPProtocol(H11Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.access_log = False


class _Server(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        # The enclosing Arc Llama process owns signal handling.
        yield


class InferenceObserver:
    def __init__(
        self, base_url: str, task: str, queue: asyncio.Queue[dict[str, Any] | None]
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task = task
        self.queue = queue
        self._first = True
        self._closed = False
        secret = secrets.token_urlsafe(24)
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.post(f"/{secret}/v1/chat/completions")
        async def relay(request: Request) -> StreamingResponse:
            raw = await request.body()
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("object required")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid inference request") from exc
            client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0), trust_env=False)
            try:
                self.emit({"type": "activity", "phase": "generating"})
                await self.count(client, body)
                response = await client.send(
                    client.build_request(
                        "POST",
                        self.base_url + "/v1/chat/completions",
                        content=raw,
                        headers={
                            "Content-Type": "application/json",
                            **(
                                {"Authorization": request.headers["authorization"]}
                                if "authorization" in request.headers
                                else {}
                            ),
                        },
                    ),
                    stream=True,
                )
            except BaseException:
                with anyio.CancelScope(shield=True):
                    await client.aclose()
                raise

            async def stream():
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:
                    with anyio.CancelScope(shield=True):
                        await response.aclose()
                        await client.aclose()

            return StreamingResponse(
                stream(),
                status_code=response.status_code,
                headers={
                    k: response.headers[k]
                    for k in ("content-type", "content-encoding")
                    if k in response.headers
                },
            )

        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.socket.setblocking(False)
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}/{secret}"
        self.server = _Server(
            uvicorn.Config(
                app,
                lifespan="off",
                access_log=True,
                http=_HTTPProtocol,
                log_config=None,
                log_level=None,
                ws="none",
                timeout_graceful_shutdown=0,
            )
        )
        self.server_task = asyncio.create_task(self.server.serve(sockets=[self.socket]))

    def emit(self, event: dict[str, Any]) -> None:
        if not self._closed:
            self.queue.put_nowait({"type": "_inference", "event": event})

    async def count(self, client: httpx.AsyncClient, body: dict[str, Any]) -> None:
        # Advisory only: a bounded count failure must never prevent inference.
        async def assess(payload: dict[str, Any]) -> dict[str, Any] | None:
            try:
                response = await client.post(
                    self.base_url + "/v1/chat/budget",
                    json={**payload, "max_tokens": 1},
                    timeout=3.5,
                )
                response.raise_for_status()
                value = response.json()
                if (
                    isinstance(value, dict)
                    and type(value.get("prompt_tokens")) is int
                    and value["prompt_tokens"] >= 0
                ):
                    return value
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            return None

        measured = await assess(body)
        if measured is None:
            return
        event = {
            "type": "context",
            "prompt_tokens": measured["prompt_tokens"],
            "context_tokens": measured.get("context_tokens"),
            "exact": measured.get("exact") is True,
            "source": "agent_request",
        }
        initial, self._first = self._first, False
        if initial:
            bare = await assess(
                {"model": body.get("model"), "messages": [{"role": "user", "content": self.task}]}
            )
            if bare is not None:
                event["setup_tokens"] = max(0, measured["prompt_tokens"] - bare["prompt_tokens"])
        self.emit(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.server_task, 2.0)
        finally:
            self.socket.close()
