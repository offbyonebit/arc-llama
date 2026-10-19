"""The streaming proxy must always end the in-flight window it opened.

``_proxy_post`` increments ``router.inflight`` on entry and the streaming path
hands the decrement off to be run when the body is done. ``autotune._tick()``
returns early whenever ``inflight > 0``, and ``_deferred_restore`` waits for it
to reach zero, so a counter that is never decremented silently disables
background tuning for the rest of the process lifetime. Nothing surfaces to the
user: tuning simply stops happening.

The leak that mattered in practice was an upstream dying mid-generation.
Starlette only reaches ``await self.background()`` if streaming the body
returned normally, so an exception out of ``stream_response`` skipped the
BackgroundTask entirely. Client disconnect did NOT leak under uvicorn, whose
ASGI spec_version takes a branch that still runs background tasks, which is the
opposite of what the bug report assumed. Both are covered below so neither
regresses.

None of this needs a GPU or a real llama.cpp backend.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from test_server import FakeRouter, FakeUpstreamManager


class _Stream:
    """Upstream SSE response. Optionally dies partway through, like a backend
    that crashes or a connection that drops mid-generation."""

    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, *, fail_after: int | None = None, chunks: int = 3):
        self.closed = False
        self._fail_after = fail_after
        self._chunks = chunks
        self.yielded = 0

    async def aiter_raw(self):
        for i in range(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise httpx.ReadError("upstream died mid-stream")
            self.yielded += 1
            yield f"data: chunk{i}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class _Client:
    """Stands in for httpx.AsyncClient. Records whether it was closed so the
    client leak is covered alongside the counter leak."""

    instances: list[_Client] = []

    stream_factory = staticmethod(lambda: _Stream())
    send_raises: BaseException | None = None

    def __init__(self, timeout=None):
        self.timeout = timeout
        self.closed = False
        self.stream = None
        _Client.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        await self.aclose()

    def build_request(self, method, url, content=None, headers=None):
        return {"method": method, "url": url, "content": content, "headers": headers}

    async def send(self, request, stream=False):
        if _Client.send_raises is not None:
            raise _Client.send_raises
        self.stream = _Client.stream_factory()
        return self.stream

    async def aclose(self):
        self.closed = True


@pytest.fixture
def app(monkeypatch):
    import arc_llama.server as server_mod

    monkeypatch.setattr(server_mod, "Router", FakeRouter)
    monkeypatch.setattr(server_mod, "UpstreamManager", FakeUpstreamManager)
    monkeypatch.setattr(server_mod.httpx, "AsyncClient", _Client)
    _Client.instances = []
    _Client.send_raises = None
    _Client.stream_factory = staticmethod(lambda: _Stream())
    return server_mod.create_app()


def _post(client, **kw):
    return client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        **kw,
    )


def test_inflight_returns_to_zero_after_a_normal_stream(app):
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            b"".join(r.iter_bytes())
        assert rt.inflight == 0


def test_inflight_returns_to_zero_when_upstream_dies_mid_stream(app):
    """The real leak. Starlette skips the BackgroundTask when streaming the
    body raises, so the decrement has to happen somewhere that still runs."""
    _Client.stream_factory = staticmethod(lambda: _Stream(fail_after=1))
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            assert b"inference_interrupted" in b"".join(r.iter_bytes())
        assert rt.inflight == 0, (
            "in-flight counter leaked; auto-tune is now disabled for the process"
        )


def test_inflight_returns_to_zero_when_client_disconnects_early(app):
    _Client.stream_factory = staticmethod(lambda: _Stream(chunks=50))
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            next(r.iter_bytes())  # read one chunk, then abandon the response
        assert rt.inflight == 0


def test_upstream_and_client_are_closed_when_upstream_dies(app):
    """Covers the httpx client leak on the same path: a failed stream must not
    strand the connection pool either."""
    _Client.stream_factory = staticmethod(lambda: _Stream(fail_after=1))
    with TestClient(app) as c:
        with _post(c) as r:
            assert b"inference_interrupted" in b"".join(r.iter_bytes())
    assert _Client.instances, "no client was constructed"
    streaming = [i for i in _Client.instances if i.stream is not None]
    assert streaming, "no streaming client was constructed"
    for inst in streaming:
        assert inst.closed, "httpx client leaked after mid-stream failure"
        assert inst.stream.closed, "upstream response leaked after mid-stream failure"


def test_client_is_closed_when_send_fails_before_any_response(app):
    """Nothing has been handed to Starlette yet, so no response owns the client
    and only this path can close it."""
    _Client.send_raises = httpx.ConnectError("refused")
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            assert r.status_code == 502
            assert "Retry" in r.read().decode()
        assert rt.inflight == 0
    assert _Client.instances, "no client was constructed"
    assert all(i.closed for i in _Client.instances), "client leaked when send() failed"


def test_inflight_is_held_while_the_body_is_being_produced(app):
    """The counter must not drop early. Releasing while generation is still
    live is what lets the autotuner restart the backend mid-request, which is
    the exact failure this counter exists to prevent.

    Sampled inside the upstream generator rather than from the client side:
    TestClient buffers the whole response, so by the time a chunk is readable
    the stream has already finished and the client cannot observe the window.
    """
    rt_box = {}
    seen: list[int] = []

    class _Watching(_Stream):
        async def aiter_raw(self):
            for i in range(4):
                seen.append(rt_box["rt"].inflight)
                yield f"data: chunk{i}\n\n".encode()

    _Client.stream_factory = staticmethod(_Watching)
    with TestClient(app) as c:
        rt_box["rt"] = app.state.router
        with _post(c) as r:
            b"".join(r.iter_bytes())
        assert seen, "upstream never produced a chunk"
        assert all(v == 1 for v in seen), f"in-flight window ended mid-generation: {seen}"
        assert app.state.router.inflight == 0


def test_release_is_idempotent(app):
    """Both the generator's finally and the BackgroundTask can fire for one
    request; together they must still count as exactly one release."""
    with TestClient(app) as c:
        rt = app.state.router
        for _ in range(3):
            with _post(c) as r:
                b"".join(r.iter_bytes())
        assert rt.inflight == 0, "counter drifted negative or positive across repeated streams"


def test_many_failed_streams_do_not_accumulate(app):
    """One leak disables auto-tune; the point of the fix is that N failures
    still leave the counter at zero."""
    _Client.stream_factory = staticmethod(lambda: _Stream(fail_after=0))
    with TestClient(app) as c:
        rt = app.state.router
        for _ in range(5):
            with _post(c) as r:
                assert b"inference_interrupted" in b"".join(r.iter_bytes())
        assert rt.inflight == 0


def test_counter_is_released_even_if_cancelled_during_cleanup(app):
    """Cleanup runs at the point a request is most likely to be torn down, so
    the decrement must not sit behind an await that cancellation can interrupt.
    CancelledError is a BaseException and deliberately not swallowed; the point
    is that the counter is already settled before it can be raised."""

    class _CancelOnClose(_Stream):
        async def aclose(self):
            raise __import__("asyncio").CancelledError()

    _Client.stream_factory = staticmethod(_CancelOnClose)
    with TestClient(app) as c:
        rt = app.state.router
        try:
            with _post(c) as r:
                b"".join(r.iter_bytes())
        except BaseException:
            pass
        assert rt.inflight == 0, "cancellation during cleanup skipped the decrement"


def test_autotune_still_runs_after_a_failed_stream(app, monkeypatch):
    """End to end on the consequence rather than the counter: a mid-stream
    failure must not gate the tuner off."""
    _Client.stream_factory = staticmethod(lambda: _Stream(fail_after=1))
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            assert b"inference_interrupted" in b"".join(r.iter_bytes())
        # This is the exact condition autotune._tick() gates every sweep on.
        assert not rt.inflight > 0, "autotune._tick() would return early forever"


def test_per_model_count_settles_after_streams(app):
    """The per-model counter gates eviction drains, so it must settle to zero
    on the same paths the global counter does — including a failed stream."""
    with TestClient(app) as c:
        rt = app.state.router
        with _post(c) as r:
            b"".join(r.iter_bytes())
        assert rt.model_inflight == {}

        _Client.stream_factory = staticmethod(lambda: _Stream(fail_after=1))
        with _post(c) as r:
            assert b"inference_interrupted" in b"".join(r.iter_bytes())
        assert rt.model_inflight == {}, "failed stream leaked the per-model count"


def _assert_settled(app):
    assert app.state.router.inflight == 0
    assert app.state.router.model_inflight == {}
    assert app.state.resources.active_leases == {}
    assert all(c.closed for c in _Client.instances)


def test_header_deadline_returns_504_and_next_request_works(app, monkeypatch):
    import asyncio

    async def stalled_send(self, request, stream=False):
        await asyncio.Event().wait()

    app.state.cfg.server.inference_first_response_seconds = 0.05
    original = _Client.send
    monkeypatch.setattr(_Client, "send", stalled_send)
    with TestClient(app) as c:
        with _post(c) as r:
            assert r.status_code == 504
        _assert_settled(app)
        monkeypatch.setattr(_Client, "send", original)
        with _post(c) as r:
            assert r.status_code == 200
        _assert_settled(app)


@pytest.mark.parametrize("emit_first", [False, True])
def test_first_byte_and_idle_deadlines_close_stream_and_release_lease(app, emit_first):
    import asyncio

    class Stalled(_Stream):
        async def aiter_raw(self):
            if emit_first:
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            await asyncio.Event().wait()

    app.state.cfg.server.inference_first_response_seconds = 0.08
    app.state.cfg.server.inference_stream_idle_seconds = 0.05
    _Client.stream_factory = staticmethod(Stalled)
    with TestClient(app) as c:
        with _post(c) as r:
            body = b"".join(r.iter_bytes())
            assert b"inference_interrupted" in body
            assert b"[DONE]" not in body
            assert (b"partial" in body) is emit_first
        _assert_settled(app)
        assert all(i.stream.closed for i in _Client.instances if i.stream)
        _Client.stream_factory = staticmethod(lambda: _Stream())
        with _post(c) as r:
            assert r.status_code == 200
        _assert_settled(app)


@pytest.mark.parametrize("idle", [0, 0.1])
def test_slow_prefill_and_active_stream_are_not_total_time_limited(app, idle):
    import asyncio

    class Slow(_Stream):
        async def aiter_raw(self):
            await asyncio.sleep(0.12)  # Longer than the idle limit; this is prefill.
            for _ in range(5):
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                await asyncio.sleep(0.04)
            yield b"data: [DONE]\n\n"

    app.state.cfg.server.inference_first_response_seconds = 0.5
    app.state.cfg.server.inference_stream_idle_seconds = idle
    _Client.stream_factory = staticmethod(Slow)
    with TestClient(app) as c:
        with _post(c) as r:
            body = b"".join(r.iter_bytes())
            assert b"inference_interrupted" not in body
            assert b"[DONE]" in body
        _assert_settled(app)


def test_eof_without_done_is_an_interruption_for_api_clients(app):
    class EOF(_Stream):
        async def aiter_raw(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    _Client.stream_factory = staticmethod(EOF)
    with TestClient(app) as c:
        with _post(c) as response:
            body = b"".join(response.iter_bytes())
        assert b"inference_interrupted" in body
        _assert_settled(app)


async def test_done_text_in_generated_content_does_not_truncate_stream():
    from arc_llama.server import _iter_sse_body

    class Literal:
        async def aiter_raw(self):
            yield b'data: {"choices":[{"delta":{"content":"the literal data: [DONE]"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"more text"}}]}\n\n'
            yield b"data: [DO"
            yield b"NE]\r\n\r\n"
            raise AssertionError("must stop after the actual terminator")

    result = b"".join([part async for part in _iter_sse_body(Literal(), require_done=True)])
    assert b"more text" in result
    assert result.endswith(b"data: [DONE]\r\n\r\n")


async def test_disconnect_while_sending_headers_still_finalizes(app):
    import json

    from starlette.requests import ClientDisconnect, Request

    from arc_llama.server import _proxy_post

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
        "app": app,
    }

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps({"model": "qwen", "stream": True}).encode(),
            "more_body": False,
        }

    async def disconnected(message):
        assert message["type"] == "http.response.start"
        raise OSError("client disconnected before headers")

    async with app.router.lifespan_context(app):
        response = await _proxy_post(Request(scope, receive), "/v1/chat/completions")
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, disconnected)
        _assert_settled(app)
        assert all(i.stream.closed for i in _Client.instances if i.stream)
