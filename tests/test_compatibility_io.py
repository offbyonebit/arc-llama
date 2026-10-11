from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from arc_llama import compatibility_io as io
from arc_llama.model_compatibility import header_architecture


async def test_remote_read_is_bounded_when_range_is_ignored(monkeypatch):
    blocks, closed = [], []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(100):
                blocks.append(index)
                yield b"x" * 16_384

        async def aclose(self):
            closed.append(True)

    def response(request):
        assert "/" + ("a" * 40) + "/" in str(request.url)
        assert request.headers["Range"] == "bytes=0-262143"
        return httpx.Response(200, stream=Stream())

    client = httpx.AsyncClient
    monkeypatch.setattr(
        io.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(response), **kw)
    )
    assert len(await io.remote_header("u/r", "m.gguf", "a" * 40)) == 262_144
    assert len(blocks) == 16 and closed


async def test_metadata_over_limit_is_unknown_not_silently_truncated(monkeypatch):
    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 100))
    monkeypatch.setattr(io.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    monkeypatch.setattr(io, "_MAX_METADATA", 50)
    with pytest.raises(ValueError, match="read limit"):
        await io.remote_metadata("u/r")


async def test_hub_access_and_cross_host_redirect_do_not_leak_token(monkeypatch):
    from huggingface_hub import utils

    monkeypatch.setattr(
        utils, "build_hf_headers", lambda: {"Authorization": "Bearer private-token"}
    )
    seen = []

    def response(request):
        seen.append((request.url.host, request.headers.get("Authorization")))
        if len(seen) == 1:
            return httpx.Response(
                302, headers={"Location": "https://weights.example.invalid/m.gguf"}
            )
        return httpx.Response(200, content=b"GGUF")

    client = httpx.AsyncClient
    monkeypatch.setattr(
        io.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(response), **kw)
    )
    assert await io.remote_header("u/r", "m.gguf", "a" * 40) == b"GGUF"
    assert seen[0][1] == "Bearer private-token" and seen[1][1] is None


async def test_cancelled_header_closes_stream_and_client(monkeypatch):
    started, closed = asyncio.Event(), []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.append(True)

    client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    monkeypatch.setattr(io.httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    task = asyncio.create_task(io.remote_header("u/r", "m.gguf", "a" * 40))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


async def test_probe_cancel_kills_reaps_and_removes_fixture(monkeypatch):
    started, exited = asyncio.Event(), asyncio.Event()
    commands, killed = [], []
    process = SimpleNamespace(
        returncode=None, stdout=asyncio.StreamReader(), stderr=asyncio.StreamReader()
    )

    async def wait():
        started.set()
        await exited.wait()
        return process.returncode

    def kill():
        killed.append(True)
        process.returncode = -9
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        exited.set()

    process.wait, process.kill = wait, kill

    async def launch(*args, **kwargs):
        commands.append(args)
        assert header_architecture(Path(args[2]).read_bytes()) == "llama"
        assert args[args.index("--device") + 1] == "none"
        assert args[args.index("-ngl") + 1] == "0"
        return process

    monkeypatch.setattr(io.asyncio, "create_subprocess_exec", launch)
    task = asyncio.create_task(io.probe_output("runtime", "llama"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed == [True] and exited.is_set()
    assert not Path(commands[0][2]).exists()


async def test_noisy_probe_drains_both_streams_with_bounded_retention(monkeypatch):
    process = SimpleNamespace(
        returncode=1, stdout=asyncio.StreamReader(), stderr=asyncio.StreamReader()
    )
    process.stdout.feed_data(b"x" * 500_000)
    process.stderr.feed_data(b"y" * 500_000 + b"unknown model architecture: 'llama'")
    process.stdout.feed_eof()
    process.stderr.feed_eof()

    async def wait():
        return 1

    process.wait = wait

    async def launch(*args, **kwargs):
        return process

    monkeypatch.setattr(io.asyncio, "create_subprocess_exec", launch)
    returncode, output = await io.probe_output("runtime", "llama")
    assert returncode == 1 and len(output) <= 2 * io._MAX_DIAGNOSTIC
    assert output.endswith("unknown model architecture: 'llama'")
