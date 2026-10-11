"""Cancellable, bounded external IO for architecture checks; no model weights."""

from __future__ import annotations

import asyncio
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import httpx

_MAX_HEADER = 262_144
_MAX_METADATA = 2_097_152
_MAX_DIAGNOSTIC = 131_072


async def _hub_read(url: str, limit: int, *, range_header: bool = False) -> bytes:
    from huggingface_hub.utils import build_hf_headers

    # Use the user's Hub access settings, and never forward them to another host.
    headers = build_hf_headers()
    if range_header:
        headers["Range"] = f"bytes=0-{limit - 1}"
    data = bytearray()
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=16_384):
                if not range_header and len(data) + len(chunk) > limit:
                    raise ValueError("Model metadata exceeded the read limit")
                data.extend(chunk[: limit - len(data)])
                if len(data) >= limit:
                    if not range_header:
                        # Read one extra chunk to distinguish a full, valid response
                        # from truncation at exactly the metadata limit.
                        continue
                    break
    return bytes(data)


async def remote_metadata(repo: str) -> dict[str, Any]:
    from huggingface_hub import constants

    data = await _hub_read(f"{constants.ENDPOINT}/api/models/{repo}", _MAX_METADATA)
    value = json.loads(data)
    if not isinstance(value, dict) or not isinstance(value.get("siblings", []), list):
        raise ValueError("Invalid model metadata")
    return value


async def remote_header(repo: str, file: str, revision: str) -> bytes:
    from huggingface_hub import hf_hub_url

    return await _hub_read(
        hf_hub_url(repo, file, revision=revision), _MAX_HEADER, range_header=True
    )


def _fixture(architecture: str) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    body += string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    return body + b"\0" * (-len(body) % 32)


async def _diagnostics(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    data = bytearray()
    while chunk := await stream.read(16_384):
        # Drain both streams concurrently so a noisy loader cannot deadlock or
        # grow application memory indefinitely. Keep only the bounded tail.
        data.extend(chunk)
        if len(data) > _MAX_DIAGNOSTIC:
            del data[:-_MAX_DIAGNOSTIC]
    return bytes(data)


async def probe_output(binary: str, architecture: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="arc-architecture-") as folder:
        fixture = Path(folder) / "architecture.gguf"
        fixture.write_bytes(_fixture(architecture))
        process = await asyncio.create_subprocess_exec(
            binary,
            "-m",
            str(fixture),
            "-ngl",
            "0",
            "--device",
            "none",
            "--no-warmup",
            "--port",
            "0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout = asyncio.create_task(_diagnostics(process.stdout))
        stderr = asyncio.create_task(_diagnostics(process.stderr))
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
            out, err = await asyncio.gather(stdout, stderr)
            return process.returncode or 0, (out + err).decode("utf-8", errors="replace")
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            for reader in (stdout, stderr):
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(stdout, stderr, return_exceptions=True)
