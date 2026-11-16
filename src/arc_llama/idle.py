"""Keep configured idle-unload timers suspended during long client operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from urllib.parse import quote

import httpx

log = logging.getLogger("arc_llama.idle")


@asynccontextmanager
async def hold_model(
    client: httpx.AsyncClient, name: str, *, enabled: bool
) -> AsyncIterator[None]:
    if not enabled:
        yield
        return
    path = f"/admin/idle-hold/{quote(name, safe='')}"
    response = await client.post(path, json={}, timeout=10)
    response.raise_for_status()
    identity = response.json()["id"]
    owner = asyncio.current_task()

    async def renew() -> None:
        try:
            while True:
                await asyncio.sleep(30)
                response = await client.post(path, json={"id": identity}, timeout=10)
                response.raise_for_status()
        except httpx.HTTPError:
            log.exception("idle hold lost for %s; stopping the operation", name)
            if owner is not None:
                owner.cancel()

    renewal = asyncio.create_task(renew())
    try:
        yield
    finally:
        renewal.cancel()
        with suppress(asyncio.CancelledError):
            await renewal
        try:
            await client.post(path, json={"id": identity, "release": True}, timeout=10)
        except httpx.HTTPError:
            log.warning("could not release idle hold for %s; it will expire", name)
