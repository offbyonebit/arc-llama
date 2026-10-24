"""Upstream OpenAI-compatible endpoint manager.

Fetches and caches model lists from configured upstreams so they appear
alongside local models in /v1/models and can be proxied transparently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from arc_llama.config import UpstreamConfig

log = logging.getLogger("arc_llama.upstream")

UPSTREAM_CACHE_TTL = 30.0  # seconds


@dataclass
class UpstreamModel:
    """A model discovered on an upstream endpoint."""

    id: str
    upstream_name: str
    upstream_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


class UpstreamManager:
    """Periodically fetches /v1/models from upstreams and caches results."""

    def __init__(self, upstreams: list[UpstreamConfig]):
        self._upstreams = upstreams
        self._cache: dict[str, list[UpstreamModel]] = {}
        self._last_fetch: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def models(self) -> list[UpstreamModel]:
        """Return the cached union of all upstream model lists.

        Stale entries (older than UPSTREAM_CACHE_TTL) are refreshed in the
        background on the first call that notices them.
        """
        now = time.monotonic()
        need_refresh: list[UpstreamConfig] = []
        for u in self._upstreams:
            last = self._last_fetch.get(u.name, 0)
            if now - last > UPSTREAM_CACHE_TTL:
                need_refresh.append(u)

        if need_refresh:
            # Fire off refreshes concurrently; don't block the caller on slow
            # upstreams. We intentionally do NOT hold the lock while fetching.
            tasks = [self._fetch_one(u) for u in need_refresh]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Return the cached union under the lock so we don't race with _fetch_one
        async with self._lock:
            result: list[UpstreamModel] = []
            for u in self._upstreams:
                result.extend(self._cache.get(u.name, []))
            return result

    async def _fetch_one(self, upstream: UpstreamConfig) -> None:
        url = f"{upstream.url}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("upstream %s (%s) fetch failed: %s", upstream.name, upstream.url, e)
            # Don't overwrite existing cache on transient failure
            return

        raw_models = data.get("data", []) if isinstance(data, dict) else []
        parsed: list[UpstreamModel] = []
        for m in raw_models:
            if not isinstance(m, dict):
                continue
            model_id = m.get("id", "")
            if not model_id:
                continue
            parsed.append(
                UpstreamModel(
                    id=model_id,
                    upstream_name=upstream.name,
                    upstream_url=upstream.url,
                    metadata=m.get("metadata") or {},
                )
            )

        async with self._lock:
            self._cache[upstream.name] = parsed
            self._last_fetch[upstream.name] = time.monotonic()
            log.debug("upstream %s refreshed: %d models", upstream.name, len(parsed))

    def find_model(self, model_id: str) -> UpstreamModel | None:
        """Look up a model id in the cached upstream lists."""
        for u in self._upstreams:
            for m in self._cache.get(u.name, []):
                if m.id == model_id:
                    return m
        return None

    async def proxy(
        self,
        upstream: UpstreamModel,
        path: str,
        body: bytes,
        headers: dict[str, str],
        streaming_ok: bool = True,
    ) -> tuple[httpx.AsyncClient, httpx.Response]:
        """Forward a request to the upstream and return the raw response.

        Returns the client alongside the response: the caller owns both and
        must close them (response first, then client) once the body has been
        consumed or abandoned. Returning the response alone leaked the
        client's connection pool on every proxied call — nothing after the
        return had a reference to close.
        """
        target_url = f"{upstream.upstream_url}{path}"
        client = httpx.AsyncClient(timeout=None)
        try:
            req = client.build_request("POST", target_url, content=body, headers=headers)
            if streaming_ok:
                return client, await client.send(req, stream=True)
            else:
                return client, await client.send(req, stream=False)
        except Exception:
            await client.aclose()
            raise

    def upstreams_status(self) -> list[dict[str, Any]]:
        """Return a snapshot for /admin/status."""
        result: list[dict[str, Any]] = []
        for u in self._upstreams:
            models = self._cache.get(u.name, [])
            result.append(
                {
                    "name": u.name,
                    "url": u.url,
                    "model_count": len(models),
                    "last_fetch": self._last_fetch.get(u.name),
                }
            )
        return result
