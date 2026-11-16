"""Idle llama-server health checks and bounded, router-owned recovery."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from arc_llama.launcher import LlamaServer
    from arc_llama.router import Router

log = logging.getLogger("arc_llama.watchdog")


@dataclass
class _Health:
    server: LlamaServer
    failures: int = 0
    pending: bool = False
    attempts: deque[float] = field(default_factory=deque)
    next_attempt: float = 0
    paused: bool = False
    error: str | None = None


class ServerWatchdog:
    def __init__(self, router: Router, busy: Callable[[], bool] = lambda: False):
        self.router = router
        self.busy = busy
        self.now = time.monotonic
        self.states: dict[str, _Health] = {}
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self.router.cfg.server.watchdog and self.task is None:
            self.task = asyncio.create_task(self._run(), name="server-watchdog")

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def forget(self, name: str) -> None:
        # Manual stop/rebuild cancels failed recovery; never revive that server.
        self.states.pop(name, None)

    def loaded(self, name: str) -> None:
        state = self.states.get(name)
        if state is not None:
            state.pending = False
            state.failures = 0
            state.error = None

    def error(self, name: str) -> str | None:
        state = self.states.get(name)
        return state.error if state is not None else None

    def _busy(self) -> bool:
        rt = self.router
        resources = rt.resources
        return bool(
            rt.inflight or any(rt.model_inflight.values()) or rt._loading_futures
            or self.busy()
            or (resources is not None and (
                resources.exclusive_active
                or resources._exclusive_waiters
                or any(owner != "watchdog" for owner in resources.active_leases)
            ))
        )

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.tick()
            except Exception:
                log.exception("watchdog health pass failed")

    async def _healthy(self, srv: LlamaServer) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
                response = await client.get(srv.plan.health_url)
                return response.status_code == 200 and response.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def tick(self) -> None:
        rt = self.router
        if not rt.cfg.server.watchdog or self._busy():
            return
        for name, srv in list(rt._servers.items()):
            state = self.states.get(name)
            if state is None or state.server is not srv:
                if not srv.ready:
                    continue
                state = self.states[name] = _Health(srv)
            if state.paused or (not srv.ready and not state.pending):
                continue
            if state.pending and any(n != name for n in rt.running_models()):
                continue
            # The probe can yield to a new request, manual stop or rebuild.
            healthy = srv.is_running and not state.pending and await self._healthy(srv)
            if (
                self._busy() or rt._servers.get(name) is not srv
                or self.states.get(name) is not state
                or name in rt._stopping
                or (not srv.ready and not state.pending)
            ):
                continue
            if healthy:
                state.failures = 0
                state.error = None
                continue
            state.failures += 1
            if srv.is_running and not state.pending and state.failures < 3:
                continue
            if self.now() < state.next_attempt:
                continue
            resources = rt.resources
            scope = resources.acquire("watchdog", exclusive=False) if resources else nullcontext()
            async with scope:
                async with rt._lock:
                    # No await between this last busy check and marking teardown.
                    if (
                        self._busy() or rt._servers.get(name) is not srv
                        or self.states.get(name) is not state
                        or (not srv.ready and not state.pending)
                        or (state.pending and any(n != name for n in rt.running_models()))
                    ):
                        continue
                    now = self.now()
                    while state.attempts and state.attempts[0] <= now - 3600:
                        state.attempts.popleft()
                    if len(state.attempts) >= 3:
                        state.paused = True
                        state.error = (
                            f"Watchdog stopped automatic recovery for {name}: three restarts "
                            "within an hour. Check the model log, then stop and load it manually."
                        )
                        rt.metrics["last_error"] = state.error
                        log.error("%s", state.error)
                        continue
                    state.attempts.append(now)
                    state.next_attempt = now + 60 * (2 ** (len(state.attempts) - 1))
                    state.pending = True
                    rt._stopping.add(name)
                    log.warning("watchdog: %s unhealthy; restarting llama-server", name)
                    try:
                        await srv.astop()
                        rt.metrics["stops"] += 1
                        srv.start(log_dir=rt.log_dir)
                        if not await srv.wait_ready():
                            raise RuntimeError("llama-server did not become healthy")
                        rt.metrics["loads"] += 1
                        rt.metrics["last_load_at"] = time.time()
                        rt.metrics["last_error"] = None
                        state.pending = False
                        state.failures = 0
                        state.error = None
                        rt._emit("model_loaded", model=name, reason="watchdog")
                    except Exception as exc:
                        rt.metrics["load_errors"] += 1
                        state.error = f"Watchdog recovery failed for {name}: {exc}"
                        rt.metrics["last_error"] = state.error
                        log.exception("%s", state.error)
                        try:
                            await srv.astop()
                        except Exception:
                            log.exception("watchdog: failed to stop %s after recovery failure", name)
                    finally:
                        rt._stopping.discard(name)
