"""GPU resource lease arbitration.

Arc Llama's core inference path already serialises its own GPU work through
the Router: requests are counted in ``Router.inflight`` while a forwarded
generation is live, evictions drain before stopping llama-server, and model
swaps take the swap lock. Plugins with their own heavyweight GPU work (the
vision companion generating an image, a future audio whisper pass) run
outside that machinery: they must not share the GPU with resident
llama-server processes, and they must not fight each other for it.

``ResourceLeaseManager`` sits between the two. It owns one exclusive gate,
waits for the router's in-flight requests to finish, and empties the GPU by
calling the Router's own lifecycle methods — it never manages llama-server
processes itself. Normal text inference is untouched: ``_proxy_post`` does
not take leases, and requests that arrive while an exclusive lease is held
still start models exactly as before.

Intended use from a plugin::

    resources = app.state.resources
    async with resources.acquire("vision", exclusive=True) as lease:
        ... GPU-heavy work inside the with-body ...

``acquire`` is a context manager, so the lease is released when the body
returns, raises, or the task is cancelled. Eviction only ever runs on
enter, so a failure inside the body cannot leave GPU state behind.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc_llama.router import Router

log = logging.getLogger("arc_llama.resources")

_DRAIN_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class Lease:
    """A granted lease handle yielded by ``ResourceLeaseManager.acquire``."""

    owner: str
    exclusive: bool
    acquired_at: float


class ResourceLeaseManager:
    """Arbitrates exclusive plugin GPU work against the router.

    Compact by design: one lock, one event, one owner-count table. All
    llama-server process management is delegated to the Router.
    """

    def __init__(self, router: Router | None = None, *, drain_seconds: float = 30.0):
        self.router = router
        self.drain_seconds = drain_seconds
        # Serialises exclusive leases against each other. Text inference
        # never takes this lock, so the router's fast-path behaviour is
        # unchanged.
        self._gate = asyncio.Lock()
        # True only while an exclusive lease is held. Shared leases wait
        # on the event below rather than the gate, so shares coexist freely
        # with each other but never with an active or pending exclusive.
        self.exclusive_active = False
        self._no_exclusive: asyncio.Event = asyncio.Event()
        self._no_exclusive.set()
        # Exclusive leases queued on the gate (pending or active). Kept so a
        # shared lease that arrives while an exclusive one is waiting queues
        # behind it too, and the event is only re-armed when the last one
        # settles.
        self._exclusive_waiters = 0
        self._leases: dict[str, int] = {}

    @property
    def active_leases(self) -> dict[str, int]:
        """Snapshot of currently held leases as ``owner -> hold count``."""
        return dict(self._leases)

    async def _wait_router_drained(self) -> bool:
        """Wait until the router has no in-flight forwarded requests.

        ``Router.inflight`` spans the whole request lifetime, streaming
        included, so once it reads zero no generation is using the GPU. The
        wait is bounded the same way the router's own eviction drains: a
        stuck client must not block a plugin task forever. Returns True
        when the router drained, False when the budget ran out.
        """
        if self.router is None:
            return True
        inflight = int(getattr(self.router, "inflight", 0) or 0)
        if inflight <= 0:
            return True
        deadline = time.monotonic() + self.drain_seconds
        while inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(_DRAIN_POLL_SECONDS)
            inflight = int(getattr(self.router, "inflight", 0) or 0)
        return inflight <= 0

    async def _stop_resident_models(self) -> int:
        """Empty the GPU of llama-server processes via the Router.

        Delegates to ``Router.stop_all`` rather than touching subprocess
        management here: the router owns every process, and its stop path
        is the one already exercised by ``POST /admin/stop-all``. A router
        duck-type without ``stop_all`` falls back to stopping each resident
        model through ``stop_one``.
        """
        if self.router is None:
            return 0
        stop_all = getattr(self.router, "stop_all", None)
        if stop_all is not None:
            return int(await stop_all() or 0)
        running = getattr(self.router, "running_models", None)
        stop_one = getattr(self.router, "stop_one", None)
        if running is None or stop_one is None:
            return 0
        stopped = 0
        for name in list(running()):
            if await stop_one(name):
                stopped += 1
        return stopped

    async def _prepare_gpu(self, owner: str) -> None:
        """Drain active requests, then evict resident models."""
        drained = await self._wait_router_drained()
        if not drained:
            log.warning(
                "%d router request(s) still in flight after %.0fs drain; "
                "granting the exclusive lease to %r anyway (their clients "
                "will see their generation stop)",
                int(getattr(self.router, "inflight", 0) or 0),
                self.drain_seconds,
                owner,
            )
        stopped = await self._stop_resident_models()
        if stopped:
            log.info("exclusive lease %r: stopped %d resident model(s)", owner, stopped)

    def _register(self, owner: str) -> None:
        self._leases[owner] = self._leases.get(owner, 0) + 1

    def _unregister(self, owner: str) -> None:
        """Drop one hold on *owner*; safe to call when nothing is held."""
        remaining = self._leases.get(owner, 0) - 1
        if remaining > 0:
            self._leases[owner] = remaining
        else:
            self._leases.pop(owner, None)

    @asynccontextmanager
    async def acquire(self, owner: str, *, exclusive: bool = True) -> AsyncIterator[Lease]:
        """Hold a GPU lease for *owner* for the duration of the with-body.

        With ``exclusive=True`` (the default) the lease:

          1. serialises against every other exclusive lease;
          2. waits (bounded by ``drain_seconds``) for the router's
             in-flight requests to finish;
          3. stops every resident llama-server so the plugin owns the GPU.

        With ``exclusive=False`` the lease only waits out any active or
        pending exclusive lease and then registers — shared holders coexist
        with each other but never with an exclusive one. Shared leases are
        bookkeeping only: exclusive grants do not wait for them, so the
        shared mode suits lightweight work, not GPU contention.

        The lease is released on normal exit, exception, or cancellation.
        Release is pure bookkeeping; it never touches the router, so a
        failing body can never take the manager down with it. Do not nest
        ``acquire`` for the same owner: the gate is not reentrant.
        """
        lease = Lease(owner=owner, exclusive=exclusive, acquired_at=time.time())
        if exclusive:
            # Make pending exclusivity visible before queueing on the gate,
            # so shared leases arriving now wait for this one too.
            self._exclusive_waiters += 1
            self._no_exclusive.clear()
            try:
                async with self._gate:
                    self.exclusive_active = True
                    try:
                        await self._prepare_gpu(owner)
                        self._register(owner)
                        yield lease
                    finally:
                        self._unregister(owner)
                        self.exclusive_active = False
            finally:
                self._exclusive_waiters -= 1
                if not self._exclusive_waiters:
                    self._no_exclusive.set()
        else:
            await self._no_exclusive.wait()
            self._register(owner)
            try:
                yield lease
            finally:
                self._unregister(owner)
