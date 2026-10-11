"""Stable, versioned surface for arc-llama plugins.

Plugins should import from this module rather than from private core modules
(``router``, ``launcher``, ``server``). Everything here is covered by the
plugin API version below: additive changes bump the minor number, and any
change that could break an existing plugin bumps the major number.

A plugin declares the API it was written against with ``requires_api``::

    from arc_llama.plugin_api import Plugin, plugin_context

    class AudioPlugin(Plugin):
        name = "audio"
        requires_api = "1.0"

        def register(self, app):
            ctx = plugin_context(app)

            @app.post("/plugins/audio/transcribe", dependencies=[ctx.require_admin])
            async def transcribe():
                async with ctx.gpu_lease("audio"):
                    ...

The loader skips a plugin whose required major version differs from the
running core, or whose required minor version is newer, and reports it as
``incompatible`` in ``GET /admin/plugins`` instead of letting it fail at
runtime with an ``AttributeError``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from arc_llama.plugins import Plugin

log = logging.getLogger("arc_llama.plugin_api")

PLUGIN_API_VERSION = "1.0"
"""``MAJOR.MINOR`` version of this module's contract."""

MODEL_LOADED = "model_loaded"
"""Event fired after a model's llama-server passes its health check."""
MODEL_STOPPED = "model_stopped"
"""Event fired after a model's llama-server is stopped or evicted."""
MODEL_LOAD_FAILED = "model_load_failed"
"""Event fired when a model fails to start."""
EVENTS = frozenset({MODEL_LOADED, MODEL_STOPPED, MODEL_LOAD_FAILED})

EventCallback = Callable[[str, dict[str, Any]], Any]

__all__ = [
    "EVENTS",
    "MODEL_LOADED",
    "MODEL_LOAD_FAILED",
    "MODEL_STOPPED",
    "PLUGIN_API_VERSION",
    "EventBus",
    "Plugin",
    "PluginContext",
    "api_compatible",
    "plugin_context",
]


def _parse(version: str) -> tuple[int, int] | None:
    parts = str(version).strip().split(".")
    if not parts or len(parts) > 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) == 2 else 0
    except ValueError:
        return None
    if major < 0 or minor < 0:
        return None
    return major, minor


def api_compatible(required: str | None, provided: str = PLUGIN_API_VERSION) -> bool:
    """True when a plugin requiring ``required`` can run on ``provided``.

    ``None`` means the plugin predates versioning and is always accepted, so
    existing plugins keep loading. A malformed requirement is rejected.
    """
    if required is None:
        return True
    want = _parse(required)
    have = _parse(provided)
    if want is None or have is None:
        return False
    return want[0] == have[0] and want[1] <= have[1]


class EventBus:
    """Fan-out of core lifecycle events to plugin callbacks.

    Callbacks run synchronously in the emitting task and must be quick; a
    callback that needs to do I/O should schedule its own task. A callback
    that raises is logged and never affects the core or other subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}

    def subscribe(self, event: str, callback: EventCallback) -> Callable[[], None]:
        """Register ``callback`` for ``event`` and return an unsubscribe function."""
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; choose one of {sorted(EVENTS)}")
        self._subscribers.setdefault(event, []).append(callback)

        def unsubscribe() -> None:
            callbacks = self._subscribers.get(event, [])
            if callback in callbacks:
                callbacks.remove(callback)

        return unsubscribe

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        for callback in list(self._subscribers.get(event, [])):
            try:
                callback(event, dict(payload))
            except Exception:  # noqa: BLE001 - a plugin must not break the core
                log.exception("event subscriber for %s failed", event)


class PluginContext:
    """Facade over the running core that plugins may rely on.

    Obtain one with ``plugin_context(app)``. Attributes that depend on the
    app lifespan (the router and the GPU lease manager) are resolved on each
    call, so the context can be created in ``register`` and used later in
    request handlers.
    """

    api_version = PLUGIN_API_VERSION

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    # -- authentication -------------------------------------------------

    @property
    def require_admin(self) -> Any:
        """FastAPI dependency enforcing the core admin token.

        Use as ``dependencies=[ctx.require_admin]`` on privileged routes.
        """
        from arc_llama.server import _require_admin

        return Depends(_require_admin)

    # -- GPU arbitration ------------------------------------------------

    @asynccontextmanager
    async def gpu_lease(self, owner: str, *, exclusive: bool = True) -> AsyncIterator[Any]:
        """Hold the GPU while the ``async with`` body runs.

        ``exclusive=True`` drains in-flight text requests and evicts resident
        models before the body runs; text requests and model loads queue until
        the body exits. ``exclusive=False`` only waits out exclusive work.
        """
        resources = getattr(self._app.state, "resources", None)
        if resources is None:
            raise RuntimeError("gpu_lease is only available while the server is running")
        async with resources.acquire(owner, exclusive=exclusive) as lease:
            yield lease

    # -- read-only status -----------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        """Registered local models with their current load state."""
        router = getattr(self._app.state, "router", None)
        if router is None:
            return []
        out: list[dict[str, Any]] = []
        for m in router.all_models():
            srv = router._servers.get(m.name)
            out.append(
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "aliases": list(m.aliases),
                    "gpu_pci_slot": m.gpu_pci_slot,
                    "loaded": bool(srv and srv.is_running and srv.ready),
                    "vision": bool((m.recipe or {}).get("mmproj")),
                }
            )
        return out

    def gpus(self) -> list[dict[str, Any]]:
        cfg = getattr(self._app.state, "cfg", None)
        if cfg is None:
            return []
        return [
            {
                "pci_slot": g.pci_slot,
                "name": g.name,
                "arch": g.arch,
                "backend": g.backend,
                "vram_mb": g.vram_mb,
                "enabled": g.enabled,
            }
            for g in cfg.gpus
        ]

    @property
    def base_url(self) -> str:
        """Loopback URL of the core OpenAI-compatible API."""
        cfg = getattr(self._app.state, "cfg", None)
        port = cfg.server.port if cfg is not None else 11437
        return f"http://127.0.0.1:{port}/v1"

    # -- lifecycle events -----------------------------------------------

    def subscribe(self, event: str, callback: EventCallback) -> Callable[[], None]:
        """Receive ``model_loaded``, ``model_stopped`` and ``model_load_failed``."""
        return _event_bus(self._app).subscribe(event, callback)


def _event_bus(app: FastAPI) -> EventBus:
    bus = getattr(app.state, "events", None)
    if bus is None:
        bus = EventBus()
        app.state.events = bus
    return bus


def plugin_context(app: FastAPI) -> PluginContext:
    """Return the plugin facade for ``app``."""
    ctx = getattr(app.state, "plugin_context", None)
    if ctx is None:
        ctx = PluginContext(app)
        app.state.plugin_context = ctx
    return ctx
