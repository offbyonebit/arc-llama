"""Plugin extension point for arc-llama.

arc-llama's core is a simple inference machine: it discovers GGUFs, manages
``llama-server`` subprocesses, and exposes an OpenAI-compatible API. Feature
projects (audio, vision, custom admin surfaces, ...) should live *outside* the
core as installable add-ons rather than being folded into it.

Plugins are discovered through Python packaging entry points under the
``arc_llama.plugins`` group. A plugin package declares, in its own
``pyproject.toml``:

```toml
[project.entry-points."arc_llama.plugins"]
audio = "arc_llama_audio.plugin:create_plugin"
```

The entry point value is a ``module:attr`` reference. ``attr`` may be:

* a ``Plugin`` subclass (instantiated with no arguments),
* a zero-argument callable that returns a ``Plugin``, or
* a ``Plugin`` instance.

Loading is lazy: the plugin module is only imported when ``load_plugins`` runs
(at app creation), so a plugin's optional dependencies (e.g. ``torch``,
``sounddevice``) are never imported at core import time, and a plugin that
fails to import cannot stop the core from starting.

The contract is deliberately tiny. A plugin is any object exposing:

* ``name`` (str) — a stable, unique identifier;
* ``register(app)`` — called once at app creation to add FastAPI routes;
* ``startup(app)`` — optional, called inside the app lifespan on start;
* ``shutdown(app)`` — optional, called inside the app lifespan on stop.

``startup``/``shutdown`` may be sync or async. Every hook is isolated: an
exception in one plugin is logged and does not affect the core or other
plugins.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from fastapi import FastAPI

log = logging.getLogger("arc_llama.plugins")

ENTRY_POINT_GROUP = "arc_llama.plugins"


class Plugin:
    """Base class for arc-llama plugins.

    Subclass and override the hooks you need. ``register`` runs once at app
    creation (add FastAPI routes here); ``startup``/``shutdown`` run inside the
    app lifespan. Any hook may be a coroutine function.
    """

    name: str = "unnamed"

    def register(self, app: FastAPI) -> None:
        """Add routes/middleware to the app. Called once, before startup."""

    def startup(self, app: FastAPI) -> None:
        """Run when the app starts. May be async."""

    def shutdown(self, app: FastAPI) -> None:
        """Run when the app shuts down. May be async."""


def discover_entry_points() -> list[Any]:
    """Return installed entry points for the plugin group.

    Uses ``importlib.metadata``, which is stdlib and cheap. Returns an empty
    list when the group has no registrations (the common case), so the core
    behaves exactly as before when no plugins are installed.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return []
    eps = entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=ENTRY_POINT_GROUP))
    return [ep for ep in eps if ep.group == ENTRY_POINT_GROUP]


def _instantiate(obj: Any) -> Any:
    """Turn an entry-point value into a plugin instance.

    Accepts a class (instantiated with no args), a factory callable (called
    with no args), or an already-constructed instance.
    """
    if isinstance(obj, type):
        return obj()
    if callable(obj):
        return obj()
    return obj


def load_plugins(entry_points: Any = None, *, enabled: set[str] | None = None) -> list[Any]:
    """Load and instantiate plugins from entry points.

    ``entry_points`` defaults to the installed ``arc_llama.plugins`` group.
    Pass an explicit iterable of entry-point-like objects (anything with a
    ``.name`` and a ``.load()``) to test without installing a package.

    ``enabled`` is an optional set of plugin names to allow; when provided,
    only those names are loaded. When omitted, the ``ARC_LLAMA_PLUGINS`` env
    var (comma-separated names) is honoured if set, otherwise every discovered
    plugin is loaded.

    A plugin that fails to import or instantiate is skipped with a warning, so
    a broken add-on can never take the core down.
    """
    if entry_points is None:
        entry_points = discover_entry_points()
    if enabled is None:
        env = os.environ.get("ARC_LLAMA_PLUGINS")
        enabled = {n.strip() for n in env.split(",") if n.strip()} if env else None

    plugins: list[Any] = []
    for ep in entry_points:
        name = getattr(ep, "name", None) or str(ep)
        if enabled is not None and name not in enabled:
            log.debug("plugin %s not enabled; skipping", name)
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop core
            log.warning("plugin %s failed to import: %s", name, exc)
            continue
        try:
            plugin = _instantiate(obj)
        except Exception as exc:  # noqa: BLE001
            log.warning("plugin %s failed to instantiate: %s", name, exc)
            continue
        if not hasattr(plugin, "register"):
            log.warning("plugin %s has no register() method; skipping", name)
            continue
        plugins.append(plugin)
    return plugins


def register_plugins(app: FastAPI, plugins: list[Any]) -> None:
    """Call ``register`` on every plugin, isolating failures."""
    for plugin in plugins:
        try:
            plugin.register(app)
        except Exception:  # noqa: BLE001
            log.exception("plugin %s register() failed", getattr(plugin, "name", "?"))


async def _run_hook(plugin: Any, hook_name: str, app: FastAPI) -> None:
    hook = getattr(plugin, hook_name, None)
    if hook is None:
        return
    try:
        result = hook(app)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001
        log.exception("plugin %s %s() failed", getattr(plugin, "name", "?"), hook_name)


async def startup_plugins(plugins: list[Any], app: FastAPI) -> None:
    """Run every plugin's ``startup`` hook (sync or async), isolating failures."""
    for plugin in plugins:
        await _run_hook(plugin, "startup", app)


async def shutdown_plugins(plugins: list[Any], app: FastAPI) -> None:
    """Run every plugin's ``shutdown`` hook (sync or async), isolating failures."""
    for plugin in plugins:
        await _run_hook(plugin, "shutdown", app)
