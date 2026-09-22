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

For UIs that present installed plugins, a plugin may additionally expose an
optional ``info()`` hook returning a JSON-serializable metadata dict (for
example ``{"version": ..., "description": ..., "ui": ...}``). The hook is
checked defensively: a missing one simply omits those fields, and a failing
one is logged and ignored so metadata can never break plugin loading.
"""

from __future__ import annotations

import inspect
import logging
import os
from copy import deepcopy
from typing import Any

from fastapi import FastAPI

log = logging.getLogger("arc_llama.plugins")

ENTRY_POINT_GROUP = "arc_llama.plugins"
_UI_PLACEMENTS = {"toolbar", "plugins", "chat"}
_COMPOSER_MODES = {"text", "attachments", "custom"}


def _validated_ui(data: Any) -> dict[str, Any]:
    """Keep plugin UI metadata declarative, small, and safe to render."""
    if not isinstance(data, dict):
        return {}
    result = deepcopy(data)
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in data.get("actions", []):
        if not isinstance(raw, dict):
            continue
        action_id = raw.get("id")
        label = raw.get("label")
        if not isinstance(action_id, str) or not action_id or action_id in seen:
            continue
        if not isinstance(label, str) or not label:
            continue
        placement = raw.get("placement", "plugins")
        if placement not in _UI_PLACEMENTS:
            placement = "plugins"
        action = {"id": action_id, "label": label, "placement": placement}
        for key in ("icon", "route", "method", "description"):
            if isinstance(raw.get(key), str) and raw[key]:
                action[key] = raw[key]
        composer = raw.get("composer")
        if isinstance(composer, dict):
            mode = composer.get("mode")
            if mode in _COMPOSER_MODES:
                composer_meta: dict[str, str] = {"mode": mode}
                for key in ("placeholder", "result"):
                    value = composer.get(key)
                    if isinstance(value, str) and value:
                        composer_meta[key] = value
                action["composer"] = composer_meta
        seen.add(action_id)
        actions.append(action)
    if "actions" in data:
        result["actions"] = actions
    return result


class Plugin:
    """Base class for arc-llama plugins.

    Subclass and override the hooks you need. ``register`` runs once at app
    creation (add FastAPI routes here); ``startup``/``shutdown`` run inside the
    app lifespan. Any hook may be a coroutine function.
    """

    name: str = "unnamed"

    def register(self, app: FastAPI) -> None:
        """Add routes/middleware to the app. Called once, before startup."""

    def info(self) -> dict[str, Any]:
        """Return UI-facing plugin metadata (optional, best effort).

        The catalog treats ``version``, ``description``, ``ui``, and ``api``
        as known optional keys; anything else JSON-serializable is passed
        through unchanged. This hook never influences loading: the default
        implementation returns ``{}`` and a broken override is ignored.
        """
        return {}

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
    return [ep for ep in eps if getattr(ep, "group", None) == ENTRY_POINT_GROUP]


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


def load_plugins(
    entry_points: Any = None,
    *,
    enabled: set[str] | None = None,
    discovery: PluginDiscovery | None = None,
) -> list[Any]:
    """Load and instantiate plugins from entry points.

    ``entry_points`` defaults to the installed ``arc_llama.plugins`` group.
    Pass an explicit iterable of entry-point-like objects (anything with a
    ``.name`` and a ``.load()``) to test without installing a package.

    ``enabled`` is an optional set of plugin names to allow; when provided,
    only those names are loaded. When omitted, the ``ARC_LLAMA_PLUGINS`` env
    var (comma-separated names) is honoured if set, otherwise every discovered
    plugin is loaded.

    Every discovered plugin — including ones that failed to import,
    instantiate, or satisfy the contract — is recorded in ``discovery``
    (a PluginDiscovery, or a fresh one when omitted) so the admin catalog
    can show what happened without running any plugin code. A plugin that
    fails to import or instantiate is skipped for execution purposes, so a
    broken add-on can never take the core down; loading never raises.
    """
    if entry_points is None:
        entry_points = discover_entry_points()
    if enabled is None:
        env = os.environ.get("ARC_LLAMA_PLUGINS")
        enabled = {n.strip() for n in env.split(",") if n.strip()} if env else None
    if discovery is None:
        discovery = PluginDiscovery()

    plugins: list[Any] = []
    for ep in entry_points:
        name = getattr(ep, "name", None) or str(ep)
        if enabled is not None and name not in enabled:
            log.debug("plugin %s not enabled; skipping", name)
            discovery.record(
                name, status="disabled", enabled=False, error="not in the enabled plugin list"
            )
            continue
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop core
            log.warning("plugin %s failed to import: %s", name, exc)
            discovery.record(name, status="failed", error=f"import failed: {exc}")
            continue
        try:
            plugin = _instantiate(obj)
        except Exception as exc:  # noqa: BLE001
            log.warning("plugin %s failed to instantiate: %s", name, exc)
            discovery.record(name, status="failed", error=f"instantiate failed: {exc}")
            continue
        if not hasattr(plugin, "register"):
            log.warning("plugin %s has no register() method; skipping", name)
            discovery.record(
                name, status="failed", error="no register() method; contract not satisfied"
            )
            continue
        plugins.append(plugin)
        discovery.record(name, status="loaded", plugin=plugin)
    return plugins


class PluginDiscovery:
    """Mutable record of one app creation's plugin discovery outcomes.

    Keeps failed and disabled plugins visible to the admin catalog without
    keeping any live object around: importing/instantiation failures are
    retained as name + status + short error, and disabled ones as name +
    ``enabled=False``. Loaded ones are retained as name + reference so
    ``build_catalog`` can merge live metadata.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def record(
        self,
        name: str,
        *,
        status: str,
        enabled: bool = True,
        error: str = "",
        plugin: Any = None,
    ) -> None:
        if name in self._records:
            return
        entry: dict[str, Any] = {
            "name": name,
            "status": status,
            "enabled": enabled,
        }
        if error:
            entry["error"] = str(error)[:1024]
        if plugin is not None:
            entry["plugin"] = plugin
        self._records[name] = entry

    def catalog(self) -> list[dict[str, Any]]:
        """JSON-serializable descriptors for every *discovered* plugin —
        failed and disabled included — with the same shape as
        build_catalog, so they can be merged into one admin catalog.
        No plugin code runs while serving this list."""
        out: list[dict[str, Any]] = []
        for name in sorted(self._records):
            rec = self._records[name]
            if rec["status"] == "loaded":
                # Live plugins go through the normal catalog path so their
                # register() outcome (active/error) stays authoritative.
                continue
            entry: dict[str, Any] = {
                "name": name,
                "status": rec["status"],
            }
            if rec.get("error"):
                entry["error"] = rec["error"]
            out.append(entry)
        return out

    def loaded_names(self) -> set[str]:
        return {
            rec["name"]
            for rec in self._records.values()
            if rec["status"] == "loaded"
        }


def register_plugins(app: FastAPI, plugins: list[Any]) -> None:
    """Call ``register`` on every plugin, isolating failures.

    When ``app.state.plugin_status`` exists (a dict created by the server),
    each plugin's outcome is recorded there so the UI can present a stable
    status key instead of guessing from logs.
    """
    for plugin in plugins:
        name = getattr(plugin, "name", "?")
        try:
            plugin.register(app)
            if hasattr(app.state, "plugin_status"):
                app.state.plugin_status[name] = "active"
        except Exception:  # noqa: BLE001
            log.exception("plugin %s register() failed", name)
            if hasattr(app.state, "plugin_status"):
                app.state.plugin_status[name] = "error"


def plugin_info(plugin: Any) -> dict[str, Any]:
    """Read one plugin's optional metadata, defensively.

        Plugins written against the original contract have no ``info`` hook and
        yield an empty dict; a hook that raises or returns a non-mapping is
        dropped with a warning rather than affecting anything else. The catalog
        treats ``version``, ``description``, ``ui``, and ``api`` as known
        optional keys; anything else JSON-serializable is passed through as is.
        """
    hook = getattr(plugin, "info", None)
    if hook is None:
        return {}
    try:
        data = hook()
    except Exception:  # noqa: BLE001 - metadata must never break the catalog
        log.warning(
            "plugin %s info() failed; ignoring metadata", getattr(plugin, "name", "?")
        )
        return {}
    if not isinstance(data, dict):
        log.warning("plugin %s info() returned a non-mapping; ignoring metadata", getattr(plugin, "name", "?"))
        return {}
    result = deepcopy(data)
    if "ui" in result:
        result["ui"] = _validated_ui(result["ui"])
    return result


def build_catalog(
    plugins: list[Any],
    status: dict[str, str] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-serializable descriptors for the installed plugins.

    ``status`` maps plugin names to a stable status key (e.g. ``active`` or
    ``error``) and defaults to ``registered`` for every plugin. The base
    shape (``name`` and ``status``) is always present; whatever the plugin's
    optional ``info()`` hook returns is merged in unchanged, so older
    plugins keep working unchanged and arbitrary metadata is passed through.

    ``extra`` carries additional catalog records that never run: failed and
    disabled discovery outcomes. They are appended after the live plugins,
    preserving their provided status keys (``disabled``, ``failed``).
    Names already present in the live list are never duplicated.
    """
    if status is None:
        status = {}
    catalog: list[dict[str, Any]] = []
    live_names: set[str] = set()
    for plugin in plugins:
        name = getattr(plugin, "name", None)
        if not name:
            continue
        live_names.add(name)
        entry: dict[str, Any] = {"name": name, "status": status.get(name, "registered")}
        entry.update(plugin_info(plugin))
        catalog.append(entry)
    for record in extra or []:
        name = record.get("name")
        if not isinstance(name, str) or not name or name in live_names:
            continue
        live_names.add(name)
        catalog.append(dict(record))
    return catalog


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
