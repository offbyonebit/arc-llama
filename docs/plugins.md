# Plugins

arc-llama's core is a simple inference machine: it discovers GGUFs, manages
`llama-server` subprocesses, and exposes an OpenAI-compatible API. Feature
projects — audio, vision, custom admin surfaces — should live *outside* the
core as installable add-ons.

Plugins are discovered through Python packaging entry points and loaded
lazily, so the core's dependency set is unchanged and a plugin's optional
dependencies are never imported at core import time.

## The public plugin API (`arc_llama.plugin_api`)

Plugins should import only from `arc_llama.plugin_api`. Every other core
module (`router`, `launcher`, `server`, ...) is private and may change in any
release. The module carries a `MAJOR.MINOR` version, currently `1.0`:
additions bump the minor number, breaking changes bump the major number.

Declare the version your plugin was written against:

```python
from arc_llama.plugin_api import Plugin, plugin_context

class AudioPlugin(Plugin):
    name = "audio"
    requires_api = "1.0"
```

A plugin whose required major version differs from the core's, or whose
required minor version is newer, is not loaded. It appears in
`GET /admin/plugins`, `arc-llama plugin list`, and `arc-llama doctor` with
status `incompatible` and the reason. Plugins without `requires_api` load as
before.

`plugin_context(app)` returns a `PluginContext` with:

| Member | Purpose |
| --- | --- |
| `require_admin` | FastAPI dependency enforcing the core admin token: `dependencies=[ctx.require_admin]` |
| `gpu_lease(owner, exclusive=True)` | `async with` block that drains text requests and evicts resident models before GPU work, and holds new loads until it exits |
| `models()` | registered local models with `loaded`, `aliases`, and `vision` flags |
| `gpus()` | configured GPUs with arch, backend, and VRAM |
| `base_url` | loopback URL of the core OpenAI-compatible API |
| `subscribe(event, callback)` | receive `model_loaded`, `model_stopped`, and `model_load_failed`; returns an unsubscribe function |
| `api_version` | the running core's plugin API version |

Event callbacks run synchronously inside the core and must return quickly.
Schedule a task for anything slow. A failing callback is logged and ignored.

### Starting a new plugin

```bash
arc-llama plugin new audio_tools
cd arc-llama-audio-tools
pip install -e ".[dev]"
pytest
```

The generated package declares its entry point, targets the current plugin
API, shows an admin-protected route that holds a GPU lease, subscribes to a
lifecycle event, and includes a test. `arc-llama plugin list` shows whether
each installed plugin loads.

## The plugin contract

A plugin is any object exposing:

| Member | Type | When |
| --- | --- | --- |
| `name` | `str` | stable, unique identifier |
| `register(app)` | sync | once, at app creation — add FastAPI routes here |
| `startup(app)` | sync or async | inside the app lifespan, on start |
| `shutdown(app)` | sync or async | inside the app lifespan, on stop |
| `info()` | sync, optional | best-effort UI metadata, read after loading |

`startup` and `shutdown` are optional. Every hook is isolated: an exception in
one plugin is logged and does not affect the core or other plugins.

## Optional UI metadata: `info()`

To appear with more than just a status line in dashboards (the bundled web
UI's *Plugins* panel, `GET /admin/plugins`), a plugin may expose an `info()`
hook returning a JSON-serializable dict:

```python
class AudioPlugin(Plugin):
    name = "audio"

    def info(self) -> dict:
        return {
            "version": "1.2.0",                  # optional str
            "description": "Transcription routes",  # optional str
            "ui": {"panel": "Audio", "url": "/audio"},  # optional, any JSON
            "api": ["/v1/audio/transcriptions"],  # optional, any JSON
        }
```

The hook is purely advisory: it is read defensively after loading, a missing
or failing hook simply omits those fields, and it never influences loading
or startup. Anything the server cannot render is ignored; the catalog shape
is always `name` plus a stable `status` key (`registered`, `active`, or
`error`), with every other field passed through from `info()` untouched.

The simplest plugin subclasses `arc_llama.plugins.Plugin`:

```python
from fastapi import FastAPI
from arc_llama.plugins import Plugin

class AudioPlugin(Plugin):
    name = "audio"

    def register(self, app: FastAPI) -> None:
        @app.get("/v1/audio/transcriptions")
        async def transcriptions():
            ...

    async def startup(self, app: FastAPI) -> None:
        # import heavy deps here, not at module import time
        ...
```

## Install / discovery flow

1. A plugin package declares an entry point in its own `pyproject.toml`:

   ```toml
   [project.entry-points."arc_llama.plugins"]
   audio = "arc_llama_audio.plugin:create_plugin"
   ```

   The value is a `module:attr` reference. `attr` may be a `Plugin` subclass
   (instantiated with no arguments), a zero-argument callable returning a
   plugin, or a plugin instance.

2. `arc_llama.server.create_app` calls `load_plugins()`, which reads the
   `arc_llama.plugins` entry-point group via `importlib.metadata` and
   instantiates each plugin. This happens at app creation, not at import time.

3. `register_plugins` calls each plugin's `register(app)` before the static
   web UI is mounted, so plugin routes are not shadowed by the catch-all UI.

4. `startup_plugins` / `shutdown_plugins` run inside the app lifespan.

### Web UI and `GET /admin/plugins`

The bundled web UI shows a *Plugins* panel on the dashboard, populated from
`GET /admin/plugins`. That route is read-only and returns the catalog of
plugins discovered at app creation: each entry carries at minimum the
plugin's `name` and a `status` key, plus whatever metadata the plugin
publishes through its optional `info()` hook (typically `version`,
`description`, `ui`, and `api`). A plugin whose `register` hook raised is
reported with `status: "error"` instead of being hidden. The route never
runs plugin code and serves an empty list when no plugins are installed.

### Declarative dashboard actions

Plugins may expose buttons that users can place through the dashboard's
**Customize UI** editor:

```python
def info(self):
    return {"ui": {"actions": [{
        "id": "generate", "label": "Generate image", "icon": "image",
        "placement": "toolbar", "route": "/plugins/vision/generate",
        "method": "POST", "composer": {"mode": "text", "result": "image"},
    }]}}
```

The dashboard accepts only string action metadata and never executes
plugin-supplied JavaScript. Placement and hidden state are stored through
`/admin/ui/layout`.

### Plugin pages

A plugin that serves its own HTML can list pages that the dashboard links to
from its plugin card:

```python
def info(self):
    return {"ui": {"pages": [{"id": "main", "label": "Open audio", "path": "/plugins/audio/"}]}}
```

Paths must be same-origin and start with `/plugins/`; anything else is
dropped. Pages open in a new tab, so plugin markup never runs inside the
dashboard.

### Enabling / disabling

By default every discovered plugin is loaded. To restrict which plugins load,
set the `ARC_LLAMA_PLUGINS` environment variable to a comma-separated list of
plugin names:

```bash
ARC_LLAMA_PLUGINS=audio,vision arc-llama serve
```

## Lifecycle

```
import arc_llama            # no plugin code imported
create_app()                # discover + instantiate + register() routes
  └─ lifespan startup       # startup() hooks (sync or async)
  └─ lifespan shutdown      # shutdown() hooks (sync or async)
```

A plugin that fails to import or instantiate is skipped with a warning, so a
broken add-on can never prevent the core from starting. When no plugins are
installed, `load_plugins()` returns an empty list and the core behaves exactly
as before.

## Security considerations

- **Trust boundary.** Plugins run in-process with the same privileges as
  arc-llama. Only install plugins from sources you trust; a plugin can read
  the config (including `server.admin_token`), access the filesystem, and open
  network connections.
- **No sandboxing.** There is no isolation between plugins and the core. A
  malicious or buggy plugin can crash the process or exfiltrate data.
- **Admin surface.** Plugin routes are *not* automatically gated by the admin
  token. If a plugin exposes privileged operations, it must apply its own
  authorization (e.g. reuse `arc_llama.server._require_admin` or check
  `app.state.cfg.server.admin_token`).
- **Lazy imports.** Keep heavy or optional dependencies out of module-level
  imports; import them inside `register`/`startup` so a missing dependency
  degrades gracefully instead of breaking core startup.
- **Failure isolation.** Hook exceptions are caught and logged, but a plugin
  that corrupts shared state (e.g. `app.state`) is not protected against.

## Testing a plugin without installing it

`load_plugins` accepts an explicit iterable of entry-point-like objects
(anything with a `.name` and a `.load()`), so tests can exercise discovery
without installing a package. See `tests/test_plugins.py` for a tiny fake
plugin that registers a route and records its lifecycle hooks.
### Composer actions

Plugin UI actions can opt into the main chat composer instead of opening a
plugin-specific page. Add a `composer` object to an action:

```json
{
  "id": "example.run",
  "label": "Example",
  "route": "/plugins/example/run",
  "method": "POST",
  "composer": {
    "mode": "text",
    "placeholder": "Enter a prompt…",
    "result": "json"
  }
}
```

`mode` may be `text`, `attachments`, or `custom`. Text actions use the main
prompt field; attachment actions keep the file control enabled and receive the
prompt plus extracted attachment text; custom actions can provide their own
route behavior. `result` may be `image` (the built-in image renderer) or
`json` (a compact result message). Actions without `composer` retain the
existing route/navigation behavior, so older plugins remain compatible.
