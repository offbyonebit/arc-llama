"""Generate a new plugin package skeleton (``arc-llama plugin new``)."""

from __future__ import annotations

import re
from pathlib import Path

from arc_llama.plugin_api import PLUGIN_API_VERSION

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def validate_plugin_name(name: str) -> str:
    """Return ``name`` if it is a usable plugin id, else raise ValueError."""
    if not _NAME_RE.match(name):
        raise ValueError(
            "plugin name must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores (max 40 characters)"
        )
    return name


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) + "Plugin"


def scaffold_files(name: str) -> dict[str, str]:
    """Return ``{relative path: content}`` for a new plugin package."""
    validate_plugin_name(name)
    package = f"arc_llama_{name}"
    dist = f"arc-llama-{name.replace('_', '-')}"
    cls = _class_name(name)
    major = PLUGIN_API_VERSION.split(".")[0]
    return {
        "pyproject.toml": f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{dist}"
version = "0.1.0"
description = "An arc-llama plugin"
requires-python = ">=3.10"
dependencies = ["arc-llama"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "httpx>=0.27"]

[project.entry-points."arc_llama.plugins"]
{name} = "{package}:create_plugin"

[tool.hatch.build.targets.wheel]
packages = ["src/{package}"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
''',
        f"src/{package}/__init__.py": f'''"""{name} plugin for arc-llama."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from arc_llama.plugin_api import MODEL_LOADED, Plugin, plugin_context


class {cls}(Plugin):
    name = "{name}"
    # Plugin API this package targets. Bump the minor number when you start
    # relying on newer additions; a different major number will not load.
    requires_api = "{PLUGIN_API_VERSION}"

    def __init__(self) -> None:
        self.last_loaded: str | None = None

    def register(self, app: FastAPI) -> None:
        ctx = plugin_context(app)

        @app.get("/plugins/{name}/status")
        async def status() -> dict[str, Any]:
            return {{
                "plugin": self.name,
                "api_version": ctx.api_version,
                "models": ctx.models(),
                "last_loaded": self.last_loaded,
            }}

        # Privileged routes should require the core admin token, and GPU work
        # should hold a lease so text models drain and step aside first.
        @app.post("/plugins/{name}/work", dependencies=[ctx.require_admin])
        async def work() -> dict[str, str]:
            async with ctx.gpu_lease(self.name):
                pass  # GPU-heavy work goes here.
            return {{"status": "done"}}

    def startup(self, app: FastAPI) -> None:
        plugin_context(app).subscribe(MODEL_LOADED, self._on_model_loaded)

    def _on_model_loaded(self, event: str, payload: dict[str, Any]) -> None:
        self.last_loaded = payload.get("model")

    def info(self) -> dict[str, Any]:
        return {{
            "version": "0.1.0",
            "description": "Describe what this plugin adds.",
            "api": ["/plugins/{name}/status", "/plugins/{name}/work"],
        }}


def create_plugin() -> {cls}:
    return {cls}()
''',
        "tests/test_plugin.py": f'''from __future__ import annotations

import httpx
from fastapi import FastAPI

from arc_llama.plugin_api import api_compatible, plugin_context
from {package} import create_plugin


def test_targets_supported_plugin_api() -> None:
    assert api_compatible(create_plugin().requires_api)


async def test_status_route() -> None:
    app = FastAPI()
    plugin = create_plugin()
    plugin.register(app)
    plugin_context(app)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/plugins/{name}/status")
    assert response.status_code == 200
    assert response.json()["plugin"] == "{name}"
    assert response.json()["api_version"].startswith("{major}.")
''',
        "README.md": f'''# {dist}

An [arc-llama](https://github.com/offbyonebit/arc-llama) plugin.

```bash
pip install -e ".[dev]"
pytest
arc-llama doctor        # lists the plugin and its status
arc-llama serve         # routes appear under /plugins/{name}/
```

Import only from `arc_llama.plugin_api`; other core modules are private and
may change without notice.
''',
    }


def write_scaffold(name: str, target: Path) -> list[Path]:
    """Write a new plugin package into ``target`` (must not already exist)."""
    files = scaffold_files(name)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} already exists and is not empty")
    written: list[Path] = []
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
