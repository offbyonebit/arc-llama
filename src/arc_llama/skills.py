"""User skill loader.

Skills are plain Python files dropped into a configured directory (default
``~/.config/arc-llama/skills``). Each file can register tools by importing the
global tool registry from ``arc_llama.agent.tools``:

    from arc_llama.agent.tools import tool

    @tool(
        name="get_timestamp",
        description="Return the current time in ISO format.",
        parameters={"type": "object", "properties": {}},
    )
    def _get_timestamp(arguments, ctx):
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

The loader imports every ``*.py`` file once at server startup. Import errors are
caught and logged so a broken skill cannot prevent the server from starting.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

log = logging.getLogger("arc_llama.skills")


def _load_skill_module(path: Path) -> ModuleType | None:
    """Import a single skill file and return its module."""
    module_name = f"arc_llama.skill.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        log.error("Could not create module spec for skill: %s", path)
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        log.exception("Failed to load skill %s: %s", path, e)
        sys.modules.pop(module_name, None)
        return None
    return module


def load_skills(skills_dir: str | Path) -> None:
    """Import every ``*.py`` file in *skills_dir*.

    Files are imported in sorted order. Broken skills are logged and skipped.
    """
    directory = Path(skills_dir).expanduser()
    if not directory.is_dir():
        log.debug("Skills directory does not exist, skipping: %s", directory)
        return

    for path in sorted(directory.glob("*.py")):
        log.info("Loading skill: %s", path)
        _load_skill_module(path)
