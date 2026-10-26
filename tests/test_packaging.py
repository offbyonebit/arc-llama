"""The sdist must not ship local-only notes or validation artefacts.

Hatchling honours .gitignore but not .git/info/exclude, so files hidden only
there (agent notes) leaked into an sdist built from the working checkout.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _excluded(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if any(fnmatch.fnmatch(part, pattern.rstrip("/")) for part in _prefixes(path)):
                return True
        elif fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
            return True
    return False


def _prefixes(path: str) -> list[str]:
    parts = path.split("/")[:-1]
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def test_sdist_excludes_local_notes_and_validation_artefacts():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    for path in (
        "CLAUDE.md",
        "LOCAL_REPO_BOUNDARIES.md",
        "docs/release-validation-windows-2026-10-22/windows-ui-first-failure.png",
        "docs/release-validation-linux-2026-10-03.md",
        ".venv-test/bin/python",
    ):
        assert _excluded(path, patterns), path
    for path in ("src/arc_llama/server.py", "README.md", "docs/testing.md", "tests/test_packaging.py"):
        assert not _excluded(path, patterns), path
