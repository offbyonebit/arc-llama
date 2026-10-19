"""Independent project copies and conflict-checked source apply/undo."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

EXCLUDED = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".ssh",
    ".aws",
    ".env",
}
MAX_FILES = 4000
MAX_BYTES = 128 * 1024 * 1024
MAX_FILE = 16 * 1024 * 1024


class WorkConflictError(ValueError):
    """Source preflight failed before any writes."""


def excluded(name: str) -> bool:
    return name in EXCLUDED or name.startswith((".venv", ".env."))


def safe_path(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("Project root was replaced by a symbolic link")
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p == ".." or excluded(p) for p in path.parts):
        raise ValueError("Use a relative project file path outside excluded folders.")
    target = root / path
    probe = root
    for part in path.parts:
        probe = probe / part
        if probe.is_symlink() or (hasattr(probe, "is_junction") and probe.is_junction()):
            raise ValueError("Symbolic links and junctions are not supported in Work files.")
        if probe != target and probe.exists() and not probe.is_dir():
            raise ValueError("A parent path is a file rather than a directory")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path leaves the project copy.")
    return target


def inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        parent = Path(current)
        dirs[:] = sorted(
            d
            for d in dirs
            if not excluded(d)
            and not (parent / d).is_symlink()
            and (parent / d).resolve().is_relative_to(root.resolve())
        )
        for name in sorted(files):
            if excluded(name):
                continue
            path = parent / name
            if path.is_symlink() or not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            if size > MAX_FILE or total > MAX_BYTES or len(result) >= MAX_FILES:
                raise ValueError(
                    "Project exceeds Work copy limits (4000 files, 16 MiB per file, 128 MiB total). Select a smaller project folder."
                )
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def copy_project(source: Path, dest: Path) -> dict[str, str]:
    manifest = inventory(source)
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for name, expected in manifest.items():
            src = safe_path(source, name)
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = src.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(
                    "Project changed while taking its snapshot. Retry creating the job."
                )
            target.write_bytes(data)
            target.chmod(src.stat().st_mode & 0o777)
    except BaseException:
        shutil.rmtree(dest)
        raise
    return manifest


def changes(base: Path, tree: Path) -> tuple[dict[str, str | None], str]:
    before, after = inventory(base), inventory(tree)
    changed = {
        name: after.get(name)
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    }
    parts = []
    for name in changed:
        old = (base / name).read_bytes() if name in before else b""
        new = (tree / name).read_bytes() if name in after else b""
        if b"\0" in old + new:
            parts.append(f"Binary file changed: {name}\n")
        else:
            parts.extend(
                difflib.unified_diff(
                    old.decode("utf-8", "replace").splitlines(True),
                    new.decode("utf-8", "replace").splitlines(True),
                    fromfile=f"before/{name}",
                    tofile=f"after/{name}",
                )
            )
        if sum(len(p) for p in parts) > 120_000:
            parts.append("\nDiff preview truncated. File apply still uses complete contents.\n")
            break
    return changed, "".join(parts)


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("A changed path is now a directory; resolve it manually before applying.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_file(path: Path, data: bytes | None, mode: int = 0o644) -> None:
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".arc-work-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_update(
    source: Path, base: Path, tree: Path, receipt: dict[str, Any], *, undo: bool = False
) -> None:
    """Check every affected path first. An interrupted apply can be undone, not replayed."""
    baseline = inventory(base)
    for name, applied in receipt.items():
        try:
            current = digest(safe_path(source, name))
        except ValueError as exc:
            raise WorkConflictError(str(exc)) from exc
        expected = (baseline.get(name), applied) if undo else (baseline.get(name),)
        if current not in expected:
            raise WorkConflictError(
                f"Source changed independently: {name}. Nothing was overwritten."
            )
    # Validate all reads before writing; snapshot source for rollback on ordinary I/O failure.
    prior = {
        name: (safe_path(source, name).read_bytes() if safe_path(source, name).is_file() else None)
        for name in receipt
    }
    modes = {
        name: (
            safe_path(source, name).stat().st_mode & 0o777
            if safe_path(source, name).is_file()
            else 0o644
        )
        for name in receipt
    }
    wanted = {}
    for name in receipt:
        candidate = safe_path(base if undo else tree, name)
        content = candidate.read_bytes() if candidate.is_file() else None
        wanted[name] = content
        if (
            not undo
            and (hashlib.sha256(content).hexdigest() if content is not None else None)
            != receipt[name]
        ):
            raise WorkConflictError("Project copy changed after review. Review its changes again.")
    written = []
    try:
        for name, data in wanted.items():
            target = safe_path(source, name)
            atomic_file(target, data, modes[name])
            written.append(name)
    except BaseException:
        for name in reversed(written):
            atomic_file(safe_path(source, name), prior[name], modes[name])
        raise


def revision(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
