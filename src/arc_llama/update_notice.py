"""Quiet, daily package-version checks; never installs or changes a runtime."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from time import time
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from arc_llama import __version__
from arc_llama.config import Config

PYPI_URL = "https://pypi.org/pypi/arc-llama/json"
CHECK_INTERVAL = 24 * 60 * 60
UPGRADE_COMMAND = "pip install -U arc-llama"


def _path(cfg: Config) -> Path:
    return Path(cfg.paths.state_dir).expanduser() / "update-notice.sqlite3"


def _notice(latest: str | None) -> dict[str, str] | None:
    try:
        if latest and not Version(latest).is_prerelease and Version(latest) > Version(__version__):
            return {"current": __version__, "latest": latest, "command": UPGRADE_COMMAND}
    except InvalidVersion:
        pass
    return None


def latest_stable(payload: Any) -> str | None:
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, dict):
        return None
    versions = []
    for text, files in releases.items():
        try:
            version = Version(text)
        except (InvalidVersion, TypeError):
            continue
        if (not version.is_prerelease and not version.is_devrelease and isinstance(files, list)
                and any(isinstance(item, dict) and not item.get("yanked", False) for item in files)):
            versions.append(version)
    return str(max(versions)) if versions else None


def _claim(path: Path, now: float) -> tuple[str | None, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=0.1)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS release_check (id INTEGER PRIMARY KEY, checked_at REAL, latest TEXT)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT checked_at, latest FROM release_check WHERE id=1").fetchone()
        latest = row[1] if row else None
        if row and now - row[0] < CHECK_INTERVAL:
            return latest, False
        # Persist before networking: failures and concurrent processes still
        # share the daily limit. No database lock is held during the request.
        db.execute("INSERT OR REPLACE INTO release_check VALUES (1, ?, ?)", (now, latest))
        db.commit()
        return latest, True


def _save(path: Path, now: float, latest: str) -> None:
    with closing(sqlite3.connect(path, timeout=0.1)) as db:
        db.execute("UPDATE release_check SET latest=? WHERE id=1 AND checked_at=?", (latest, now))
        db.commit()


def cached_notice(cfg: Config) -> dict[str, str] | None:
    if not cfg.server.check_updates or not _path(cfg).is_file():
        return None
    try:
        with closing(sqlite3.connect(_path(cfg), timeout=0.1)) as db:
            row = db.execute("SELECT latest FROM release_check WHERE id=1").fetchone()
        return _notice(row[0] if row else None)
    except (OSError, sqlite3.Error):
        return None


async def check_for_update(cfg: Config) -> dict[str, str] | None:
    if not cfg.server.check_updates:
        return None
    now = time()
    try:
        latest, due = await asyncio.to_thread(_claim, _path(cfg), now)
    except (OSError, sqlite3.Error):
        return None  # Do not make an uncached request if daily state cannot be saved.
    if due:
        try:
            async with httpx.AsyncClient(timeout=2.0, follow_redirects=False) as client:
                response = await client.get(PYPI_URL, headers={"Accept": "application/json"})
                response.raise_for_status()
                found = latest_stable(response.json())
            if found:
                latest = found
                await asyncio.to_thread(_save, _path(cfg), now, latest)
        except (httpx.HTTPError, ValueError, OSError, sqlite3.Error):
            pass
    return _notice(latest)
