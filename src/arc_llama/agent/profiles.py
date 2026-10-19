"""Work profiles: named sets of the harness's tools, so a workflow only gets what it needs.

A profile is an allow-list over what the user's harness offers: MCP servers, skills, and the
web tools. Anything not listed is switched off for that run. No profile means everything the
harness is set up with stays available."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

PROFILE_ID = re.compile(r"[A-Za-z0-9_-]{6,64}")
MAX_PROFILES = 100
MAX_ENTRIES = 300


def clean_profile(data: object) -> dict[str, Any]:
    """Validate a profile body. Raises ValueError with a message fit to show the user."""
    if not isinstance(data, dict):
        raise ValueError("Profile must be an object")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Give the profile a name")
    out: dict[str, Any] = {"name": name.strip()[:60], "description": str(data.get("description") or "")[:200]}
    for key in ("mcp", "skills", "tools_off"):
        values = data.get(key, [])
        if not isinstance(values, list) or len(values) > MAX_ENTRIES or not all(isinstance(v, str) and 0 < len(v) <= 200 for v in values):
            raise ValueError(f"{key} must be a list of names")
        out[key] = list(dict.fromkeys(values))
    web = data.get("web", False)
    if not isinstance(web, bool):
        raise ValueError("web must be true or false")
    out["web"] = web
    return out


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list(self) -> list[dict[str, Any]]:
        return sorted(self._read().values(), key=lambda p: p.get("name", "").lower())

    def get(self, profile_id: str) -> dict[str, Any] | None:
        return self._read().get(profile_id)

    def save(self, profile_id: str, data: object) -> dict[str, Any]:
        if not PROFILE_ID.fullmatch(profile_id):
            raise ValueError("invalid profile id")
        profile = clean_profile(data)
        everything = self._read()
        if profile_id not in everything and len(everything) >= MAX_PROFILES:
            raise ValueError("Too many profiles. Delete one first.")
        if any(p["name"].lower() == profile["name"].lower() for pid, p in everything.items() if pid != profile_id):
            raise ValueError(f"There is already a profile named {profile['name']}")
        record = {"id": profile_id, **profile, "updated": time.time()}
        everything[profile_id] = record
        self._write(everything)
        return record

    def delete(self, profile_id: str) -> bool:
        everything = self._read()
        if profile_id not in everything:
            return False
        del everything[profile_id]
        self._write(everything)
        return True
