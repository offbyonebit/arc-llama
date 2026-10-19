"""Saved Work sessions: the conversation a person had in the Work view, so it can be reopened.

The browser owns the shape of a session (tasks, steps, answers). The server only stores it
as JSON, bounded in size, one file per session, and lists them newest first."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
MAX_SESSION_BYTES = 4 * 1024 * 1024
MAX_SESSIONS = 200


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, session_id: str) -> Path:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self.directory / f"{session_id}.json"

    def save(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        path = self._path(session_id)
        record: dict[str, Any] = {
            "id": session_id,
            "title": str(data.get("title") or "Untitled")[:120],
            "folder": str(data.get("folder") or "")[:500],
            "model": str(data.get("model") or "")[:200],
            "harness": str(data.get("harness") or "")[:40],
            "compacted": data.get("compacted") if isinstance(data.get("compacted"), dict) else None,
            "runs": data.get("runs") if isinstance(data.get("runs"), list) else [],
            "updated": time.time(),
        }
        blob = json.dumps(record)
        if len(blob.encode()) > MAX_SESSION_BYTES:
            raise ValueError("session is too large to save")
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._prune()
        return self._summary(record)

    def get(self, session_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, session_id: str) -> bool:
        try:
            self._path(session_id).unlink()
            return True
        except OSError:
            return False

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in self.directory.glob("*.json") if self.directory.is_dir() else []:
            try:
                out.append(self._summary(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda item: item["updated"], reverse=True)

    @staticmethod
    def _summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "title": record.get("title", "Untitled"),
            "folder": record.get("folder", ""),
            "updated": record.get("updated", 0),
            "tasks": len(record.get("runs") or []),
        }

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[MAX_SESSIONS:]:
            old.unlink(missing_ok=True)
