"""Saved Work sessions: the conversation a person had in the Work view, so it can be reopened.

The browser owns the shape of a session (tasks, steps, answers). The server only stores it
as JSON, bounded in size, one file per session, and lists them newest first."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
MAX_SESSION_BYTES = 4 * 1024 * 1024
MAX_SESSIONS = 200


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._links_lock = threading.Lock()
        self._last_updated = 0.0

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
            "origin": data.get("origin") if data.get("origin") in ("opencode", "hermes") else "",
            "compacted": data.get("compacted") if isinstance(data.get("compacted"), dict) else None,
            "runs": data.get("runs") if isinstance(data.get("runs"), list) else [],
            "updated": self._next_timestamp(),
        }
        blob = json.dumps(record)
        if len(blob.encode()) > MAX_SESSION_BYTES:
            raise ValueError("session is too large to save")
        self._write(path, blob)
        self._prune()
        return self._summary(record)

    def _next_timestamp(self) -> float:
        # Windows' clock can return the same time.time() for saves in quick succession, which made the
        # newest-first order fall back to file order. Keep this store's timestamps strictly increasing.
        now = max(time.time(), self._last_updated + 1e-6)
        self._last_updated = now
        return now

    @staticmethod
    def _write(path: Path, blob: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _links_path(self, session_id: str) -> Path:
        return self.directory / "harness-links" / self._path(session_id).name

    def _links(self, session_id: str) -> list[str]:
        try:
            value = json.loads(self._links_path(session_id).read_text(encoding="utf-8"))
            return [sid for sid in value if isinstance(sid, str) and SESSION_ID.fullmatch(sid)] if isinstance(value, list) else []
        except (OSError, ValueError):
            return []

    def link_harness_session(self, session_id: str, native_id: str) -> None:
        """Associate IDs only, independently of browser saves and transcript contents."""
        path = self._links_path(session_id)
        if not isinstance(native_id, str) or not SESSION_ID.fullmatch(native_id) or not native_id.startswith(("oc-", "hermes-")):
            raise ValueError("invalid harness session id")
        with self._links_lock:
            links = self._links(session_id)
            if native_id not in links:
                links.append(native_id)
                self._write(path, json.dumps(links[-2000:]))

    def get(self, session_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, session_id: str) -> bool:
        try:
            self._path(session_id).unlink()
            self._links_path(session_id).unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in self.directory.glob("*.json") if self.directory.is_dir() else []:
            try:
                summary = self._summary(json.loads(path.read_text(encoding="utf-8")))
                summary["harness_sessions"] = self._links(summary["id"])
                out.append(summary)
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
            "harness": record.get("harness", ""),
            "origin": record.get("origin", ""),
        }

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[MAX_SESSIONS:]:
            old.unlink(missing_ok=True)
            self._links_path(old.stem).unlink(missing_ok=True)
