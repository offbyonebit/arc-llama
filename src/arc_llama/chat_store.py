"""Persistent chat-history store for arc-llama.

Chats are stored as individual JSON files under ``<state_dir>/chats``. Each
file contains a lightweight chat record with a message list so that the agent
can reference prior conversations.

Chats can optionally live in a folder (a subdirectory). Empty folder means the
legacy root directory, so existing flat chats keep working.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChatMessage:
    """A single message inside a chat."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class Chat:
    """A persisted conversation."""

    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[ChatMessage] = field(default_factory=list)
    folder: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "folder": self.folder,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chat:
        return cls(
            id=data.get("id", ""),
            title=data.get("title", "Untitled chat"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            messages=[ChatMessage.from_dict(m) for m in data.get("messages", [])],
            folder=data.get("folder", "") or "",
        )

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without full message contents."""
        return {
            "id": self.id,
            "title": self.title,
            "folder": self.folder,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }


class ChatStore:
    """On-disk JSON store for chat histories."""

    IGNORE_PATTERNS = (".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist")

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path_cache: dict[str, Path] = {}

    @staticmethod
    def _sanitize(name: str) -> str:
        """Make a user-provided name safe for the filesystem."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

    def _folder_path(self, folder: str) -> Path:
        """Return the directory for a folder; empty folder means the root."""
        if not folder:
            return self.directory
        return self.directory / self._sanitize(folder)

    def _chat_path(self, chat_id: str, folder: str) -> Path:
        """Return the path to a chat file, normalising the id."""
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", chat_id)
        return self._folder_path(folder) / f"{safe_id}.json"

    def _find_chat_path(self, chat_id: str) -> Path | None:
        """Locate a chat file anywhere in the store by id."""
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", chat_id)
        cached = self._path_cache.get(safe_id)
        if cached is not None:
            if cached.is_file():
                return cached
            self._path_cache.pop(safe_id, None)

        # Most older chats live at the store root, so check that path before
        # walking folders. Keep the recursive lookup for folder and legacy
        # layouts, but stop as soon as the matching file is found.
        root_path = self.directory / f"{safe_id}.json"
        if root_path.is_file():
            self._path_cache[safe_id] = root_path
            return root_path

        for path in self.directory.rglob(f"{safe_id}.json"):
            if path.is_file():
                self._path_cache[safe_id] = path
                return path
        return None

    def create(
        self,
        chat_id: str,
        title: str,
        messages: list[ChatMessage] | None = None,
        *,
        folder: str = "",
    ) -> Chat:
        """Create a new chat. Raises ``FileExistsError`` if it already exists."""
        if self._find_chat_path(chat_id) is not None:
            raise FileExistsError(f"Chat already exists: {chat_id}")
        now = time.time()
        chat = Chat(
            id=chat_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=list(messages or []),
            folder=folder,
        )
        self._save(chat)
        return chat

    def get(self, chat_id: str) -> Chat | None:
        """Load a chat by id. Returns None if not found."""
        path = self._find_chat_path(chat_id)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return Chat.from_dict(data)

    def save(self, chat: Chat) -> None:
        """Persist a chat, updating ``updated_at``.

        If the chat has been moved to a different folder, the old file is
        removed.
        """
        chat.updated_at = time.time()
        old_path = self._find_chat_path(chat.id)
        new_path = self._chat_path(chat.id, chat.folder)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        self._write(new_path, chat)
        if old_path is not None and old_path != new_path:
            old_path.unlink()
            self._path_cache[re.sub(r"[^a-zA-Z0-9_.-]", "_", chat.id)] = new_path
            self._prune_empty_folders()

    def _save(self, chat: Chat) -> None:
        """Persist a chat without the move-detection logic."""
        path = self._chat_path(chat.id, chat.folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(path, chat)

    def _write(self, path: Path, chat: Chat) -> None:
        path.write_text(
            json.dumps(chat.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def move(self, chat_id: str, folder: str) -> Chat:
        """Move an existing chat to a different folder."""
        chat = self.get(chat_id)
        if chat is None:
            raise FileNotFoundError(f"Chat not found: {chat_id}")
        chat.folder = folder
        self.save(chat)
        return chat

    def delete(self, chat_id: str) -> bool:
        """Delete a chat. Returns True if it existed."""
        path = self._find_chat_path(chat_id)
        if path is None:
            return False
        path.unlink()
        self._path_cache.pop(re.sub(r"[^a-zA-Z0-9_.-]", "_", chat_id), None)
        self._prune_empty_folders()
        return True

    def _prune_empty_folders(self) -> None:
        """Remove empty folder subdirectories left behind by moves/deletes."""
        if not self.directory.exists():
            return
        for path in sorted(self.directory.iterdir(), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    def list_chats(self, folder: str | None = None) -> list[Chat]:
        """Return stored chats sorted by updated_at descending.

        If ``folder`` is None, all chats are returned. Pass an empty string
        for the root folder only.
        """
        if folder is None:
            paths = self.directory.rglob("*.json")
        else:
            paths = self._folder_path(folder).glob("*.json")

        chats: list[Chat] = []
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                chats.append(Chat.from_dict(data))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(chats, key=lambda c: c.updated_at, reverse=True)

    def list_folders(self) -> list[dict[str, Any]]:
        """Return all folders with chat counts, sorted by name.

        The root/legacy folder is reported as ``{"name": "", "count": n}``.
        """
        counts: dict[str, int] = {}
        for chat in self.list_chats():
            counts[chat.folder] = counts.get(chat.folder, 0) + 1
        return [{"name": name, "count": count} for name, count in sorted(counts.items())]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        folder: str | None = None,
    ) -> list[tuple[Chat, list[int]]]:
        """Search chat titles and message contents for *query*.

        Returns a list of ``(chat, matching_message_indices)`` tuples, ordered by
        most recently updated chat first. The query is matched case-insensitively
        and supports no special syntax.
        """
        query_lower = query.lower()
        results: list[tuple[Chat, list[int]]] = []
        for chat in self.list_chats(folder=folder):
            matches: list[int] = []
            if query_lower in chat.title.lower():
                matches.append(-1)
            for idx, msg in enumerate(chat.messages):
                if query_lower in msg.content.lower():
                    matches.append(idx)
            if matches:
                results.append((chat, matches[:limit]))
        return results[:limit]

    def wipe(self) -> None:
        """Delete the entire store directory. Useful for tests."""
        if self.directory.exists():
            shutil.rmtree(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)
        self._path_cache.clear()

    def export_all(self) -> list[dict[str, Any]]:
        """Return every stored chat as a list of plain dicts."""
        return [chat.to_dict() for chat in self.list_chats()]

    def import_chats(
        self,
        data: list[dict[str, Any]],
        *,
        overwrite: bool = False,
    ) -> dict[str, int | list[str]]:
        """Import a list of chat dicts.

        Args:
            data: list of chat dicts in the format produced by ``Chat.to_dict``.
            overwrite: if True, replace an existing chat with the same id.

        Returns:
            A summary dict with ``imported``, ``skipped``, ``errors`` counts.
        """
        imported = 0
        skipped = 0
        errors: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                errors.append("skipped non-dict entry")
                continue
            chat_id = item.get("id")
            if not chat_id:
                errors.append("skipped chat with missing id")
                continue
            existing_path = self._find_chat_path(chat_id)
            if existing_path is not None and not overwrite:
                skipped += 1
                continue
            try:
                chat = Chat.from_dict(item)
            except (TypeError, ValueError) as e:
                errors.append(f"{chat_id}: invalid chat data ({e})")
                continue
            self._save(chat)
            imported += 1
        return {"imported": imported, "skipped": skipped, "errors": len(errors), "error_details": errors}
