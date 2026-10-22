"""Read native harness text history without launching a harness or changing its files.

Opening a native conversation produces a Work copy. Follow-ups use Work's normal
bounded conversation context, selected local model, tools, and approval policy.
Native permissions, credentials, system prompts, and tool calls are never replayed.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from arc_llama.agent.sessions import MAX_SESSION_BYTES, MAX_SESSIONS, SESSION_ID

MAX_MESSAGES = 2000


class NativeHistoryError(ValueError):
    """A native database exists but cannot safely supply the requested history."""


def history_paths() -> dict[str, Path]:
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    hermes = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return {"opencode": data / "opencode" / "opencode.db", "hermes": hermes / "state.db"}


@contextlib.contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    # mode=ro refuses a missing file and retains SQLite's live WAL visibility.
    # immutable=1 would miss recent conversations in the running harness.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + 2
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        yield connection
    finally:
        connection.close()


def _id(harness: str, native_id: str) -> str:
    return ("oc-" if harness == "opencode" else "hermes-") + native_id


def _summary(harness: str, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": _id(harness, row["id"]), "title": row["title"] or "Untitled",
        "folder": row["folder"] or "", "updated": row["updated"] or 0,
        "tasks": row["tasks"], "harness": harness, "native": True,
    }


def _rows(connection: sqlite3.Connection, harness: str, native_id: str | None = None) -> list[sqlite3.Row]:
    if harness == "opencode":
        query = """SELECT s.id, s.title, s.directory AS folder, s.time_updated / 1000.0 AS updated,
            (SELECT count(*) FROM message m WHERE m.session_id = s.id AND
             json_extract(CASE WHEN json_valid(m.data) THEN m.data ELSE '{}' END, '$.role') = 'user') AS tasks
            FROM session s WHERE s.parent_id IS NULL AND s.time_archived IS NULL"""
    else:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(sessions)")}
        folder = "s.cwd" if "cwd" in columns else "''"
        query = f"""SELECT s.id, s.title, {folder} AS folder,
            COALESCE((SELECT max(timestamp) FROM messages WHERE session_id = s.id), s.ended_at, s.started_at) AS updated,
            (SELECT count(*) FROM messages WHERE session_id = s.id AND role = 'user') AS tasks
            FROM sessions s WHERE s.parent_session_id IS NULL"""
        if "archived" in columns:
            query += " AND s.archived = 0"
    parameters: tuple[Any, ...] = ()
    if native_id is not None:
        query += " AND s.id = ?"
        parameters = (native_id,)
    query += " ORDER BY updated DESC LIMIT ?"
    return list(connection.execute(query, (*parameters, MAX_SESSIONS)))


def list_native_sessions() -> tuple[list[dict[str, Any]], list[str]]:
    sessions: list[dict[str, Any]] = []
    warnings: list[str] = []
    for harness, path in history_paths().items():
        if not path.is_file():
            continue
        try:
            with _connect(path) as connection:
                sessions.extend(_summary(harness, row) for row in _rows(connection, harness)
                                if SESSION_ID.fullmatch(_id(harness, row["id"])))
        except (sqlite3.Error, OSError, ValueError):
            warnings.append(f"{harness.title()} history could not be read. Saved Work sessions are still available.")
    return sorted(sessions, key=lambda item: item["updated"], reverse=True), warnings


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n\n".join(p["text"] for p in value
                            if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text")
                            and isinstance(p.get("text"), str))
    return ""


def _messages(connection: sqlite3.Connection, harness: str, native_id: str) -> list[tuple[str, str, float]]:
    if harness == "opencode":
        queries = [("message", "data"), ("part", "data")]
    else:
        queries = [("messages", "content")]
    for table, column in queries:
        size = connection.execute(
            f"SELECT count(*), COALESCE(sum(length(CAST({column} AS BLOB))), 0) FROM {table} WHERE session_id = ?",
            (native_id,),
        ).fetchone()
        if size[0] > MAX_MESSAGES * 20 or size[1] > MAX_SESSION_BYTES:
            raise NativeHistoryError("This harness session is too large to import. Open it in the harness or choose a shorter session.")
    out: list[tuple[str, str, float]] = []
    if harness == "opencode":
        messages = connection.execute(
            "SELECT id, data, time_created FROM message WHERE session_id = ? ORDER BY time_created, id LIMIT ?",
            (native_id, MAX_MESSAGES + 1),
        ).fetchall()
        parts: dict[str, list[str]] = {}
        for row in connection.execute(
            "SELECT message_id, data FROM part WHERE session_id = ? ORDER BY time_created, id", (native_id,),
        ):
            part = json.loads(row["data"])
            if isinstance(part, dict) and part.get("type") == "text" and not part.get("ignored"):
                parts.setdefault(row["message_id"], []).append(_content(part.get("text")))
        for row in messages:
            message = json.loads(row["data"])
            if not isinstance(message, dict):
                continue
            out.append((message.get("role", ""), "\n\n".join(parts.get(row["id"], [])), row["time_created"]))
    else:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(messages)")}
        active = " AND active = 1" if "active" in columns else ""
        messages = connection.execute(
            f"SELECT role, content, timestamp FROM messages WHERE session_id = ?{active} ORDER BY timestamp, id LIMIT ?",
            (native_id, MAX_MESSAGES + 1),
        ).fetchall()
        for row in messages:
            content = row["content"] or ""
            if content.startswith("["):
                with contextlib.suppress(ValueError):
                    content = _content(json.loads(content))
            out.append((row["role"], content, row["timestamp"] * 1000))
    if len(messages) > MAX_MESSAGES:
        raise NativeHistoryError("This harness session has too many messages to import. Open it in the harness or choose a shorter session.")
    return out


def get_native_session(session_id: str) -> dict[str, Any] | None:
    if not SESSION_ID.fullmatch(session_id):
        return None
    harness = "opencode" if session_id.startswith("oc-") else "hermes" if session_id.startswith("hermes-") else None
    if harness is None:
        return None
    native_id = session_id[len("oc-") if harness == "opencode" else len("hermes-"):]
    path = history_paths()[harness]
    if not path.is_file():
        return None
    try:
        with _connect(path) as connection:
            rows = _rows(connection, harness, native_id)
            if not rows:
                return None
            record = _summary(harness, rows[0])
            runs: list[dict[str, Any]] = []
            for role, text, timestamp in _messages(connection, harness, native_id):
                if role not in ("user", "assistant") or not text:
                    continue
                if role == "user":
                    runs.append({"task": text, "answer": [], "steps": [], "status": "done",
                                 "startedAt": timestamp, "endedAt": timestamp, "error": "",
                                 "folder": record["folder"], "folderName": Path(record["folder"]).name or "project",
                                 "harness": harness, "model": "", "imported": True})
                elif runs:
                    runs[-1]["answer"].append(text)
                    runs[-1]["endedAt"] = timestamp
            record.update(runs=runs, model="", compacted=None, origin=harness)
            if len(json.dumps(record).encode()) > MAX_SESSION_BYTES:
                raise NativeHistoryError("This harness session is too large to import. Open it in the harness or choose a shorter session.")
            return record
    except NativeHistoryError:
        raise
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exc:
        raise NativeHistoryError("The harness history could not be read. Open the session in its harness and try again.") from exc
