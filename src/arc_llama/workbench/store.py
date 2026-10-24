"""Transactional job records and an append-only event history."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class WorkStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        self.path = root / "work.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(kind, id));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    created REAL NOT NULL, data TEXT NOT NULL);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA synchronous=FULL")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def get(self, kind: str, identity: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM records WHERE kind=? AND id=?", (kind, identity)
            ).fetchone()
        if row is None:
            raise KeyError(identity)
        return json.loads(row[0])

    def records(self, kind: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT data FROM records WHERE kind=?", (kind,)).fetchall()
        return sorted(
            (json.loads(row[0]) for row in rows), key=lambda x: x.get("updated_at", 0), reverse=True
        )

    def save(self, kind: str, value: dict[str, Any], event: dict[str, Any] | None = None) -> None:
        value["updated_at"] = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
                (kind, value["id"], json.dumps(value)),
            )
            if event is not None:
                db.execute(
                    "INSERT INTO events(job_id,created,data) VALUES(?,?,?)",
                    (value["id"], time.time(), json.dumps(event)),
                )

    def create(self, kind: str, **fields: Any) -> dict[str, Any]:
        value = {"id": uuid.uuid4().hex, "created_at": time.time(), **fields}
        self.save(kind, value)
        return value

    def events(self, identity: str, after: int = 0) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT seq,created,data FROM events WHERE job_id=? AND seq>? ORDER BY seq LIMIT 200",
                (identity, after),
            ).fetchall()
        return [
            {"seq": seq, "created_at": created, **json.loads(data)} for seq, created, data in rows
        ]


class WorkOwnerError(RuntimeError):
    pass


class WorkOwner:
    """One process may own a Work queue; the OS releases this lock after a crash."""

    def __init__(self, path: Path):
        self.path = path
        self.stream: Any = None

    def acquire(self) -> None:
        stream = self.path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise WorkOwnerError(
                "Another server owns this Work state. Use its Work tab or configure a different state directory."
            ) from exc
        self.stream = stream

    def release(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
