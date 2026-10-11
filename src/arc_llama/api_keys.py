"""Per-client API keys for the inference API.

Keys are stored hashed in ``<state_dir>/api-keys.json`` (mode 0600), never in
``config.toml``. The plaintext is shown once, when the key is created.

Enforcement (see ``server._require_client``): once at least one key exists,
callers that are not on loopback must present a key or the admin token as
``Authorization: Bearer <key>``. Loopback callers are always allowed, so the
bundled UI, the TUI, and local tools keep working without configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("arc_llama.api_keys")

KEY_PREFIX = "arc_"
FILE_NAME = "api-keys.json"
_FLUSH_INTERVAL_SECONDS = 30.0


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class ApiKey:
    id: str
    name: str
    sha256: str
    created_at: float
    last_used_at: float | None = None
    requests: int = 0

    def public(self) -> dict[str, object]:
        """Everything except the hash."""
        data = asdict(self)
        data.pop("sha256")
        return data


class ApiKeyStore:
    """Thread-safe key registry backed by one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._keys: dict[str, ApiKey] = {}
        self._dirty = False
        self._last_flush = time.monotonic()
        self._mtime_ns: int | None = None
        self._load()

    @classmethod
    def for_state_dir(cls, state_dir: str | Path) -> ApiKeyStore:
        return cls(Path(state_dir).expanduser() / FILE_NAME)

    def _file_mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh_locked(self) -> None:
        """Pick up keys created or revoked by another process (the CLI)."""
        if self._file_mtime() != self._mtime_ns:
            usage = {k.id: (k.requests, k.last_used_at) for k in self._keys.values()}
            self._keys = {}
            self._load()
            for key in self._keys.values():
                if key.id in usage:
                    key.requests, key.last_used_at = max(
                        usage[key.id], (key.requests, key.last_used_at),
                        key=lambda pair: pair[0],
                    )

    def _load(self) -> None:
        self._mtime_ns = self._file_mtime()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            # Refusing to start would lock out every remote client; an empty
            # store is safer only if we also make the failure loud.
            log.error("could not read %s (%s); no API keys are active", self.path, exc)
            return
        for entry in raw.get("keys", []) if isinstance(raw, dict) else []:
            try:
                key = ApiKey(
                    id=str(entry["id"]),
                    name=str(entry["name"]),
                    sha256=str(entry["sha256"]),
                    created_at=float(entry["created_at"]),
                    last_used_at=(
                        float(entry["last_used_at"]) if entry.get("last_used_at") else None
                    ),
                    requests=int(entry.get("requests", 0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._keys[key.id] = key

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        data = {"schema": 1, "keys": [asdict(k) for k in self._keys.values()]}
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, self.path)
        self._mtime_ns = self._file_mtime()
        self._dirty = False
        self._last_flush = time.monotonic()

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        with self._lock:
            self._refresh_locked()
            return len(self._keys)

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            self._refresh_locked()
            return [k.public() for k in sorted(self._keys.values(), key=lambda k: k.created_at)]

    def create(self, name: str) -> tuple[ApiKey, str]:
        """Create a key and return it with its plaintext (shown only now)."""
        name = name.strip()
        if not name or len(name) > 64:
            raise ValueError("key name must be 1..64 characters")
        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        key = ApiKey(
            id=secrets.token_hex(4),
            name=name,
            sha256=_digest(plaintext),
            created_at=time.time(),
        )
        with self._lock:
            self._refresh_locked()
            while key.id in self._keys:
                key.id = secrets.token_hex(4)
            self._keys[key.id] = key
            self._save_locked()
        return key, plaintext

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            self._refresh_locked()
            if self._keys.pop(key_id, None) is None:
                return False
            self._save_locked()
            return True

    def verify(self, presented: str) -> ApiKey | None:
        """Return the matching key and record its use, or None."""
        if not presented:
            return None
        digest = _digest(presented)
        with self._lock:
            self._refresh_locked()
            match = None
            for key in self._keys.values():
                # Compare every entry so timing does not reveal position.
                if secrets.compare_digest(key.sha256, digest):
                    match = key
            if match is None:
                return None
            match.requests += 1
            match.last_used_at = time.time()
            self._dirty = True
            if time.monotonic() - self._last_flush > _FLUSH_INTERVAL_SECONDS:
                self._flush_locked()
            return match

    def _flush_locked(self) -> None:
        try:
            self._save_locked()
        except OSError as exc:
            log.warning("could not record API key usage: %s", exc)

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._flush_locked()
