"""Structured, safe-to-report failures from local model startup."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

_HTTP_STATUS = {
    "model_missing": 404,
    "draft_missing": 404,
    "runtime_missing": 503,
    "runtime_incompatible": 503,
    "gpu_unavailable": 503,
    "port_in_use": 409,
    "out_of_memory": 507,
    "startup_timeout": 504,
    "process_exited": 503,
    "switch_busy": 409,
}

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(authorization|password|passwd|secret|token|api[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_FLAG = re.compile(
    r"^--?(?:authorization|password|passwd|secret|token|api[-_]?key|hf[-_]?token)$",
    re.IGNORECASE,
)


def _redact_argv(values: Sequence[object]) -> list[Any]:
    redacted: list[Any] = []
    hide_next = False
    for raw in values:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        if not isinstance(raw, str):
            redacted.append(redact_details(raw))
            continue
        flag, separator, _value = raw.partition("=")
        if _SECRET_FLAG.match(flag):
            if separator:
                redacted.append(f"{flag}=[REDACTED]")
            else:
                redacted.append(raw)
                hide_next = True
        else:
            redacted.append(raw)
    return redacted


def redact_details(value: Any, *, key: str = "") -> Any:
    """Return JSON-compatible diagnostic data with common secrets removed."""
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_details(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if key.lower() in {"argv", "command", "args"}:
            return _redact_argv(value)
        return [redact_details(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class StartupFailureError(RuntimeError):
    """A model-start failure with a stable category and safe diagnostics."""

    def __init__(
        self,
        category: str,
        message: str,
        action: str,
        *,
        details: Mapping[str, Any] | None = None,
        diagnostics_id: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.action = action
        self.details = redact_details(details or {})
        self.diagnostics_id = diagnostics_id or f"{category}-{uuid.uuid4().hex[:12]}"
        self.http_status = http_status or _HTTP_STATUS.get(category, 503)

    def to_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        error: dict[str, Any] = {
            "category": self.category,
            "message": self.message,
            "action": self.action,
            "diagnostics_id": self.diagnostics_id,
        }
        if include_details:
            error["details"] = bounded_details(self.details)
        return {"error": error}


# Diagnostics payloads are unbounded argv/log tails from adversarial startup
# failures; keep responses readable and capped when they reach an API body.
_MAX_DETAIL_CHARS = 8_000


def bounded_details(details: Any, *, limit: int = _MAX_DETAIL_CHARS) -> dict[str, Any]:
    """Bound diagnostics structure as well as strings before returning JSON."""
    if not isinstance(details, Mapping):
        return {}
    remaining = 128

    def clip(value: Any, depth: int = 0) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining <= 0 or depth > 6:
            return "[truncated]"
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if remaining <= 0:
                    break
                result[str(key)[:128]] = clip(item, depth + 1)
            return result
        if isinstance(value, (list, tuple)):
            result_list = []
            for item in value:
                if remaining <= 0:
                    break
                result_list.append(clip(item, depth + 1))
            return result_list
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + "[truncated]"
        return value

    result = clip(details)
    encoded = json.dumps(result, ensure_ascii=True, default=str)
    if len(encoded) > limit:
        # A JSON-encoded preview can expand each character to six bytes.
        return {"truncated": True, "preview": encoded[:max(0, (limit - 100) // 6)]}
    return result
