"""Observe bounded SSE events without changing the forwarded response bytes."""
from __future__ import annotations

import json
import math
from typing import Any


def generation_rate(payload: Any) -> float | None:
    """Use backend generation timings, never whole-request elapsed time."""
    if not isinstance(payload, dict):
        return None
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        usage = payload.get("usage")
        timings = usage.get("timings") if isinstance(usage, dict) else None
    if not isinstance(timings, dict):
        return None
    value = timings.get("predicted_per_second")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value > 0:
            return float(value)
    count, elapsed = timings.get("predicted_n"), timings.get("predicted_ms")
    if (isinstance(count, (int, float)) and not isinstance(count, bool)
            and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)):
        if math.isfinite(count) and math.isfinite(elapsed) and count > 0 and elapsed > 0:
            rate = count * 1000.0 / elapsed
            if math.isfinite(rate):
                return rate
    return None


def has_generated_content(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return False
    for choice in payload["choices"]:
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("text"), str) and choice["text"]:
            return True
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if any(isinstance(delta.get(k), str) and delta[k] for k in ("content", "reasoning_content", "reasoning")):
            return True
        for call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                if call["function"].get("name") or call["function"].get("arguments"):
                    return True
    return False


class StreamMetricsObserver:
    """Parse complete data lines across network chunks with a fixed memory cap.

    Oversized/malformed events are ignored for metrics and still forwarded by
    the caller. Role-only and heartbeat events do not count as generated tokens.
    This observer supports the single-data-line JSON SSE emitted by llama-server.
    """

    def __init__(self, max_line_bytes: int = 65_536):
        self.max_line_bytes = max_line_bytes
        self._pending = bytearray()
        self._discarding = False
        self.first_token_at: float | None = None
        self.generation_tok_s: float | None = None

    def feed(self, chunk: bytes, now: float) -> None:
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            end = len(chunk) if newline < 0 else newline
            if not self._discarding:
                if len(self._pending) + end - offset > self.max_line_bytes:
                    self._pending.clear()
                    self._discarding = True
                else:
                    self._pending.extend(chunk[offset:end])
            if newline < 0:
                break
            if not self._discarding:
                self._line(bytes(self._pending), now)
            self._pending.clear()
            self._discarding = False
            offset = newline + 1

    def _line(self, line: bytes, now: float) -> None:
        if not line.startswith(b"data:"):
            return
        try:
            payload = json.loads(line[5:].strip())
        except (ValueError, UnicodeError, RecursionError):
            return
        if self.first_token_at is None and has_generated_content(payload):
            self.first_token_at = now
        rate = generation_rate(payload)
        if rate is not None:
            self.generation_tok_s = rate
