"""Persistent per-model performance history.

``ModelTimings`` keeps the last few dozen samples in memory, which is right
for "how is it doing now" and useless for "did that driver update slow it
down". ``PerfHistory`` rolls the same real-traffic samples up into one point
per model, metric and hour (median and count) and appends finished hours to
``<state_dir>/perf-history.jsonl``. Points older than ``RETENTION_DAYS`` are
dropped when the file is loaded.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("arc_llama.perf_history")

FILE_NAME = "perf-history.jsonl"
BUCKET_SECONDS = 3600
RETENTION_DAYS = 90
METRICS = ("generation_tok_s", "ttft_s")


class PerfHistory:
    def __init__(self, path: Path, *, clock: Any = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._points: list[dict[str, Any]] = []
        # (model, metric) -> (bucket start, samples)
        self._open: dict[tuple[str, str], tuple[int, list[float]]] = {}
        self._load()

    @classmethod
    def for_state_dir(cls, state_dir: str | Path) -> PerfHistory:
        return cls(Path(state_dir).expanduser() / FILE_NAME)

    def _bucket(self, at: float) -> int:
        return int(at // BUCKET_SECONDS * BUCKET_SECONDS)

    def _load(self) -> None:
        cutoff = self._clock() - RETENTION_DAYS * 86400
        kept: list[dict[str, Any]] = []
        dropped = False
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("could not read %s: %s", self.path, exc)
            return
        for line in lines:
            try:
                point = json.loads(line)
                if not (
                    isinstance(point, dict)
                    and isinstance(point.get("model"), str)
                    and point.get("metric") in METRICS
                    and isinstance(point.get("t"), (int, float))
                    and isinstance(point.get("median"), (int, float))
                ):
                    raise ValueError
            except ValueError:
                dropped = True
                continue
            if point["t"] < cutoff:
                dropped = True
                continue
            kept.append(point)
        self._points = kept
        if dropped:
            self._rewrite()

    def _rewrite(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                "".join(json.dumps(p, separators=(",", ":")) + "\n" for p in self._points),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("could not rewrite %s: %s", self.path, exc)

    def _close_bucket(self, key: tuple[str, str], start: int, samples: list[float]) -> None:
        if not samples:
            return
        point = {
            "t": start,
            "model": key[0],
            "metric": key[1],
            "median": round(statistics.median(samples), 3),
            "n": len(samples),
        }
        self._points.append(point)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(point, separators=(",", ":")) + "\n")
        except OSError as exc:
            log.warning("could not append to %s: %s", self.path, exc)

    def add(self, model: str, metric: str, value: float) -> None:
        if metric not in METRICS or not math.isfinite(value) or value < 0:
            return
        bucket = self._bucket(self._clock())
        key = (model, metric)
        with self._lock:
            start, samples = self._open.get(key, (bucket, []))
            if start != bucket:
                self._close_bucket(key, start, samples)
                start, samples = bucket, []
            samples.append(float(value))
            self._open[key] = (start, samples)

    def flush(self) -> None:
        """Write every open bucket (called on shutdown)."""
        with self._lock:
            for key, (start, samples) in self._open.items():
                self._close_bucket(key, start, samples)
            self._open.clear()

    def query(
        self, model: str | None = None, metric: str | None = None, days: float = 30
    ) -> list[dict[str, Any]]:
        """Closed points plus the current partial hour, oldest first."""
        cutoff = self._clock() - days * 86400
        with self._lock:
            points = list(self._points)
            for (m, met), (start, samples) in self._open.items():
                if samples:
                    points.append(
                        {
                            "t": start,
                            "model": m,
                            "metric": met,
                            "median": round(statistics.median(samples), 3),
                            "n": len(samples),
                            "partial": True,
                        }
                    )
        return sorted(
            (
                p for p in points
                if p["t"] >= cutoff
                and (model is None or p["model"] == model)
                and (metric is None or p["metric"] == metric)
            ),
            key=lambda p: (p["t"], p["model"], p["metric"]),
        )
