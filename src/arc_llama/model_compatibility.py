"""Bounded, explicit GGUF architecture checks against the installed runtime.

A zero-tensor fixture checks architecture recognition, never successful inference.
Remote reads are pinned to a Hub commit and capped; no weight download is needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import struct
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

import httpx

from arc_llama.compatibility_io import probe_output, remote_header, remote_metadata
from arc_llama.config import Config

_MAX_HEADER = 262_144
_ARCH = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
log = logging.getLogger(__name__)
_PROBE_LOCK = Lock()
_PROBE_CACHE: dict[tuple[str, str, str], str] = {}


def header_architecture(data: bytes) -> str | None:
    """Read an early architecture field; truncated/invalid metadata stays unknown."""
    offset = 0

    def take(n: int) -> bytes:
        nonlocal offset
        if n < 0 or offset + n > len(data):
            raise ValueError("Metadata outside bounded header")
        result = data[offset : offset + n]
        offset += n
        return result

    def number(fmt: str) -> int:
        return int(struct.unpack(fmt, take(struct.calcsize(fmt)))[0])

    def string() -> str:
        return take(number("<Q")).decode("utf-8")

    def skip(kind: int, depth: int = 0) -> None:
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind in widths:
            take(widths[kind])
        elif kind == 8:
            take(number("<Q"))
        elif kind == 9 and depth == 0:
            element, count = number("<I"), number("<Q")
            if count > _MAX_HEADER:
                raise ValueError("Array too large")
            if element in widths:
                take(count * widths[element])
            else:
                for _ in range(count):
                    skip(element, depth + 1)
        else:
            raise ValueError("Unsupported field type")

    try:
        if take(4) != b"GGUF" or number("<I") not in {2, 3}:
            return None
        number("<Q")  # Tensor count; tensor contents are never read.
        count = number("<Q")
        if count > _MAX_HEADER:
            return None
        for _ in range(count):
            key, kind = string(), number("<I")
            if key == "general.architecture":
                arch = string() if kind == 8 else ""
                return arch if _ARCH.fullmatch(arch) else None
            skip(kind)
    except (ValueError, UnicodeError, struct.error):
        return None
    return None


def runtime_identity(command: str) -> tuple[str, str] | None:
    """Invalidate recognition when the executable or adjacent libraries change."""
    resolved = shutil.which(command) or command
    try:
        binary = Path(resolved).expanduser().resolve()
        if not binary.is_file():
            return None
        files = {binary}
        for pattern in ("*.so*", "*.dll", "*.dylib"):
            files.update(p.resolve() for p in binary.parent.glob(pattern) if p.is_file())
        records = []
        for path in sorted(files):
            st = path.stat()
            records.append(
                (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            )
        digest = hashlib.sha256(repr(records).encode()).hexdigest()
        return str(binary), digest
    except (OSError, RuntimeError, ValueError):
        return None


async def probe_architecture(binary: str, fingerprint: str, architecture: str) -> str:
    """Cache only definitive architecture-specific evidence; failures are retryable."""
    if not _ARCH.fullmatch(architecture):
        return "unknown"
    cache_key = (binary, fingerprint, architecture)
    with _PROBE_LOCK:
        if cache_key in _PROBE_CACHE:
            return _PROBE_CACHE[cache_key]
    returncode, output = await probe_output(binary, architecture)
    if returncode == 0:
        return "unknown"
    if re.search(
        r"unknown model architecture:\s*['\"]" + re.escape(architecture) + r"['\"]", output
    ):
        outcome = "rejected"
    elif re.search(r"key not found in model:\s*" + re.escape(architecture) + r"\.[a-z_]+", output):
        outcome = "recognized"
    else:
        log.info(
            "Compatibility loader evidence inconclusive: architecture=%s exit=%s",
            architecture,
            returncode,
        )
        return "unknown"
    with _PROBE_LOCK:
        if len(_PROBE_CACHE) >= 128:
            _PROBE_CACHE.pop(next(iter(_PROBE_CACHE)))
        _PROBE_CACHE[cache_key] = outcome
    return outcome


def unknown_assessment() -> dict[str, Any]:
    return {
        "status": "unknown",
        "label": "Compatibility unknown",
        "detail": "The check could not establish architecture support for this runtime.",
        "action": "Review the model publisher's runtime requirements.",
        "scope": "Architecture recognition only; encoding, GPU execution, adapters, and inference remain unverified.",
    }


class CompatibilityChecks:
    """App-scoped admission, total deadlines, and shutdown for explicit checks."""

    def __init__(self, *, timeout: float = 25, concurrency: int = 2) -> None:
        self.timeout = timeout
        self._slots = asyncio.Semaphore(concurrency)
        self._active: set[asyncio.Task[Any]] = set()
        self._closed = False

    async def assess(self, cfg: Config, body: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            return {**unknown_assessment(), "detail": "The server is shutting down."}
        task = asyncio.create_task(self._assess(cfg, body))
        self._active.add(task)
        try:
            return await asyncio.wait_for(task, timeout=self.timeout)
        except asyncio.TimeoutError:
            log.info("Compatibility check deadline exceeded (including admission)")
            return {
                **unknown_assessment(),
                "detail": "The compatibility check timed out.",
                "action": "Check connectivity and try again.",
            }
        finally:
            self._active.discard(task)

    async def _assess(self, cfg: Config, body: dict[str, Any]) -> dict[str, Any]:
        async with self._slots:
            return await assess_compatibility(cfg, body)

    async def shutdown(self) -> None:
        self._closed = True
        active = list(self._active)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def assess_compatibility(cfg: Config, body: dict[str, Any]) -> dict[str, Any]:
    """Explicit read-only assessment for one remote option or registered model."""
    answer = unknown_assessment()
    gpu = next((g for g in cfg.gpus if g.enabled), None)
    answer["backend"] = gpu.backend if gpu else None
    identity = await asyncio.to_thread(runtime_identity, cfg.paths.llama_server)
    if identity is None:
        return {
            **answer,
            "detail": "The configured llama-server executable is unavailable.",
            "action": "Install or select a runtime in System, then check again.",
        }
    binary, fingerprint = identity
    answer["runtime"] = Path(binary).name
    answer["runtime_fingerprint"] = fingerprint[:12]
    if "name" in body and cfg.find_model(body["name"]) is None:
        raise ValueError("Unknown registered model")
    phase = "local header" if "name" in body else "Hugging Face metadata"
    try:
        if "name" in body:
            model = cfg.find_model(body["name"])
            if model is None:
                raise ValueError("Unknown registered model")

            def read_header() -> bytes:
                with Path(model.path).expanduser().open("rb") as source:
                    return source.read(_MAX_HEADER)

            header = await asyncio.to_thread(read_header)
            answer["source"] = "Downloaded GGUF header"
        else:
            repo, file = body["repo"], body["file"]
            info = await remote_metadata(repo)
            revision = info.get("sha")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
                return {
                    **answer,
                    "detail": "Hugging Face did not provide an immutable model revision.",
                }
            if file not in {
                item.get("rfilename") for item in info.get("siblings", []) if isinstance(item, dict)
            }:
                raise FileNotInRepositoryError("File is not in this repository")
            phase = "Hugging Face header"
            header = await remote_header(repo, file, revision)
            answer["revision"] = revision
            answer["source"] = "Selected GGUF header on Hugging Face"
        architecture = header_architecture(header)
        if not architecture:
            return {
                **answer,
                "detail": "Architecture metadata was unavailable in the bounded GGUF header read. No compatibility claim can be made.",
            }
        answer["architecture"] = architecture
        phase = "runtime probe"
        outcome = await probe_architecture(binary, fingerprint, architecture)
        if await asyncio.to_thread(runtime_identity, cfg.paths.llama_server) != identity:
            return {
                **answer,
                "detail": "The runtime changed during this check.",
                "action": "Check again using the current runtime.",
            }
        if outcome == "rejected":
            return {
                **answer,
                "status": "incompatible",
                "label": "Requires a different runtime",
                "detail": f"This runtime explicitly rejects the {architecture} architecture.",
                "action": "Update or select a runtime supporting this architecture in System, then check again.",
            }
        if outcome == "recognized":
            return {
                **answer,
                "status": "recognized",
                "label": "Architecture recognized",
                "detail": f"This runtime recognizes {architecture}. This does not establish full model compatibility.",
                "action": "Review encoding and feature requirements; successful inference is still needed to verify this model.",
            }
    except FileNotInRepositoryError:
        raise
    except asyncio.CancelledError:
        log.debug("Compatibility check cancelled during %s", phase)
        raise
    except Exception as exc:  # Optional metadata/loader failures must not break discovery.
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        # Do not record exception messages, URLs, headers, paths, or tokens.
        log.warning(
            "Compatibility check failed: phase=%s cause=%s http_status=%s",
            phase,
            type(exc).__name__,
            status,
        )
        if phase == "runtime probe":
            detail, action = (
                "The runtime probe could not complete.",
                "Review the runtime setup and try again.",
            )
        elif phase == "local header":
            detail, action = (
                "The downloaded model header could not be read.",
                "Check the model file and try again.",
            )
        elif status in {401, 403, 404}:
            detail, action = (
                "Model metadata is unavailable or requires access.",
                "Check model access and try again.",
            )
        else:
            detail, action = (
                "Model metadata could not be read.",
                "Check connectivity and try again.",
            )
        return {**answer, "detail": detail, "action": action}
    return answer


class FileNotInRepositoryError(ValueError):
    """A valid request selected a file absent from the immutable repository."""


def validate_request(body: dict[str, Any]) -> None:
    name = body.get("name")
    if name is not None:
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ValueError("name must be a registered model name")
        return
    repo, file = body.get("repo"), body.get("file")
    if not isinstance(repo, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo
    ):
        raise ValueError("repo must look like owner/name")
    if not isinstance(file, str) or len(file) > 500 or not file.lower().endswith(".gguf"):
        raise ValueError("file must be a GGUF repository path")
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts or "\\" in file or any(ord(c) < 32 for c in file):
        raise ValueError("file must be a relative repository path")
