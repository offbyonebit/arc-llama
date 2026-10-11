"""Bounded, explicit GGUF architecture checks against the installed runtime.

A zero-tensor fixture checks architecture recognition, never successful inference.
Remote reads are pinned to a Hub commit and capped; no weight download is needed.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

import httpx

from arc_llama.config import Config

_MAX_HEADER = 262_144
_ARCH = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_PROBE_LOCK = Lock()
_PROBE_CACHE: dict[tuple[str, str, str], str] = {}


def header_architecture(data: bytes) -> str | None:
    """Read an early architecture field; truncated/invalid metadata stays unknown."""
    offset = 0

    def take(n: int) -> bytes:
        nonlocal offset
        if n < 0 or offset + n > len(data):
            raise ValueError("Metadata outside bounded header")
        result = data[offset:offset + n]
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
            records.append((str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        digest = hashlib.sha256(repr(records).encode()).hexdigest()
        return str(binary), digest
    except (OSError, RuntimeError, ValueError):
        return None


def probe_architecture(binary: str, fingerprint: str, architecture: str) -> str:
    """Return recognized/rejected/unknown, based on exact loader diagnostics."""
    if not _ARCH.fullmatch(architecture):
        return "unknown"
    cache_key = (binary, fingerprint, architecture)
    with _PROBE_LOCK:
        if cache_key in _PROBE_CACHE:
            return _PROBE_CACHE[cache_key]

    def string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    body = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    body += string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    body += b"\0" * (-len(body) % 32)
    with _PROBE_LOCK, tempfile.TemporaryDirectory(prefix="arc-architecture-") as folder:
        fixture = Path(folder) / "architecture.gguf"
        fixture.write_bytes(body)
        try:
            result = subprocess.run(
                [binary, "-m", str(fixture), "-ngl", "0", "--device", "none", "--no-warmup", "--port", "0"],
                capture_output=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode == 0:
        return "unknown"
    if re.search(r"unknown model architecture:\s*['\"]" + re.escape(architecture) + r"['\"]", output):
        outcome = "rejected"
    elif re.search(r"key not found in model:\s*" + re.escape(architecture) + r"\.[a-z_]+", output):
        outcome = "recognized"
    else:
        return "unknown"
    with _PROBE_LOCK:
        if len(_PROBE_CACHE) >= 128:
            _PROBE_CACHE.pop(next(iter(_PROBE_CACHE)))
        _PROBE_CACHE[cache_key] = outcome
    return outcome


def remote_header(repo: str, file: str, revision: str) -> bytes:
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, file, revision=revision)
    data = bytearray()
    with httpx.stream("GET", url, headers={"Range": f"bytes=0-{_MAX_HEADER - 1}"}, follow_redirects=True, timeout=10) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=16_384):
            data.extend(chunk[:_MAX_HEADER - len(data)])
            if len(data) >= _MAX_HEADER:
                break
    return bytes(data)


def assess_compatibility(cfg: Config, body: dict[str, Any]) -> dict[str, Any]:
    """Explicit read-only assessment for one remote option or registered model."""
    from huggingface_hub import HfApi

    answer: dict[str, Any] = {
        "status": "unknown", "label": "Compatibility unknown",
        "detail": "The check could not establish architecture support for this runtime.",
        "action": "Review the model publisher's runtime requirements.",
        "scope": "Architecture recognition only; encoding, GPU execution, adapters, and inference remain unverified.",
    }
    gpu = next((g for g in cfg.gpus if g.enabled), None)
    answer["backend"] = gpu.backend if gpu else None
    identity = runtime_identity(cfg.paths.llama_server)
    if identity is None:
        return {**answer, "detail": "The configured llama-server executable is unavailable.", "action": "Install or select a runtime in System, then check again."}
    binary, fingerprint = identity
    answer["runtime"] = Path(binary).name
    answer["runtime_fingerprint"] = fingerprint[:12]
    try:
        if "name" in body:
            model = cfg.find_model(body["name"])
            if model is None:
                raise ValueError("Unknown registered model")
            with Path(model.path).expanduser().open("rb") as source:
                header = source.read(_MAX_HEADER)
            answer["source"] = "Downloaded GGUF header"
        else:
            repo, file = body["repo"], body["file"]
            info = HfApi().model_info(repo)
            revision = info.sha
            if not revision or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
                return {**answer, "detail": "Hugging Face did not provide an immutable model revision."}
            if file not in {s.rfilename for s in (info.siblings or [])}:
                raise ValueError("File is not in this repository")
            header = remote_header(repo, file, revision)
            answer["revision"] = revision
            answer["source"] = "Selected GGUF header on Hugging Face"
        architecture = header_architecture(header)
        if not architecture:
            return {**answer, "detail": "Architecture metadata was unavailable in the bounded GGUF header read. No compatibility claim can be made."}
        answer["architecture"] = architecture
        outcome = probe_architecture(binary, fingerprint, architecture)
        if runtime_identity(cfg.paths.llama_server) != identity:
            return {**answer, "detail": "The runtime changed during this check.", "action": "Check again using the current runtime."}
        if outcome == "rejected":
            return {**answer, "status": "incompatible", "label": "Requires a different runtime", "detail": f"This runtime explicitly rejects the {architecture} architecture.", "action": "Update or select a runtime supporting this architecture in System, then check again."}
        if outcome == "recognized":
            return {**answer, "status": "recognized", "label": "Architecture recognized", "detail": f"This runtime recognizes {architecture}. This does not establish full model compatibility.", "action": "Review encoding and feature requirements; successful inference is still needed to verify this model."}
    except ValueError:
        raise
    except Exception:  # Network, gated models, unreadable files, or loader failures stay unknown.
        return {**answer, "detail": "Model metadata could not be read. It may be private, gated, offline, or unavailable.", "action": "Check access and connectivity, then try again."}
    return answer


def validate_request(body: dict[str, Any]) -> None:
    name = body.get("name")
    if name is not None:
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ValueError("name must be a registered model name")
        return
    repo, file = body.get("repo"), body.get("file")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("repo must look like owner/name")
    if not isinstance(file, str) or len(file) > 500 or not file.lower().endswith(".gguf"):
        raise ValueError("file must be a GGUF repository path")
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts or "\\" in file or any(ord(c) < 32 for c in file):
        raise ValueError("file must be a relative repository path")
