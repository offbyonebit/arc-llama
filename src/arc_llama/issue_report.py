"""Build prefilled GitHub problem-report links.

Arc Llama has no server of its own to receive reports. Instead, this module
builds a GitHub URL with the user's environment already filled in. The user
opens it in their browser, reviews every field, and submits it under their own
GitHub account. Nothing is sent anywhere until they press GitHub's own button,
and no token is ever embedded.

Two destinations are supported:

* ``issue``: the ``bug-report.yml`` issue form. GitHub prefills form fields
  whose ``id`` matches a query parameter.
* ``discussion``: a new discussion in the Q&A category. This only works once
  Discussions is enabled for the repository.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from arc_llama import __version__
from arc_llama.support_bundle import _safe_text

REPO_URL = "https://github.com/offbyonebit/arc-llama"
ISSUE_TEMPLATE = "bug-report.yml"
DISCUSSION_CATEGORY = "q-a"

# Browsers and GitHub accept longer URLs, but some proxies and GitHub's own
# redirect through sign-in start failing past roughly 8 KB. Stay below that.
MAX_URL_LENGTH = 7_500
DEFAULT_LOG_LINES = 40
MIN_LOG_LINES = 5

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_ARC_KEY = re.compile(r"\b(?:sk|arc)[-_][A-Za-z0-9]{20,}\b")
_ASSIGNMENT = re.compile(
    r"(?i)\b(admin_token|hf_token|token|password|secret|api[_-]?key)(\s*[=:]\s*)\S+"
)


def redact(text: str) -> str:
    """Remove home paths, tokens, and credential-shaped values from text."""
    home = str(Path.home())
    if "\\" in home:
        # Windows errors and JSON logs often show paths with doubled backslashes.
        text = re.sub(re.escape(home.replace("\\", "\\\\")), "$HOME", text, flags=re.IGNORECASE)
    text = _safe_text(text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _HF_TOKEN.sub("[REDACTED]", text)
    text = _ARC_KEY.sub("[REDACTED]", text)
    return _ASSIGNMENT.sub(r"\1\2[REDACTED]", text)


@dataclass
class ReportContext:
    """Everything a report can carry. Each field is plain, redacted text."""

    version: str = __version__
    python: str = field(default_factory=lambda: sys.version.split()[0])
    os_name: str = field(default_factory=platform.platform)
    gpus: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    model_count: int | None = None
    log_name: str = ""
    log_lines: list[str] = field(default_factory=list)
    hardware: list[str] = field(default_factory=list)

    def environment_text(self) -> str:
        lines = [
            f"Arc Llama: {self.version}",
            f"OS: {self.os_name}",
            f"Python: {self.python}",
            f"Backend: {', '.join(self.backends) if self.backends else 'not configured'}",
        ]
        if self.model_count is not None:
            lines.append(f"Registered models: {self.model_count}")
        return redact("\n".join(lines + self.hardware))

    def gpu_text(self) -> str:
        return redact("\n".join(self.gpus)) if self.gpus else "No Intel GPU detected."

    def log_text(self, max_lines: int | None = None) -> str:
        if max_lines == 0 and self.log_lines:
            return (
                "Log left out to keep the link short. Run `arc-llama support-bundle` and attach it."
            )
        lines = self.log_lines if max_lines is None else self.log_lines[-max_lines:]
        if not lines:
            return "No llama-server log found."
        header = f"Last {len(lines)} lines of {self.log_name or 'the newest log'}:"
        return redact(header + "\n" + "\n".join(lines))


def gpu_lines_from_detected(gpus: Iterable[Any]) -> list[str]:
    """Describe GPUs found by :func:`arc_llama.detect.detect_gpus`."""
    lines = []
    for gpu in gpus:
        vram = f"{gpu.vram_mb} MiB" if getattr(gpu, "vram_mb", None) else "VRAM unknown"
        arch = getattr(getattr(gpu, "arch", None), "value", "unknown")
        lines.append(
            f"{gpu.name} (0x{gpu.device_id:04x}, {arch}, {vram}, driver {gpu.driver or 'unknown'})"
        )
    return lines


def gpu_lines_from_config(gpus: Iterable[Any]) -> list[str]:
    """Describe GPUs recorded in the config, for use inside a running server."""
    lines = []
    for gpu in gpus:
        vram = f"{gpu.vram_mb} MiB" if gpu.vram_mb else "VRAM unknown"
        state = "" if gpu.enabled else ", disabled"
        lines.append(f"{gpu.name or 'Intel GPU'} ({gpu.arch}, {vram}, {gpu.backend}{state})")
    return lines


def hardware_lines(cfg: Any | None) -> list[str]:
    """Runtime pin, driver stack, RAM and the device llama-server would get.

    Runs vulkaninfo and llama-server --list-devices, so it is for the CLI only.
    """
    from arc_llama.autotune import driver_stack_key
    from arc_llama.platform_checks import mesa_version
    from arc_llama.runtime_pin import LLAMA_CPP_BUILD

    lines = [f"Pinned llama.cpp: {LLAMA_CPP_BUILD}"]
    if sys.platform != "win32":
        mesa = mesa_version()
        lines.append(f"Mesa: {'.'.join(map(str, mesa)) if mesa else 'unknown'}")
    lines.append(f"Driver stack: {driver_stack_key().replace('|', ', ')}")
    ram = _total_ram_mib()
    lines.append(f"RAM: {ram} MiB" if ram else "RAM: unknown")
    try:
        lines += _device_lines(cfg)
    except Exception as exc:  # noqa: BLE001 - the rest of the report still helps
        lines.append(f"Chosen device: could not ask llama-server ({type(exc).__name__})")
    return lines


def _total_ram_mib() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "avail", "page", "page_avail", "virt", "virt_avail", "ext")
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.total) // 2**20
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
    except (OSError, ValueError, AttributeError):
        return None


def _device_lines(cfg: Any | None) -> list[str]:
    from arc_llama.arch import Arch, profile_for
    from arc_llama.binary import list_devices
    from arc_llama.launcher import build_env, vulkan_device_arg
    from arc_llama.onboarding import runtime_path

    gpu = next((g for g in cfg.gpus if g.enabled), None) if cfg is not None else None
    if cfg is None or gpu is None:
        return ["Chosen device: no enabled GPU in the config"]
    binary = runtime_path(cfg)
    if binary is None:
        return [f"Chosen device: none yet, no llama.cpp runtime installed ({gpu.backend})"]
    if gpu.backend == "vulkan":
        devices = list_devices(binary, "Vulkan")
        device = vulkan_device_arg([gpu], binary)
        if gpu.vulkan_index is not None:
            why = "vulkan_index in the config"
        elif device:
            why = "matched to the GPU by name or memory size"
        else:
            why = "no --device passed, so llama.cpp picks its default"
        chosen = device or "llama.cpp default"
        prefix = "Vulkan"
    else:
        arch = Arch(gpu.arch) if gpu.arch in {a.value for a in Arch} else Arch.UNKNOWN
        env = build_env(
            profile_for(arch), gpu, llama_server=binary, oneapi_setvars=getattr(cfg.paths, "oneapi_setvars", None)
        )
        devices = list_devices(binary, "SYCL", env=env)
        chosen = env.get("ONEAPI_DEVICE_SELECTOR", "unset")
        same = chosen == f"level_zero:{gpu.sycl_index}"
        why = f"sycl_index {gpu.sycl_index} in the config" if same else "matched to the GPU by name"
        prefix = "SYCL"
    listed = "; ".join(f"{prefix}{i} {name}" + (f" ({mib} MiB)" if mib else "") for i, name, mib in devices)
    return [
        f"llama.cpp {prefix} devices: {listed or 'none listed'}",
        f"Chosen device: {chosen} for {gpu.name or 'the configured GPU'} ({why})",
    ]


def newest_log(log_dir: Path | None, lines: int = DEFAULT_LOG_LINES) -> tuple[str, list[str]]:
    """Return the newest ``*.log`` name in *log_dir* and its last *lines* lines."""
    if log_dir is None:
        return "", []
    try:
        logs = [p for p in Path(log_dir).expanduser().glob("*.log") if p.is_file()]
    except OSError:
        return "", []
    if not logs:
        return "", []
    newest = max(logs, key=lambda p: p.stat().st_mtime)
    try:
        text = newest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return newest.name, []
    return newest.name, text.splitlines()[-lines:]


def collect_context(
    cfg: Any | None,
    *,
    detected_gpus: Iterable[Any] | None = None,
    log_dir: Path | None = None,
) -> ReportContext:
    """Gather a :class:`ReportContext` without running inference or subprocesses."""
    ctx = ReportContext()
    if detected_gpus is not None:
        ctx.gpus = gpu_lines_from_detected(detected_gpus)
    elif cfg is not None:
        ctx.gpus = gpu_lines_from_config(cfg.gpus)
    if cfg is not None:
        ctx.backends = sorted({g.backend for g in cfg.gpus if g.enabled})
        ctx.model_count = len(cfg.models)
        if log_dir is None and getattr(cfg.paths, "state_dir", None):
            log_dir = Path(cfg.paths.state_dir)
    ctx.log_name, ctx.log_lines = newest_log(log_dir)
    return ctx


@dataclass
class ReportLink:
    url: str
    kind: str
    log_lines_included: int
    trimmed: bool


def _issue_url(ctx: ReportContext, title: str, description: str, log_lines: int) -> str:
    params = {
        "template": ISSUE_TEMPLATE,
        "title": title or "[Bug] ",
        "environment": ctx.environment_text(),
        "gpu": ctx.gpu_text(),
        "logs": ctx.log_text(log_lines),
    }
    if description:
        params["what-happened"] = description
    return f"{REPO_URL}/issues/new?{urlencode(params)}"


def _discussion_body(ctx: ReportContext, description: str, log_lines: int) -> str:
    return "\n\n".join(
        (
            "### What happened\n\n"
            + (description or "_Describe what you tried and what went wrong._"),
            "### Environment\n\n```text\n" + ctx.environment_text() + "\n```",
            "### GPU\n\n```text\n" + ctx.gpu_text() + "\n```",
            "### Log tail\n\n```text\n" + ctx.log_text(log_lines) + "\n```",
        )
    )


def _discussion_url(ctx: ReportContext, title: str, description: str, log_lines: int) -> str:
    params = {
        "category": DISCUSSION_CATEGORY,
        "title": title or "Problem: ",
        "body": _discussion_body(ctx, description, log_lines),
    }
    return f"{REPO_URL}/discussions/new?{urlencode(params)}"


def build_report_link(
    ctx: ReportContext,
    *,
    kind: str = "issue",
    title: str = "",
    description: str = "",
    max_length: int = MAX_URL_LENGTH,
) -> ReportLink:
    """Build the prefilled link, dropping old log lines until it fits."""
    if kind not in {"issue", "discussion"}:
        raise ValueError("kind must be 'issue' or 'discussion'")
    builder = _issue_url if kind == "issue" else _discussion_url
    title = redact(title)[:200]
    description = redact(description)[:1500]
    available = len(ctx.log_lines)
    lines = available
    url = builder(ctx, title, description, lines)
    # Shrink the log first; it is the least important part to keep whole.
    while len(url) > max_length and lines > MIN_LOG_LINES:
        lines = max(MIN_LOG_LINES, lines // 2)
        url = builder(ctx, title, description, lines)
    if len(url) > max_length:
        lines = 0
        url = builder(ctx, title, description[:300], lines)
    return ReportLink(
        url=url,
        kind=kind,
        log_lines_included=min(lines, available),
        trimmed=lines < available,
    )
