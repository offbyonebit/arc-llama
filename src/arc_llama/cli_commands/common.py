"""Shared CLI helpers used by multiple command modules."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import click
from rich.console import Console

from arc_llama.arch import Backend
from arc_llama.config import Config, load_config

console = Console()

_IS_WINDOWS = sys.platform == "win32"



class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


def setup_logging(verbose: bool) -> None:
    """Configure root logging according to the verbose flag and env."""
    level = logging.DEBUG if verbose else logging.INFO
    if os.environ.get("ARC_LLAMA_LOG_JSON"):
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger()
        root.setLevel(level)
        root.handlers.clear()
        root.addHandler(handler)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def save_or_die(cfg: Config, path: Path) -> None:
    try:
        cfg.save(path)
    except OSError as e:
        console.print(f"[red]failed to write config to {path}: {e}[/red]")
        sys.exit(1)


def resolve_llama_server(explicit: str | None) -> str:
    """Find a usable llama-server binary, in order of preference.

    If the user explicitly passed a path, preserve it even when it does not
    exist so the caller can report the exact location in an error.
    """
    if explicit:
        return explicit
    candidates = [
        os.environ.get("ARC_LLAMA_SERVER", ""),
        shutil.which("llama-server") or "",
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return "llama-server"  # leave as-is; PATH at runtime may resolve


def slugify_for_name(parent: str, file: str) -> str:
    base = parent.lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if not base:
        base = "model"
    m = re.search(r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|UD-[A-Z0-9_]+)", file, re.IGNORECASE)
    if m:
        base = f"{base}-{m.group(1).lower()}"
    return base


def experimental_agent_enabled() -> bool:
    """Return True if the experimental coding-agent commands should be exposed."""
    return os.environ.get("ARC_LLAMA_EXPERIMENTAL_AGENT", "").lower() in ("1", "true", "yes")


def server_url_from_ctx(ctx: click.Context, server_url: str | None) -> str:
    if server_url:
        return server_url.rstrip("/")
    cfg = load_config(ctx.obj["config_path"])
    return f"http://{cfg.server.host}:{cfg.server.port}"


def state_dir_from_config(cfg: Config) -> Path | None:
    if cfg.paths.state_dir:
        return Path(cfg.paths.state_dir).expanduser()
    return None


def print_gpu_table(gpus) -> None:
    from rich.table import Table

    table = Table(title="Intel GPUs")
    table.add_column("PCI slot")
    table.add_column("Arch")
    table.add_column("Name")
    table.add_column("Driver")
    table.add_column("DRM card")
    table.add_column("VRAM")
    for g in gpus:
        vram = f"{g.vram_gb} GB" if g.vram_gb else "?"
        table.add_row(
            g.pci_slot,
            g.arch.value,
            g.name,
            g.driver or "—",
            g.drm_card or "—",
            vram,
        )
    console.print(table)


def gather_workload_profile(
    cfg: Config,
    context: str | None,
    style: str | None,
    priority: str | None,
) -> None:
    """Record the three workload answers into cfg.workload.

    Flags always win and make Docker/CI non-blocking. When a flag is absent
    and stdin is interactive, ask — every question offers "not-sure", which
    keeps the default (empty = unprofiled, tuner behaves as before). Nothing
    here asks about ubatch, KV type, or flags directly; the answers steer the
    tuner indirectly via the [workload] section.
    """

    def _norm(v: str | None) -> str:
        return "" if v is None or v == "not-sure" else v

    if sys.stdin.isatty():
        if context is None:
            context = click.prompt(
                "Typical conversation length? (short <8k / long ~32k / very_long 100k+)",
                type=click.Choice(["short", "long", "very_long", "not-sure"]),
                default="not-sure",
            )
        if style is None:
            style = click.prompt(
                "Mostly agentic tool-calling loops, or mostly conversational chat?",
                type=click.Choice(["agentic", "conversational", "not-sure"]),
                default="not-sure",
            )
        if priority is None:
            priority = click.prompt(
                "What hurts more: waiting for the first token, or the speed after it starts?",
                type=click.Choice(["first_token", "throughput", "not-sure"]),
                default="not-sure",
            )
    cfg.workload.context_length = _norm(context)
    cfg.workload.style = _norm(style)
    cfg.workload.priority = _norm(priority)


def print_serve_banner(cfg: Config) -> None:
    """Print the applied Arc profile per GPU + model at serve startup."""
    from arc_llama.arch import Arch, aot_arch_for
    from arc_llama.detect import detect_gpus

    console.print("[bold]arc-llama serve[/bold] — applied Arc profiles:")
    live: dict[str, object] = {}
    if not _IS_WINDOWS:
        try:
            live = {g.pci_slot: g for g in detect_gpus(enrich=False)}
        except Exception:
            live = {}
    for gpu in cfg.gpus:
        if not gpu.enabled:
            continue
        backend = gpu.backend or Backend.SYCL.value
        vram = f"{gpu.vram_mb} MB" if gpu.vram_mb else "VRAM unknown"
        parts = [f"GPU {gpu.pci_slot}: {gpu.arch} ({backend}) · {vram}"]
        det = live.get(gpu.pci_slot)
        from arc_llama.platform_checks import rebar_likely_enabled

        if det is not None and getattr(det, "sysfs_path", ""):
            r = rebar_likely_enabled(det.sysfs_path, det.vram_mb)
            parts.append("ReBAR on" if r else ("ReBAR OFF" if r is False else "ReBAR ?"))
        if backend == Backend.SYCL.value:
            aot = aot_arch_for(det.device_id) if det is not None else None
            if aot is None:
                try:
                    a = Arch(gpu.arch) if gpu.arch else None
                except ValueError:
                    a = None
                if a == Arch.BATTLEMAGE:
                    aot = "bmg-g21"
                elif a == Arch.ALCHEMIST:
                    aot = "acm-g10"
            if aot is not None:
                parts.append(
                    f"JIT cold-start ~20s (AOT: -DGGML_SYCL_DEVICE_ARCH={aot})"
                )
            else:
                parts.append("JIT cold-start")
        console.print("  " + " · ".join(parts))
    for m in cfg.models:
        r = m.recipe or {}
        ctx = r.get("ctx", "?")
        kv_k = r.get("cache_type_k", "f16")
        kv_v = r.get("cache_type_v", "f16")
        kv_txt = kv_k if kv_k == kv_v else f"{kv_k}/{kv_v}"
        console.print(f"  model {m.name}: ctx={ctx} · KV {kv_txt} · port {m.port}")
    if not cfg.models:
        console.print("  [dim]no models registered — `arc-llama add` something first[/dim]")


def print_autotune_banner(cfg: Config) -> None:
    """Tell the operator whether background tuning is active and how to disable it."""
    if not getattr(cfg, "tune", None) or not cfg.tune.auto:
        console.print("[dim]Auto-tune: off (use --auto-tune or set [tune] auto=true to enable).[/dim]")
        return
    untuned_count = sum(1 for m in cfg.models if m.tune_state in ("untuned", "skipped"))
    if untuned_count:
        console.print(
            f"[dim]Auto-tune: on -- {untuned_count} model(s) eligible; "
            f"sweep starts after {cfg.tune.idle_seconds}s idle. "
            f"Use --no-auto-tune to disable.[/dim]"
        )
    else:
        console.print("[dim]Auto-tune: on -- no eligible models.[/dim]")


def print_tune_status_table(cfg: Config) -> None:
    """Print the per-model tune state for `arc-llama tune --status`."""
    from datetime import datetime, timezone

    from rich.table import Table

    table = Table(title="Tune status")
    table.add_column("Model")
    table.add_column("State")
    table.add_column("Tuned at")
    table.add_column("Fingerprint")
    table.add_column("Error")
    for m in cfg.models:
        tuned_at = ""
        if m.tuned_at:
            tuned_at = datetime.fromtimestamp(m.tuned_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            m.name,
            m.tune_state,
            tuned_at,
            (m.tune_fingerprint[:16] + "...") if m.tune_fingerprint else "",
            m.tune_error[:60],
        )
    console.print(table)


__all__ = [
    "console",
    "_IS_WINDOWS",
    "setup_logging",
    "save_or_die",
    "resolve_llama_server",
    "slugify_for_name",
    "experimental_agent_enabled",
    "server_url_from_ctx",
    "state_dir_from_config",
    "print_gpu_table",
    "gather_workload_profile",
    "print_serve_banner",
    "print_autotune_banner",
    "print_tune_status_table",
]
