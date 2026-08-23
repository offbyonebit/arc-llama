"""arc-llama CLI.

Top-level commands:

  arc-llama init       Auto-detect GPUs and write an initial config.
  arc-llama doctor     Diagnose the local environment (drivers, oneAPI, perms).
  arc-llama list       List registered models and their state.
  arc-llama gpus       List detected Intel GPUs.
  arc-llama add        Register a model — local file or HF download.
  arc-llama remove     Remove a model from the config.
  arc-llama serve      Run the OpenAI-compatible router.
  arc-llama benchmark  Measure prompt-eval / generation tok/s for a model.
  arc-llama tune       Staged greedy autotune; persist the winning recipe.
  arc-llama tui        Launch the server management TUI.
  arc-llama systemd    Print a systemd --user service unit for `arc-llama serve`.

The agent/coding-assistant commands (`agent`, `code`, `agent-tui`) are
experimental and only appear when ARC_LLAMA_EXPERIMENTAL_AGENT=1 is set.

A small static web UI is bundled and served at `/` on the same port as
`arc-llama serve` — open it in a browser for a model-picker + load/stop view.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.table import Table

from arc_llama import __version__
from arc_llama import benchmark as benchmark_mod
from arc_llama.agent import run_agent
from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.interactive import InteractiveAgent
from arc_llama.agent.mcp_client import MCPClientManager
from arc_llama.arch import Arch, Backend, aot_arch_for
from arc_llama.binary import detect_backends, detect_llama_server_backend
from arc_llama.chat_store import ChatMessage, ChatStore
from arc_llama.config import (
    AUDIO_ENGINE_LLAMACPP,
    Config,
    default_config_path,
    init_config_from_detection,
    load_config,
)
from arc_llama.detect import DetectedGPU, detect_gpus, lspci_intel_gpus
from arc_llama.launcher import resolve_binary
from arc_llama.models import (
    add_audio_model,
    add_local_model,
    add_voice,
    discover_ggufs,
    download_asr_from_hf,
    download_from_hf,
    parse_hf_spec,
    register_discovered,
)
from arc_llama.platform_checks import (
    DoctorReport,
    format_bytes,
    kernel_module_loaded,
    level_zero_loader_present,
    max_memory_bar_bytes,
    oneapi_setvars_path,
    parse_kernel_version,
    rebar_likely_enabled,
    user_in_groups,
)
from arc_llama.server_caps import probe_server_caps
from arc_llama.skills import load_skills
from arc_llama.tts import TTS_ENGINE_OMNIVOICE
from arc_llama.tts import engine_names as tts_engine_names
from arc_llama.tts import engines as tts_engines
from arc_llama.tts import get_engine as get_tts_engine

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


def _setup_logging(verbose: bool) -> None:
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


def _save_or_die(cfg: Config, path: Path) -> None:
    try:
        cfg.save(path)
    except OSError as e:
        console.print(f"[red]failed to write config to {path}: {e}[/red]")
        sys.exit(1)


def _resolve_llama_server(explicit: str | None) -> str:
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


# ===========================================================================
# Top-level group
# ===========================================================================


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to config.toml (default: $XDG_CONFIG_HOME/arc-llama/config.toml).",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: Path | None) -> None:
    """Plug-and-play llama.cpp runtime for Intel Arc GPUs."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path or default_config_path()


# ===========================================================================
# init
# ===========================================================================


def _gather_workload_profile(
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
        # Spelled out rather than `v in (None, "not-sure")` so the type
        # checker can narrow v to str on the fallthrough.
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


@cli.command()
@click.option(
    "--llama-server",
    type=click.Path(),
    default=None,
    help="Path to your built llama-server binary (SYCL or Vulkan backend).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
@click.option(
    "--scan/--no-scan",
    default=True,
    help="After init, walk scan paths for .gguf files and auto-register them (default: on).",
)
@click.option(
    "--scan-path",
    "scan_paths",
    multiple=True,
    type=click.Path(),
    help="Extra directory to walk for GGUFs. Repeatable.",
)
@click.option(
    "--workload-context",
    type=click.Choice(["short", "long", "very_long", "not-sure"]),
    default=None,
    help="Typical conversation length: short (<8k), long (~32k), very_long (100k+). "
    "'not-sure' keeps the default.",
)
@click.option(
    "--workload-style",
    type=click.Choice(["agentic", "conversational", "not-sure"]),
    default=None,
    help="Mostly agentic tool-calling loops, or mostly conversational chat. "
    "'not-sure' keeps the default.",
)
@click.option(
    "--workload-priority",
    type=click.Choice(["first_token", "throughput", "not-sure"]),
    default=None,
    help="What hurts more: waiting for the first token, or the speed after it "
    "starts. 'not-sure' keeps the default.",
)
@click.pass_context
def init(
    ctx: click.Context,
    llama_server: str | None,
    force: bool,
    scan: bool,
    scan_paths: tuple[str, ...],
    workload_context: str | None,
    workload_style: str | None,
    workload_priority: str | None,
) -> None:
    """Detect GPUs and write a starter config; auto-register any GGUFs found."""
    config_path: Path = ctx.obj["config_path"]
    if config_path.exists() and not force:
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("Use --force to overwrite, or edit it directly.")
        sys.exit(1)
    gpus = detect_gpus()
    if not gpus:
        if _IS_WINDOWS:
            console.print(
                "[yellow]No Intel GPUs detected — Windows auto-detection is not "
                "supported yet. Create a config manually or run this on WSL.[/yellow]"
            )
        else:
            console.print("[red]No Intel GPUs detected.[/red]")
            console.print("Run [bold]arc-llama doctor[/bold] for a diagnosis.")
        sys.exit(2)
    server_path = _resolve_llama_server(llama_server)
    server_bin = Path(server_path).expanduser()
    runtime_missing = not server_bin.exists()
    if runtime_missing and llama_server is not None:
        # An explicit --llama-server path that does not exist is a mistake to surface.
        console.print(f"[red]llama-server binary not found: {server_path}[/red]")
        sys.exit(3)

    bin_backend = None
    if runtime_missing:
        # No binary yet: still write the config (GPUs are detected) and point the
        # user at install-runtime, which fills in paths.llama_server for them.
        console.print("[yellow]No llama-server binary found yet.[/yellow]")
        console.print(
            "[dim]Run [bold]arc-llama install-runtime[/bold] to download a portable "
            "Vulkan build (no oneAPI needed), then [bold]arc-llama serve[/bold].[/dim]"
        )
    else:
        bin_backend = detect_llama_server_backend(server_bin)
        if bin_backend is None:
            console.print(
                f"[yellow]Could not determine backend of {server_bin}; "
                f"ensure it supports the GPUs you configured.[/yellow]"
            )
        else:
            console.print(f"[dim]Detected llama-server backend: {bin_backend.value}[/dim]")

    cfg = init_config_from_detection(
        gpus, llama_server_path=None if runtime_missing else server_path
    )
    # init_config_from_detection defaults every GPU to SYCL; align to the binary
    # we actually have so `serve` applies the right backend env.
    if bin_backend is not None:
        for gpu_cfg in cfg.gpus:
            gpu_cfg.backend = bin_backend.value
    if scan_paths:
        cfg.paths.scan_paths = list(scan_paths)
    _gather_workload_profile(cfg, workload_context, workload_style, workload_priority)
    _save_or_die(cfg, config_path)
    console.print(f"[green]Wrote config to {config_path}[/green]")
    _print_gpu_table(gpus)
    if scan:
        added = _do_scan(cfg, [Path(p) for p in scan_paths])
        if added:
            _save_or_die(cfg, config_path)
            console.print(
                f"[green]Auto-registered {len(added)} model(s):[/green] "
                + ", ".join(m.name for m in added)
            )
        else:
            console.print(
                "[dim]No GGUFs found in scan paths. "
                "Drop one in `paths.models_dir` or pass --scan-path next time.[/dim]"
            )


# ===========================================================================
# doctor
# ===========================================================================


def _doctor_marker(ok: bool | None, severity: str = "info") -> str:
    if ok is True:
        return "[green]ok[/green]"
    if ok is None:
        return "[dim]?[/dim]"
    if severity == "fail":
        return "[red]FAIL[/red]"
    return "[yellow]warn[/yellow]"


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Diagnose the local environment for competitive Arc inference."""
    config_path: Path = ctx.obj["config_path"]
    console.print("[bold]arc-llama doctor[/bold]\n")
    report = DoctorReport()

    cfg: Config | None = None
    if config_path.exists():
        try:
            cfg = load_config(config_path)
        except Exception:
            console.print("[yellow]  warning: could not read config[/yellow]")

    # Kernel + driver (Linux-only diagnostics)
    if _IS_WINDOWS:
        console.print(f"  platform:      Windows {platform.release()}")
        console.print("  [dim]Kernel/driver checks are not available on Windows.[/dim]")
    else:
        uname = platform.uname()
        console.print(f"  kernel:        {uname.release}")
        kv = parse_kernel_version(uname.release)
        has_xe = kernel_module_loaded("xe")
        has_i915 = kernel_module_loaded("i915")
        console.print(f"  xe driver:     {'loaded' if has_xe else 'not loaded'}")
        console.print(f"  i915 driver:   {'loaded' if has_i915 else 'not loaded'}")
        if not has_xe and not has_i915:
            report.add(
                "gpu_driver",
                False,
                "neither xe nor i915 module loaded",
                severity="fail",
                hint="Install/load the Intel GPU kernel driver for your generation.",
            )
        else:
            report.add(
                "gpu_driver",
                True,
                f"xe={'yes' if has_xe else 'no'} i915={'yes' if has_i915 else 'no'}",
            )

    # GPU detection (enrich=True so clinfo populates VRAM where xe doesn't via sysfs)
    gpus = detect_gpus(enrich=True)
    if gpus:
        console.print(f"\n  detected {len(gpus)} Intel GPU(s):")
        for g in gpus:
            vram = f"{g.vram_gb} GB" if g.vram_gb else "VRAM unknown"
            console.print(
                f"    - {g.name} @ {g.pci_slot}  ({g.arch.value}, driver={g.driver or '—'}, {vram})"
            )
            if g.notes:
                for n in g.notes:
                    console.print(f"        note: {n}")
            # ReBAR aperture — critical for Arc llama.cpp performance
            if not _IS_WINDOWS and g.sysfs_path:
                bar = max_memory_bar_bytes(g.sysfs_path)
                rebar = rebar_likely_enabled(g.sysfs_path, g.vram_mb)
                bar_txt = format_bytes(bar) if bar is not None else "unknown"
                if rebar is True:
                    marker = _doctor_marker(True)
                    console.print(f"        ReBAR:   {marker}  largest BAR {bar_txt}")
                    report.add("rebar", True, f"{g.pci_slot} BAR {bar_txt}")
                elif rebar is False:
                    marker = _doctor_marker(False, "fail")
                    console.print(
                        f"        ReBAR:   {marker}  largest BAR {bar_txt} "
                        f"(need BIOS Resizable BAR / Above 4G Decoding)"
                    )
                    report.add(
                        "rebar",
                        False,
                        f"{g.pci_slot} BAR {bar_txt} — ReBAR looks off",
                        severity="fail",
                        hint="Enable Resizable BAR / Above 4G Decoding in BIOS. "
                        "Without it llama.cpp falls back to slow paths on Arc.",
                    )
                else:
                    console.print(f"        ReBAR:   {_doctor_marker(None)}  largest BAR {bar_txt}")
                    report.add("rebar", None, f"{g.pci_slot} BAR {bar_txt}")
            # Battlemage wants a recent kernel
            if g.arch == Arch.BATTLEMAGE and not _IS_WINDOWS:
                kv = parse_kernel_version()
                if kv is not None and (kv[0], kv[1]) < (6, 14):
                    report.add(
                        "kernel_bmg",
                        False,
                        f"kernel {kv[0]}.{kv[1]} < 6.14 for Battlemage",
                        severity="warn",
                        hint="Kernel 6.14+ recommended for stable xe on Battlemage.",
                    )
            # AOT build guidance — eliminates the ~20s SYCL JIT cold start
            # that every model swap pays on Battlemage (where the JIT cache is
            # disabled to dodge a SIGSEGV). Compute the ocloc -device string
            # from the user's actual detected device ID rather than a static
            # hint, so an A770 owner sees `acm-g10` and a B580 owner sees
            # `bmg-g21`.
            aot = aot_arch_for(g.device_id)
            if aot is not None:
                console.print(
                    f"        AOT:      [dim]build with "
                    f"-DGGML_SYCL_DEVICE_ARCH={aot} "
                    f"(ocloc -device {aot}) to skip the ~20s JIT cold start[/dim]"
                )
        report.add("gpus", True, f"{len(gpus)} Intel GPU(s)")
    else:
        console.print("\n  [red]no Intel GPUs detected via sysfs[/red]")
        report.add("gpus", False, "no Intel GPUs via sysfs", severity="fail")
        raw = lspci_intel_gpus()
        if raw:
            console.print("\n  raw lspci output for Intel display devices:")
            for line in raw.splitlines():
                console.print(f"    {line}")
        else:
            console.print("    lspci shows no Intel display devices either.")

    # External tools
    console.print("\n  external tools:")
    for tool in ("clinfo", "sycl-ls", "vulkaninfo", "intel_gpu_top", "nvtop", "lspci"):
        path = shutil.which(tool)
        console.print(f"    {tool:<14} {path or '— missing —'}")

    # Level Zero loader (required for SYCL)
    console.print("\n  Level Zero:")
    lz_ok, lz_path = level_zero_loader_present()
    if lz_ok:
        console.print(f"    loader:      {_doctor_marker(True)}  {lz_path}")
        report.add("level_zero", True, lz_path)
    else:
        console.print(
            f"    loader:      {_doctor_marker(False, 'warn')}  not found "
            f"(install intel-level-zero-gpu / compute-runtime for SYCL)"
        )
        report.add(
            "level_zero",
            False,
            "Level Zero loader not found",
            severity="warn",
            hint="Install intel-level-zero-gpu / intel-compute-runtime packages.",
        )

    # Permissions (Linux-only)
    if _IS_WINDOWS:
        console.print("\n  user groups:")
        console.print("    [dim]Group checks are not available on Windows.[/dim]")
    else:
        console.print("\n  user groups:")
        membership = user_in_groups("render", "video")
        for needed, ok in membership.items():
            console.print(f"    {needed:<14} {_doctor_marker(ok, 'warn' if not ok else 'info')}")
            report.add(
                f"group_{needed}",
                ok,
                needed,
                severity="warn" if not ok else "info",
                hint=("sudo usermod -aG render,video $USER && re-login" if not ok else ""),
            )
        if not all(membership.values()):
            console.print(
                "    [yellow]→ add yourself with `sudo usermod -aG render,video $USER` "
                "and re-login.[/yellow]"
            )

    # oneAPI
    console.print("\n  oneAPI:")
    setvars = oneapi_setvars_path()
    if setvars is not None:
        console.print(f"    setvars:     {_doctor_marker(True)}  {setvars}")
        report.add("oneapi_setvars", True, str(setvars))
    else:
        console.print(
            f"    setvars:     {_doctor_marker(False, 'warn')}  not found — "
            f"install Intel oneAPI Base Toolkit if building llama.cpp from source"
        )
        report.add(
            "oneapi_setvars",
            False,
            "setvars not found",
            severity="warn",
            hint="Install Intel oneAPI Base Toolkit to build a SYCL llama-server.",
        )

    # llama-server binary
    console.print("\n  llama-server binary:")
    if cfg is not None:
        llama_server = Path(cfg.paths.llama_server).expanduser()
        if llama_server.exists():
            backends = detect_backends(llama_server)
            primary = detect_llama_server_backend(llama_server)
            backend_list = ", ".join(sorted(b.value for b in backends)) if backends else "unknown"
            console.print(f"    path:        {llama_server}")
            console.print(f"    backends:    {backend_list}")
            if primary is None:
                report.add(
                    "llama_server",
                    False,
                    "binary present but no SYCL/Vulkan markers",
                    severity="fail",
                    hint="Rebuild llama-server with GGML_SYCL=ON (or Vulkan).",
                )
            else:
                report.add("llama_server", True, backend_list)
            if cfg.gpus:
                for gpu_cfg in cfg.gpus:
                    if not gpu_cfg.enabled:
                        continue
                    want = gpu_cfg.backend
                    have = {b.value for b in backends}
                    if have and want not in have:
                        console.print(
                            f"    [yellow]→ GPU {gpu_cfg.pci_slot} wants "
                            f"'{want}' but binary has [{backend_list}].[/yellow]"
                        )
                        report.add(
                            f"backend_match_{gpu_cfg.pci_slot}",
                            False,
                            f"config={want} binary=[{backend_list}]",
                            severity="warn",
                        )
        else:
            console.print(f"    [yellow]not found[/yellow] at {llama_server}")
            console.print(
                "    [yellow]→ run [bold]arc-llama install-runtime[/bold] to download "
                "a portable Vulkan build.[/yellow]"
            )
            report.add(
                "llama_server",
                False,
                f"missing at {llama_server}",
                severity="fail",
                hint="Run arc-llama install-runtime, or point paths.llama_server at a build.",
            )
    else:
        console.print("    [dim]no config loaded[/dim]")

    # Config
    console.print("\n  config:")
    if cfg is not None:
        console.print(f"    [green]found[/green] at {config_path}")
        if cfg.gpus:
            console.print("    GPUs in config:")
            for gpu_cfg in cfg.gpus:
                status = "enabled" if gpu_cfg.enabled else "disabled"
                console.print(
                    f"      - {gpu_cfg.pci_slot}  {gpu_cfg.name or gpu_cfg.arch}  "
                    f"backend={gpu_cfg.backend}  [{status}]"
                )
    else:
        console.print(
            f"    [yellow]missing[/yellow] at {config_path} — run [bold]arc-llama init[/bold]."
        )
        report.add(
            "config",
            False,
            f"missing at {config_path}",
            severity="warn",
            hint="Run arc-llama init.",
        )

    # Audio backends, only when the user has actually asked for audio.
    if cfg is not None and cfg.audio_models:
        console.print("\n  [bold]audio[/bold]")

        if any(m.task == "asr" for m in cfg.audio_models):
            caps = probe_server_caps(cfg.paths.llama_server)
            if caps.probed and not caps.supports_mmproj:
                console.print(
                    f"    [red]{Path(cfg.paths.llama_server).name} has no --mmproj[/red] "
                    "(built without multimodal support)"
                )
                report.add(
                    "llama_server_mmproj",
                    False,
                    "llama-server lacks --mmproj; cannot serve ASR",
                    severity="warn",
                    hint="Update it with `arc-llama install-runtime`.",
                )
            elif caps.probed:
                console.print("    [green]llama-server has --mmproj[/green] (multimodal build)")

        # Each TTS engine checks its own prerequisites: arc-llama cannot know
        # what OmniVoice needs without importing torch, and the next engine
        # will need something else again.
        for m in cfg.audio_models:
            if m.task != "tts":
                continue
            engine = get_tts_engine(m.engine)
            if engine is None:
                console.print(
                    f"    [red]unknown TTS engine[/red] {m.engine!r} for {m.name}"
                )
                report.add(
                    f"audio model {m.name}",
                    False,
                    f"unknown TTS engine {m.engine!r}",
                    severity="fail",
                    hint=f"Known engines: {', '.join(tts_engine_names()) or '(none)'}.",
                )
                continue
            problems = engine.preflight(cfg, m)
            for problem in problems:
                console.print(f"    [yellow]{m.name}[/yellow]: {problem}")
                report.add(f"audio model {m.name}", False, problem, severity="warn")
            if not problems:
                console.print(
                    f"    [green]{m.name}[/green] ready on the {m.engine} engine"
                )

        for m in cfg.audio_models:
            targets = [("model path", m.path)] if m.task == "asr" else []
            targets.append(("mmproj", m.audio_recipe().mmproj))
            for label, target in targets:
                if target and not Path(target).expanduser().exists():
                    console.print(f"    [red]missing {label}[/red] {m.name}: {target}")
                    report.add(
                        f"audio model {m.name}",
                        False,
                        f"{label} not found: {target}",
                        severity="warn",
                    )

        for v in cfg.voices:
            if v.ref_audio and not Path(v.ref_audio).expanduser().exists():
                console.print(f"    [red]missing reference audio[/red] {v.name}: {v.ref_audio}")
                report.add(
                    f"voice {v.name}",
                    False,
                    f"reference audio not found: {v.ref_audio}",
                    severity="warn",
                )

    # Summary of competitive-inference gates
    fails = report.failures
    warns = report.warnings
    console.print("\n  [bold]competitive-inference gates[/bold]")
    if not fails and not warns:
        console.print("    [green]all checked gates look good[/green]")
    else:
        for c in fails + warns:
            console.print(f"    {_doctor_marker(c.ok, c.severity)}  {c.name}: {c.detail}")
            if c.hint:
                console.print(f"        → {c.hint}")
    if fails:
        # Non-zero so scripts/CI can gate on a healthy Arc host.
        sys.exit(2)


# ===========================================================================
# gpus
# ===========================================================================


@cli.command("gpus")
def gpus_cmd() -> None:
    """List detected Intel GPUs."""
    gpus = detect_gpus()
    if not gpus:
        console.print("[red]No Intel GPUs detected.[/red]")
        sys.exit(2)
    _print_gpu_table(gpus)


def _print_gpu_table(gpus) -> None:
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


# ===========================================================================
# list
# ===========================================================================


@cli.command("list")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """List registered models and which one is currently loaded."""
    cfg = load_config(ctx.obj["config_path"])
    if not cfg.models:
        console.print("[yellow]No models registered. Run [bold]arc-llama add ...[/bold].[/yellow]")
        return

    loaded: set[str] = set()
    status_connected = False
    status_url = f"http://{cfg.server.host}:{cfg.server.port}/admin/status"
    headers = {}
    if cfg.server.admin_token:
        headers["Authorization"] = f"Bearer {cfg.server.admin_token}"
    try:
        r = httpx.get(status_url, timeout=2.0, headers=headers)
        r.raise_for_status()
        loaded = {m["name"] for m in r.json().get("models", []) if m.get("loaded")}
        status_connected = True
    except Exception:
        # Server not running; fall back to config-only listing.
        pass

    table = Table(title="Models")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("Backend")
    table.add_column("Port")
    table.add_column("ctx")
    table.add_column("KV")
    table.add_column("Spec")
    table.add_column("Path")
    for m in cfg.models:
        recipe = m.recipe or {}
        kv = f"{recipe.get('cache_type_k', 'f16')}/{recipe.get('cache_type_v', 'f16')}"
        spec = recipe.get("spec_type", "—")
        if recipe.get("ubatch_size"):
            spec += f" (ub={recipe['ubatch_size']})"
        gpu_cfg = cfg.find_gpu(m.gpu_pci_slot)
        backend = gpu_cfg.backend if gpu_cfg else "?"
        status = "[green]loaded[/green]" if m.name in loaded else "[dim]idle[/dim]"
        table.add_row(
            m.name,
            status,
            m.gpu_pci_slot,
            backend,
            str(m.port),
            str(recipe.get("ctx", "?")),
            kv,
            spec,
            m.path,
        )
    console.print(table)
    if not status_connected and cfg.server.host and cfg.server.port:
        console.print(
            f"[dim]Status not available from {status_url}; server may not be running.[/dim]"
        )


# ===========================================================================
# add
# ===========================================================================


@cli.command("add")
@click.argument("source")
@click.option("--name", default=None, help="Short name (default: derived from source).")
@click.option(
    "--gpu",
    "gpu_pci_slot",
    default=None,
    help="PCI slot of the GPU to bind to (default: first enabled GPU).",
)
@click.option(
    "--backend",
    "backend",
    default=None,
    type=click.Choice([Backend.SYCL.value, Backend.VULKAN.value]),
    help="Compute backend for this GPU (default: the GPU's configured backend, usually sycl).",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Backend port for this model's llama-server (default: auto).",
)
@click.option("--ctx", "ctx_override", type=int, default=None, help="Override context length.")
@click.option(
    "--kv",
    "kv_type",
    type=click.Choice(["f16", "q8_0", "q5_1", "q4_0"]),
    default=None,
    help="Override KV cache type (applies to both K and V).",
)
@click.option("--display-name", default="", help="Human-friendly name.")
@click.option(
    "--kv-class",
    type=click.Choice(
        [
            "default",
            "moe_a3b",
            "qwen3_dense",
            "qwen3_27b_dense",
            "qwen2_5",
            "gemma_swa",
            "phi4",
            "llama3",
            "deepseek_r1_distill",
        ]
    ),
    default="default",
    help="KV-class hint, used for VRAM estimation.",
)
@click.option("--alias", "aliases", multiple=True, help="Extra match strings (repeatable).")
@click.option(
    "--spec-type",
    "spec_type",
    default=None,
    help="Speculative decoding type (e.g. draft-mtp). Auto-detected for models "
    "with embedded MTP heads or a sidecar draft GGUF.",
)
@click.option(
    "--spec-draft-model",
    "spec_draft_model",
    default=None,
    help="Path to a sidecar speculative-draft GGUF (--spec-draft-model). "
    "Auto-detected from a sibling mtp-/draft- file when present.",
)
@click.option(
    "--spec-draft-ngl",
    "spec_draft_ngl",
    type=int,
    default=None,
    help="GPU layers for the draft model (--spec-draft-ngl); default 999.",
)
@click.option(
    "--ubatch-size",
    "ubatch_size",
    type=int,
    default=None,
    help="Ubatch size (-ub). Left unset by default; llama.cpp picks its own.",
)
@click.option(
    "--batch-size",
    "batch_size",
    type=int,
    default=None,
    help="Logical batch size (-b). Must be >= ubatch-size if both are set.",
)
@click.option(
    "--from-hf",
    is_flag=True,
    help="Treat SOURCE as a Hugging Face spec (`org/repo` or `org/repo:Q4_K_M`).",
)
@click.option("--hf-token", default=None, help="HF token for gated repos.")
@click.pass_context
def add(
    ctx: click.Context,
    source: str,
    name: str | None,
    gpu_pci_slot: str | None,
    backend: str | None,
    port: int | None,
    ctx_override: int | None,
    kv_type: str | None,
    display_name: str,
    kv_class: str,
    aliases: tuple[str, ...],
    spec_type: str | None,
    spec_draft_model: str | None,
    spec_draft_ngl: int | None,
    ubatch_size: int | None,
    batch_size: int | None,
    from_hf: bool,
    hf_token: str | None,
) -> None:
    """Register a model. SOURCE is either a local GGUF path or a HF spec."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    if not cfg.gpus:
        console.print("[red]No GPUs in config — run [bold]arc-llama init[/bold] first.[/red]")
        sys.exit(1)

    # Pick GPU
    if gpu_pci_slot is None:
        enabled = [g for g in cfg.gpus if g.enabled]
        if not enabled:
            console.print("[red]No enabled GPUs in config.[/red]")
            sys.exit(1)
        gpu_pci_slot = enabled[0].pci_slot

    # Apply backend override to the selected GPU.
    if backend is not None:
        gpu_cfg = cfg.find_gpu(gpu_pci_slot)
        if gpu_cfg is not None:
            gpu_cfg.backend = backend

    # Warn if the configured llama-server binary does not support the GPU backend.
    selected_gpu = cfg.find_gpu(gpu_pci_slot)
    if selected_gpu is not None:
        bin_path = Path(cfg.paths.llama_server).expanduser()
        if bin_path.exists():
            bin_backend = detect_llama_server_backend(bin_path)
            if bin_backend is not None and bin_backend.value != selected_gpu.backend:
                console.print(
                    f"[yellow]Warning: GPU {selected_gpu.pci_slot} is set to "
                    f"'{selected_gpu.backend}', but {bin_path} appears to be a "
                    f"'{bin_backend.value}' binary.[/yellow]"
                )

    # Resolve source → file path. Local file wins if it exists; otherwise try HF.
    local_candidate = Path(source).expanduser()
    treat_as_hf = from_hf or (not local_candidate.exists() and "/" in source)
    if treat_as_hf:
        try:
            spec = parse_hf_spec(source)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        target_dir = Path(cfg.paths.models_dir).expanduser() / spec.repo.split("/")[-1]
        console.print(f"[bold]Downloading[/bold] {spec.repo} → {target_dir}")
        try:
            path = download_from_hf(spec, target_dir=target_dir, token=hf_token)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        derived_name = name or _slugify_for_name(target_dir.name, path.name)
    else:
        path = local_candidate.resolve()
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        derived_name = name or _slugify_for_name(path.parent.name, path.name)

    # Recipe overrides
    overrides: dict = {}
    if ctx_override is not None:
        overrides["ctx"] = int(ctx_override)
    if kv_type is not None:
        overrides["cache_type_k"] = kv_type
        overrides["cache_type_v"] = kv_type
    if spec_type is not None:
        overrides["spec_type"] = spec_type
    if spec_draft_model is not None:
        overrides["spec_draft_model"] = str(Path(spec_draft_model).expanduser().resolve())
        overrides.setdefault("spec_type", "draft-mtp")
        overrides.setdefault("spec_draft_ngl", 999)
    if spec_draft_ngl is not None:
        overrides["spec_draft_ngl"] = spec_draft_ngl
    if ubatch_size is not None:
        overrides["ubatch_size"] = ubatch_size
    if batch_size is not None:
        overrides["batch_size"] = batch_size

    try:
        mc = add_local_model(
            cfg,
            name=derived_name,
            path=str(path),
            gpu_pci_slot=gpu_pci_slot,
            port=port,
            display_name=display_name,
            kv_class=kv_class,
            aliases=list(aliases) if aliases else None,
            recipe_overrides=overrides or None,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    _save_or_die(cfg, cfg_path)
    recipe = mc.launch_recipe()
    ctx_info = f"ctx={recipe.ctx}"
    batch_info = ""
    if recipe.ubatch_size is not None:
        batch_info += f" ub={recipe.ubatch_size}"
    if recipe.batch_size is not None:
        batch_info += f" b={recipe.batch_size}"
    console.print(
        f"[green]Registered {mc.name}[/green] on {gpu_pci_slot}, port {mc.port} "
        f"({ctx_info}{batch_info})"
    )


def _slugify_for_name(parent: str, file: str) -> str:
    import re

    base = parent.lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", base).strip("-")
    if not base:
        base = "model"
    m = re.search(r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|UD-[A-Z0-9_]+)", file, re.IGNORECASE)
    if m:
        base = f"{base}-{m.group(1).lower()}"
    return base


# ===========================================================================
# scan
# ===========================================================================


def _do_scan(cfg: Config, extra_paths: list[Path]) -> list:
    found = discover_ggufs(cfg, extra_paths=extra_paths)
    if not found:
        return []
    return register_discovered(cfg, found)


@cli.command("scan")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--gpu",
    "gpu_pci_slot",
    default=None,
    help="Bind newly discovered models to this PCI slot (default: first enabled GPU).",
)
@click.option(
    "--persist/--no-persist",
    default=True,
    help="Save the resulting config to disk (default: on). Disable for a dry-run.",
)
@click.pass_context
def scan_cmd(
    ctx: click.Context,
    paths: tuple[str, ...],
    gpu_pci_slot: str | None,
    persist: bool,
) -> None:
    """Walk scan paths for GGUFs and auto-register anything new."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    if not cfg.gpus:
        console.print("[red]No GPUs in config — run [bold]arc-llama init[/bold] first.[/red]")
        sys.exit(1)
    extras = [Path(p) for p in paths]
    found = discover_ggufs(cfg, extra_paths=extras)
    if not found:
        scanned = [cfg.paths.models_dir, *cfg.paths.scan_paths, *paths]
        console.print("[yellow]No GGUFs found.[/yellow] Scanned: " + ", ".join(scanned))
        return
    try:
        added = register_discovered(cfg, found, gpu_pci_slot=gpu_pci_slot)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if not added:
        console.print(f"[dim]Found {len(found)} GGUF(s); all already registered.[/dim]")
        return
    if persist:
        _save_or_die(cfg, cfg_path)
    console.print(
        f"[green]Registered {len(added)} new model(s):[/green] " + ", ".join(m.name for m in added)
    )
    if not persist:
        console.print("[dim]--no-persist: config NOT saved.[/dim]")


# ===========================================================================
# remove
# ===========================================================================


@cli.command("remove")
@click.argument("name")
@click.pass_context
def remove(ctx: click.Context, name: str) -> None:
    """Remove a model from the config (does NOT delete the GGUF)."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.models)
    cfg.models = [m for m in cfg.models if m.name != name]
    if len(cfg.models) == before:
        console.print(f"[yellow]No model named {name!r}.[/yellow]")
        sys.exit(1)
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed {name}.[/green]")


# ===========================================================================
# serve
def _print_autotune_banner(cfg: Config) -> None:
    """Tell the operator whether background tuning is active and how to disable it."""
    if not getattr(cfg, "tune", None) or not cfg.tune.auto:
        console.print(
            "[dim]Auto-tune: off (use --auto-tune or set [tune] auto=true to enable).[/dim]"
        )
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


# ===========================================================================


def _print_tune_status_table(cfg: Config) -> None:
    """Print the per-model tune state for `arc-llama tune --status`."""
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
            from datetime import datetime, timezone

            tuned_at = datetime.fromtimestamp(m.tuned_at, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
        table.add_row(
            m.name,
            m.tune_state,
            tuned_at,
            (m.tune_fingerprint[:16] + "...") if m.tune_fingerprint else "",
            m.tune_error[:60],
        )
    console.print(table)


def _print_serve_banner(cfg: Config) -> None:
    """Print the applied Arc profile per GPU + model at serve startup.

    Surfaces the gotchas arc-llama exists to encode — arch, backend, VRAM,
    ReBAR status, and the JIT-vs-AOT cold-start situation — so the user can
    see *why* their config is what it is without reading source comments.
    """
    console.print("[bold]arc-llama serve[/bold] — applied Arc profiles:")
    # Re-detect so we can show live ReBAR + the exact device ID for AOT hints.
    live: dict[str, DetectedGPU] = {}
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
        det: DetectedGPU | None = live.get(gpu.pci_slot)
        if det is not None and det.sysfs_path:
            r = rebar_likely_enabled(det.sysfs_path, det.vram_mb)
            parts.append("ReBAR on" if r else ("ReBAR OFF" if r is False else "ReBAR ?"))
        if backend == Backend.SYCL.value:
            aot = aot_arch_for(det.device_id) if det is not None else None
            if aot is None:
                # Fall back to a generation-level default from the configured
                # arch when the live PCI slot doesn't match (e.g. stale config
                # or running somewhere sysfs isn't available).
                try:
                    a = Arch(gpu.arch) if gpu.arch else None
                except ValueError:
                    a = None
                if a == Arch.BATTLEMAGE:
                    aot = "bmg-g21"
                elif a == Arch.ALCHEMIST:
                    aot = "acm-g10"
            if aot is not None:
                parts.append(f"JIT cold-start ~20s (AOT: -DGGML_SYCL_DEVICE_ARCH={aot})")
            else:
                parts.append("JIT cold-start")
        console.print("  " + " · ".join(parts))
    for m in cfg.models:
        recipe = m.recipe or {}
        ctx = recipe.get("ctx", "?")
        kv_k = recipe.get("cache_type_k", "f16")
        kv_v = recipe.get("cache_type_v", "f16")
        kv_txt = kv_k if kv_k == kv_v else f"{kv_k}/{kv_v}"
        console.print(f"  model {m.name}: ctx={ctx} · KV {kv_txt} · port {m.port}")
    if not cfg.models:
        console.print("  [dim]no models registered — `arc-llama add` something first[/dim]")


@cli.command("serve")
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", type=int, default=None, help="Override server port.")
@click.option(
    "--profile",
    default=None,
    help="Active MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--admin-token",
    default=None,
    help=(
        "Bearer token required for admin endpoints and auto_confirm agent runs "
        "(overrides config; also settable via ARC_LLAMA_ADMIN_TOKEN)."
    ),
)
@click.option(
    "--scan/--no-scan",
    "scan",
    default=True,
    help="Auto-register any new GGUFs found in models_dir/scan_paths on startup "
    "(default: on). Drop a model in and it just appears.",
)
@click.option(
    "--auto-tune/--no-auto-tune",
    "auto_tune",
    default=None,
    help="Enable background auto-tuning (default: from config tune.auto).",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    profile: str | None,
    admin_token: str | None,
    scan: bool,
    auto_tune: bool | None,
) -> None:
    """Run the OpenAI-compatible router."""
    cfg = load_config(ctx.obj["config_path"])
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    if profile:
        cfg.agent.profile = profile
    if admin_token:
        cfg.server.admin_token = admin_token
    if auto_tune is not None:
        cfg.tune.auto = auto_tune

    # Zero-config discovery: pick up any GGUF dropped into models_dir/scan_paths
    # since the last run, so `serve` reflects the filesystem without a manual
    # `scan`. Idempotent (already-registered paths are skipped) and best-effort
    # — a discovery failure must never stop the router from coming up.
    if scan and cfg.gpus:
        try:
            added = _do_scan(cfg, [])
        except Exception as e:  # noqa: BLE001 - discovery must not block serve
            added = []
            console.print(f"[yellow]Startup scan failed: {e}[/yellow]")
        if added:
            _save_or_die(cfg, ctx.obj["config_path"])
            console.print(
                f"[green]Auto-registered {len(added)} new model(s):[/green] "
                + ", ".join(m.name for m in added)
            )

    _print_autotune_banner(cfg)
    if not cfg.models:
        console.print(
            "[yellow]No models registered yet — drop a GGUF in "
            f"{cfg.paths.models_dir} or run `arc-llama add`.[/yellow]"
        )
    if cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]Binding to {cfg.server.host!r}, not loopback -- make sure "
            "admin_token is set to something you control (it was auto-generated "
            "if you never set one).[/yellow]"
        )
    token_source = (
        "ARC_LLAMA_ADMIN_TOKEN environment variable"
        if os.environ.get("ARC_LLAMA_ADMIN_TOKEN")
        else f"config file ({ctx.obj['config_path']})"
    )
    console.print(
        f"[dim]Admin authentication is enabled via {token_source}. "
        "Admin endpoints and auto_confirm agent runs require "
        "'Authorization: Bearer <token>'.[/dim]"
    )
    _print_serve_banner(cfg)
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        sys.exit(1)
    from arc_llama.server import create_app

    app = create_app(cfg, config_path=ctx.obj["config_path"])

    # Belt-and-suspenders for graceful shutdown: even if uvicorn's lifespan
    # handling misfires (e.g. on SIGTERM during a busy event loop), atexit
    # gives us one more chance to stop subprocesses before the parent dies.
    import atexit
    import signal as _signal

    def _shutdown_subprocesses() -> None:
        rt = getattr(app.state, "router", None)
        if rt is None:
            return
        # Async shutdown isn't possible from atexit if the loop is gone; call
        # the underlying LlamaServer.stop() synchronously instead.
        for srv in rt._servers.values():
            try:
                srv.stop()
            except Exception:
                pass

    atexit.register(_shutdown_subprocesses)

    def _on_signal(signum: int, _frame) -> None:  # noqa: ANN001
        _shutdown_subprocesses()
        # Re-raise as default so uvicorn's own handler (or python) finishes the job.
        _signal.signal(signum, _signal.SIG_DFL)
        if _IS_WINDOWS:
            sys.exit(0)
        else:
            os.kill(os.getpid(), signum)

    for s in (getattr(_signal, "SIGTERM", None), _signal.SIGINT):
        if s is None:
            continue
        try:
            _signal.signal(s, _on_signal)
        except (OSError, ValueError):
            pass

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


# ===========================================================================
# benchmark / tune
# ===========================================================================


def _server_url_from(ctx: click.Context, server_url: str | None) -> str:
    if server_url:
        return server_url.rstrip("/")
    cfg = load_config(ctx.obj["config_path"])
    return f"http://{cfg.server.host}:{cfg.server.port}"


@cli.command("benchmark")
@click.argument("model")
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option(
    "--prompt-tokens",
    "prompt_tokens",
    type=int,
    default=benchmark_mod.DEFAULT_PROMPT_TOKENS,
    show_default=True,
    help="Approximate prompt length to benchmark.",
)
@click.option(
    "--gen-tokens",
    "gen_tokens",
    type=int,
    default=benchmark_mod.DEFAULT_GEN_TOKENS,
    show_default=True,
    help="Number of tokens to generate.",
)
@click.option(
    "--sweep-ctx",
    "sweep_ctx",
    default="",
    help="Comma-separated ctx values for a sweep (e.g. 4096,8192,16384).",
)
@click.option(
    "--sweep-kv",
    "sweep_kv",
    default="",
    help="Comma-separated KV types for a sweep (e.g. f16,q8_0,q4_0).",
)
@click.option(
    "--kv",
    "kv_types",
    multiple=True,
    type=click.Choice(["f16", "q8_0", "q5_1", "q4_0"]),
    help="KV cache type(s) for --sweep-ctx (repeatable; default: f16 q8_0).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
@click.pass_context
def benchmark_cmd(
    ctx: click.Context,
    model: str,
    server_url: str | None,
    prompt_tokens: int,
    gen_tokens: int,
    sweep_ctx: str,
    sweep_kv: str,
    kv_types: tuple[str, ...],
    as_json: bool,
) -> None:
    """Measure prompt-eval and generation tok/s for MODEL.

    Requires a running `arc-llama serve` — measurements go through the
    router so they use the exact SYCL env and recipe your requests get.
    """
    cfg = load_config(ctx.obj["config_path"])
    if cfg.find_model(model) is None:
        console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
        sys.exit(1)
    url = _server_url_from(ctx, server_url)

    ctx_values = [int(x.strip()) for x in sweep_ctx.split(",") if x.strip()] if sweep_ctx else []
    kv_values = [x.strip() for x in sweep_kv.split(",") if x.strip()] if sweep_kv else []
    if kv_types:
        kv_values = list(kv_types)

    model_cfg = cfg.find_model(model)
    assert model_cfg is not None

    async def _run() -> int:
        if ctx_values or kv_values:
            recipe = model_cfg.recipe or {}
            results = await benchmark_mod.benchmark_sweep(
                url,
                model,
                ctx_values=ctx_values or [recipe.get("ctx", 4096)],
                kv_types=kv_values or [recipe.get("cache_type_k", "f16")],
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                cfg=cfg,
            )
            if as_json:
                click.echo(json.dumps([r.to_dict() for r in results], indent=2))
            else:
                benchmark_mod.print_sweep_table(results)
            return 1 if all(r.error for r in results) else 0
        result = await benchmark_mod.benchmark_model(
            url,
            model,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            cfg=cfg,
        )
        if as_json:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            benchmark_mod.print_result(result)
        return 1 if result.error else 0

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        console.print("[yellow]Benchmark interrupted.[/yellow]")
        sys.exit(130)


@cli.command("tune")
@click.argument("model", required=False)
@click.option(
    "--all",
    "all_models",
    is_flag=True,
    help="Tune every registered model sequentially.",
)
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option(
    "--target",
    type=click.Choice(["balanced", "generation", "prompt"]),
    default="balanced",
    show_default=True,
    help="What to optimise: generation tok/s, prompt-eval tok/s, or both.",
)
@click.option("--prompt-tokens", type=int, default=1024, show_default=True)
@click.option("--gen-tokens", type=int, default=128, show_default=True)
@click.option(
    "--apply/--dry-run",
    "apply_",
    default=True,
    help="Write the winning config into the model's recipe (default) or restore the original.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
@click.option(
    "--status",
    "status_only",
    is_flag=True,
    help="Print the per-model tune state table and exit without measuring.",
)
@click.pass_context
def tune_cmd(
    ctx: click.Context,
    model: str | None,
    all_models: bool,
    server_url: str | None,
    target: str,
    prompt_tokens: int,
    gen_tokens: int,
    apply_: bool,
    as_json: bool,
    status_only: bool,
) -> None:
    """Find the fastest recipe for MODEL by measuring, then persist it.

    Staged sweep over KV cache type, ubatch size, and flash attention —
    roughly 6–9 measured configs, each paying one model reload. Expect
    ~10 minutes on a Battlemage-class card. Pass `--all` to sweep every
    registered model in one run. Requires a running `arc-llama serve`.
    """
    from dataclasses import asdict

    from arc_llama.autotune import (
        compute_fingerprint,
        set_tuned_state,
    )
    from arc_llama.tune import print_multi_summary, print_report, tune_all, tune_model

    cfg = load_config(ctx.obj["config_path"])

    if status_only:
        _print_tune_status_table(cfg)
        sys.exit(0)

    if all_models and model:
        console.print("[red]Pass either MODEL or --all, not both.[/red]")
        sys.exit(1)
    if not all_models and not model:
        console.print("[red]Specify a MODEL to tune, or --all for every registered model.[/red]")
        sys.exit(1)

    url = _server_url_from(ctx, server_url)

    try:
        if all_models:
            model_names = [m.name for m in cfg.models]
            if not model_names:
                console.print("[yellow]No models registered.[/yellow]")
                sys.exit(0)

            def on_start(name: str, i: int, total: int) -> None:
                console.print(f"[bold]\\[{i}/{total}] tuning {name}[/bold]")

            reports = asyncio.run(
                tune_all(
                    url,
                    model_names,
                    target=target,
                    prompt_tokens=prompt_tokens,
                    gen_tokens=gen_tokens,
                    apply=apply_,
                    cfg=cfg,
                    on_start=on_start,
                )
            )
            # Dry-run must leave tune state untouched: recording "tuned" with a
            # matching fingerprint makes background auto-tune skip the model
            # forever, turning a look-don't-touch run into a permanent opt-out.
            if apply_:
                for r in reports:
                    if not r.error and not r.aborted:
                        m = cfg.find_model(r.model)
                        if m is not None:
                            gpu = cfg.find_gpu(m.gpu_pci_slot)
                            from arc_llama import __version__, workload

                            fp = compute_fingerprint(
                                m,
                                cfg.paths.llama_server,
                                gpu,
                                __version__,
                                workload.fingerprint_key(cfg.workload),
                            )
                            set_tuned_state(cfg, m, fp)
                try:
                    cfg.save(ctx.obj["config_path"])
                except OSError as e:
                    console.print(f"[yellow]Warning: failed to save tune state: {e}[/yellow]")
            if as_json:
                click.echo(json.dumps([asdict(r) for r in reports], indent=2, default=str))
            else:
                print_multi_summary(reports)
            sys.exit(1 if any(r.error for r in reports) else 0)

        assert model is not None
        if cfg.find_model(model) is None:
            console.print(f"[red]Model '{model}' is not registered in the config.[/red]")
            sys.exit(1)
        report = asyncio.run(
            tune_model(
                url,
                model,
                target=target,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                apply=apply_,
                cfg=cfg,
            )
        )
    except KeyboardInterrupt:
        console.print("[yellow]Tune interrupted.[/yellow]")
        sys.exit(130)

    # Same dry-run guard as the --all branch above.
    if apply_ and not report.error and not report.aborted:
        m = cfg.find_model(report.model)
        if m is not None:
            gpu = cfg.find_gpu(m.gpu_pci_slot)
            from arc_llama import __version__, workload

            fp = compute_fingerprint(
                m,
                cfg.paths.llama_server,
                gpu,
                __version__,
                workload.fingerprint_key(cfg.workload),
            )
            set_tuned_state(cfg, m, fp)
            try:
                cfg.save(ctx.obj["config_path"])
            except OSError as e:
                console.print(f"[yellow]Warning: failed to save tune state: {e}[/yellow]")

    if as_json:
        click.echo(json.dumps(asdict(report), indent=2, default=str))
    else:
        print_report(report)
    sys.exit(1 if report.error else 0)


# ===========================================================================
# install-runtime
# ===========================================================================


@cli.command("install-runtime")
@click.option(
    "--backend",
    type=click.Choice(["vulkan", "sycl"]),
    default="vulkan",
    show_default=True,
    help=(
        "Which prebuilt llama-server to fetch. Vulkan is portable (no oneAPI); "
        "SYCL is faster on Arc but needs the oneAPI runtime on Linux."
    ),
)
@click.option(
    "--runtime-version",
    "runtime_version",
    default="latest",
    show_default=True,
    help="llama.cpp release tag (e.g. b10092) or 'latest'.",
)
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=None,
    help="Install directory (default: <state_dir>/runtime).",
)
@click.option(
    "--set-default/--no-set-default",
    default=True,
    help="Write the downloaded binary's path into the config as paths.llama_server.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-download even if this version is already installed.",
)
@click.pass_context
def install_runtime_cmd(ctx, backend, runtime_version, dest, set_default, force):
    """Download a prebuilt llama-server so you can skip building llama.cpp.

    Fetches an official ggml-org/llama.cpp release binary for your platform,
    extracts it, verifies its compute backend, and (by default) points the
    config at it. Vulkan works on any Arc card with the Mesa/ANV driver and
    needs no oneAPI install.
    """
    import platform as _platform

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
    )

    from arc_llama.runtime import RuntimeInstallError, install_runtime

    cfg = load_config(ctx.obj["config_path"])
    console.print(f"[bold]Fetching {backend} llama-server[/bold] (llama.cpp {runtime_version}) ...")

    progress = Progress(
        TextColumn("[bold blue]Downloading"),
        BarColumn(),
        DownloadColumn(),
        TaskProgressColumn(),
        console=console,
    )
    state: dict = {"task_id": None}

    def on_progress(done: int, total: int) -> None:
        if state["task_id"] is None:
            state["task_id"] = progress.add_task("download", total=total or 1)
        progress.update(state["task_id"], completed=done)

    try:
        with progress:
            result = install_runtime(
                backend=backend,
                version=runtime_version,
                dest=dest,
                cfg=cfg,
                set_default=set_default,
                config_path=ctx.obj["config_path"],
                force=force,
                on_progress=on_progress,
            )
    except RuntimeInstallError as e:
        console.print(f"[red]install-runtime failed:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]install-runtime failed:[/red] {e}")
        sys.exit(1)

    console.print(f"[green]Installed[/green] {result.binary_path}")
    console.print(
        f"  backend detected: "
        f"{result.backend.value if result.backend else 'unknown'}"
        f"  (requested: {result.requested_backend})"
    )
    console.print(f"  llama.cpp tag:    {result.tag}")
    if result.set_as_default:
        console.print("  [dim]config paths.llama_server updated.[/dim]")
    if backend == "sycl" and _platform.system() == "Linux":
        console.print(
            "  [yellow]SYCL on Linux needs the oneAPI runtime present at run time "
            "(source setvars.sh). Use --backend vulkan if you don't have oneAPI.[/yellow]"
        )
    console.print("\nNext: [bold]arc-llama serve[/bold]")


# ===========================================================================
# mtp-info
# ===========================================================================


@cli.command("mtp-info")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def mtp_info_cmd(path: Path) -> None:
    """Inspect a GGUF file for MTP-relevant metadata."""
    from arc_llama.gguf_meta import mtp_info

    info = mtp_info(path)
    console.print(f"[bold]GGUF:[/bold] {info['path']}")
    console.print(f"  architecture:          {info['architecture']}")
    console.print(f"  block_count:           {info['block_count']}")
    console.print(f"  nextn_predict_layers:  {info['nextn_predict_layers']}")
    console.print(f"  has_mtp_heads:         {info['has_mtp_heads']}")
    console.print(f"  is_hybrid_ssm:         {info['is_hybrid_ssm']}")


# ===========================================================================
# systemd
# ===========================================================================


@cli.command("systemd")
@click.option("--service-name", default="arc-llama.service")
@click.option("--description", default="arc-llama OpenAI-compatible router")
@click.option("--write", is_flag=True, help="Write the unit to ~/.config/systemd/user/")
def systemd_unit(service_name: str, description: str, write: bool) -> None:
    """Print (or write) a systemd --user unit for `arc-llama serve`."""
    if _IS_WINDOWS:
        console.print("[red]systemd is not available on Windows.[/red]")
        sys.exit(1)
    arc = shutil.which("arc-llama")
    if not arc:
        arc = str(Path(sys.argv[0]).resolve())
    unit = f"""[Unit]
Description={description}
After=network.target

[Service]
Type=simple
ExecStart={arc} serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    if not write:
        click.echo(unit)
        return
    target = Path.home() / ".config" / "systemd" / "user" / service_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(unit)
    console.print(f"[green]Wrote {target}[/green]")
    console.print(
        "Enable with: [bold]systemctl --user daemon-reload && "
        f"systemctl --user enable --now {service_name}[/bold]"
    )


# ===========================================================================
# upstream
# ===========================================================================


@cli.group("audio")
def audio_group() -> None:
    """Manage speech models: transcription (STT) and synthesis (TTS)."""


def _parse_options(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Turn repeated `--option key=value` flags into the recipe's options bag.

    Values are parsed as JSON when they can be, so `num_step=16` arrives as an
    int and `normalize_text=true` as a bool. An engine that wants a string
    keeps one, because a bare word is not valid JSON and falls through.
    """
    options: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"--option expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            options[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            options[key.strip()] = value
    return options


@audio_group.command("engines")
def audio_engines() -> None:
    """List the TTS engines this build can serve."""
    table = Table(title="TTS engines")
    table.add_column("name")
    table.add_column("description")
    for engine in tts_engines():
        table.add_row(engine.name, engine.description)
    console.print(table)
    console.print(
        "\n[dim]Transcription always runs on llama-server: it is the only ASR "
        "runtime with a SYCL build.[/dim]"
    )


@audio_group.command("add")
@click.argument("source")
@click.option(
    "--task",
    type=click.Choice(["asr", "tts"]),
    default="asr",
    help="Which OpenAI endpoint this model serves: /v1/audio/transcriptions "
    "or /v1/audio/speech.",
)
@click.option(
    "--engine",
    default=None,
    help="Runtime that serves this model. ASR is always llamacpp; for TTS see "
    "`arc-llama audio engines` (default: omnivoice).",
)
@click.option(
    "--mmproj",
    default=None,
    help="ASR: path to the audio projector GGUF (mmproj-*.gguf). "
    "Auto-resolved from a sibling file or an --from-hf download.",
)
@click.option(
    "--from-hf",
    is_flag=True,
    help="Treat SOURCE as a Hugging Face spec (`org/repo` or `org/repo:Q8_0`) "
    "and download the weights and the projector together.",
)
@click.option("--hf-token", default=None, help="HF token for gated repos.")
@click.option("--ctx", "ctx_len", type=int, default=0, help="ASR: context length (-c).")
@click.option(
    "--no-strip-markers",
    is_flag=True,
    help="Keep the model's raw output framing. By default arc-llama strips "
    "Qwen3-ASR's `language English<asr_text>` prefix, which llama.cpp "
    "forwards verbatim and which breaks intent matching in Home Assistant.",
)
@click.option(
    "--python",
    "python_bin",
    default=None,
    help="TTS: interpreter for this model's backend, overriding paths.tts_python.",
)
@click.option(
    "--device",
    default=None,
    help="TTS: compute device as the engine names it (xpu, cuda:0, cpu). "
    "Default: xpu on a SYCL GPU.",
)
@click.option("--dtype", default=None, help="TTS: weight dtype (default: float16).")
@click.option(
    "--voice",
    "default_voice",
    default=None,
    help="TTS: voice used when a request's `voice` matches nothing registered.",
)
@click.option(
    "--language", default=None, help="TTS: language used when a request does not say."
)
@click.option(
    "--response-format",
    default=None,
    help="TTS: encoding used when a request omits response_format (default: mp3).",
)
@click.option(
    "--option",
    "options",
    multiple=True,
    help="TTS: engine-specific knob as key=value (repeatable), e.g. "
    "--option num_step=16.",
)
@click.option("--name", default=None, help="Short name (default: derived from SOURCE).")
@click.option(
    "--mode",
    type=click.Choice(["offline", "streaming"]),
    default="offline",
    help="Streaming is required for stream=true transcriptions.",
)
@click.option(
    "--gpu",
    "gpu_pci_slot",
    default=None,
    help="PCI slot of the GPU to bind to (default: first enabled GPU).",
)
@click.option("--port", type=int, default=None, help="Backend port for this model's process.")
@click.option("--display-name", default="", help="Human-friendly name.")
@click.option(
    "--alias",
    "aliases",
    multiple=True,
    help="Extra match strings (repeatable). Add `whisper-1` or `tts-1` for "
    "clients that hardcode OpenAI's model ids.",
)
@click.option(
    "--swappable",
    is_flag=True,
    help="Let this model be evicted by the single-resident swap policy. By "
    "default audio models stay pinned so a speech request never cold-starts "
    "your LLM.",
)
@click.option(
    "--vram-mb",
    type=int,
    default=None,
    help="Declared VRAM footprint for the load-time fit guard (default: "
    "estimated from the model size on disk).",
)
@click.pass_context
def audio_add(
    ctx: click.Context,
    source: str,
    task: str,
    engine: str | None,
    mmproj: str | None,
    from_hf: bool,
    hf_token: str | None,
    ctx_len: int,
    no_strip_markers: bool,
    python_bin: str | None,
    device: str | None,
    dtype: str | None,
    default_voice: str | None,
    language: str | None,
    response_format: str | None,
    options: tuple[str, ...],
    name: str | None,
    mode: str,
    gpu_pci_slot: str | None,
    port: int | None,
    display_name: str,
    aliases: tuple[str, ...],
    swappable: bool,
    vram_mb: int | None,
) -> None:
    """Register an audio model.

    SOURCE is a .gguf file, a model directory, a Hugging Face spec with
    --from-hf (e.g. ggml-org/Qwen3-ASR-0.6B-GGUF:Q8_0), or — for a TTS engine
    that resolves them itself — a plain repo id such as k2-fsa/OmniVoice.
    """
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    if not cfg.gpus:
        console.print("[red]No GPUs in config — run [bold]arc-llama init[/bold] first.[/red]")
        sys.exit(1)

    if gpu_pci_slot is None:
        enabled = [g for g in cfg.gpus if g.enabled]
        if not enabled:
            console.print("[red]No enabled GPUs in config.[/red]")
            sys.exit(1)
        gpu_pci_slot = enabled[0].pci_slot

    if engine is None:
        engine = AUDIO_ENGINE_LLAMACPP if task == "asr" else TTS_ENGINE_OMNIVOICE

    local_candidate = Path(source).expanduser()
    if task == "tts":
        # A TTS engine may resolve a repo id itself, so an argument that is not
        # a local path is passed straight through rather than downloaded here:
        # the engine knows the repo layout and this command does not.
        derived = name or _slugify_for_name(
            local_candidate.name if local_candidate.exists() else source.split("/")[-1], ""
        )
    else:
        treat_as_hf = from_hf or (not local_candidate.exists() and "/" in source)
        if treat_as_hf:
            try:
                spec = parse_hf_spec(source)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                sys.exit(1)
            target_dir = Path(cfg.paths.models_dir).expanduser() / spec.repo.split("/")[-1]
            console.print(f"[bold]Downloading[/bold] {spec.repo} (weights + mmproj) → {target_dir}")
            try:
                model_path, mmproj_path = download_asr_from_hf(
                    spec, target_dir=target_dir, token=hf_token
                )
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                console.print(f"[red]{e}[/red]")
                sys.exit(1)
            source = str(model_path)
            if mmproj is None:
                mmproj = str(mmproj_path)
            derived = name or _slugify_for_name(target_dir.name, model_path.name)
        else:
            p = local_candidate
            derived = name or _slugify_for_name(p.stem if p.is_file() else p.name, "")
            # A projector normally sits beside the weights under a predictable
            # name; finding it saves the user a flag they'd otherwise have to
            # discover from an error message.
            if mmproj is None and p.is_file():
                sibling = p.parent / f"mmproj-{p.name}"
                if sibling.exists():
                    mmproj = str(sibling)
                    console.print(f"[dim]Found projector beside the weights: {sibling.name}[/dim]")

    recipe_overrides: dict[str, Any] = {
        "python": python_bin,
        "device": device,
        "dtype": dtype,
        "default_voice": default_voice,
        "default_language": language,
        "default_response_format": response_format,
        "options": _parse_options(options),
    }

    try:
        entry = add_audio_model(
            cfg,
            name=derived,
            path=source,
            gpu_pci_slot=gpu_pci_slot,
            engine=engine,
            mmproj=mmproj or "",
            task=task,
            mode=mode,
            port=port,
            display_name=display_name,
            aliases=list(aliases),
            always_resident=not swappable,
            vram_mb=vram_mb,
            ctx=ctx_len,
            recipe_overrides=recipe_overrides,
            strip_asr_markers=not no_strip_markers,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    _save_or_die(cfg, cfg_path)
    console.print(
        f"[green]Registered[/green] audio model [bold]{entry.name}[/bold] "
        f"({entry.engine}, {entry.task}, {entry.mode}) on port {entry.port}"
    )
    if entry.task == "asr":
        caps = probe_server_caps(cfg.paths.llama_server)
        if caps.probed and not caps.supports_mmproj:
            console.print(
                f"[yellow]{cfg.paths.llama_server} has no --mmproj, so it was "
                "built without multimodal support and cannot serve ASR. "
                "Update it with `arc-llama install-runtime`.[/yellow]"
            )
    else:
        tts_engine = get_tts_engine(entry.engine)
        for problem in tts_engine.preflight(cfg, entry) if tts_engine else []:
            console.print(f"[yellow]{problem}[/yellow]")
        if not cfg.voices:
            console.print(
                "[dim]No voices registered yet — requests will let the model pick "
                "one. Add a cloned voice with:[/dim]\n"
                "  arc-llama audio voice add glados --ref-audio ref.wav "
                '--ref-text "the reference transcript"'
            )
    if not entry.always_resident:
        console.print(
            "[yellow]This model is swappable: a speech request will evict your "
            "LLM, and the next chat reply will pay a full cold start.[/yellow]"
        )


@audio_group.command("set-python")
@click.argument("path")
@click.pass_context
def audio_set_python(ctx: click.Context, path: str) -> None:
    """Point arc-llama at the interpreter that runs a Python TTS backend.

    OmniVoice pulls in torch, transformers and torchaudio, which arc-llama
    deliberately does not depend on — so it lives in its own virtualenv and
    this is how arc-llama finds it:

        arc-llama audio set-python ~/git/OmniVoice/.venv/bin/python
    """
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    resolved = resolve_binary(path)
    if resolved is None:
        console.print(
            f"[red]Not found: {path}[/red] "
            "(a bare name is looked up on PATH; pass a full path otherwise)"
        )
        sys.exit(1)
    if not os.access(resolved, os.X_OK):
        console.print(f"[red]Not executable: {resolved}[/red]")
        sys.exit(1)
    # Absolute, but deliberately NOT resolved. A virtualenv's `bin/python` is a
    # symlink to the base interpreter, and it is the path you invoke that tells
    # Python which prefix it is running in — following the link yields an
    # interpreter that cannot import anything the venv installed. Storing the
    # resolved target here silently registered a working OmniVoice env as a
    # broken one.
    cfg.paths.tts_python = (
        path if os.sep not in path else str(Path(path).expanduser().absolute())
    )
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]tts_python[/green] = {cfg.paths.tts_python}")


@audio_group.command("list")
@click.pass_context
def audio_list(ctx: click.Context) -> None:
    """List registered audio models."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    if not cfg.audio_models:
        console.print("No audio models registered. Add one with `arc-llama audio add`.")
        return
    table = Table(title="Audio models")
    table.add_column("name")
    table.add_column("task")
    table.add_column("engine")
    table.add_column("detail")
    table.add_column("mode")
    table.add_column("port")
    table.add_column("pinned")
    table.add_column("path")
    for m in cfg.audio_models:
        recipe = m.audio_recipe()
        if m.task == "asr":
            detail = Path(recipe.mmproj).name if recipe.mmproj else "—"
        else:
            detail = recipe.device or "auto"
        table.add_row(
            m.name,
            m.task,
            m.engine,
            detail,
            m.mode,
            str(m.port),
            "yes" if m.always_resident else "no",
            m.path,
        )
    console.print(table)


@audio_group.command("rm")
@click.argument("name")
@click.pass_context
def audio_rm(ctx: click.Context, name: str) -> None:
    """Unregister an audio model."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.audio_models)
    cfg.audio_models = [m for m in cfg.audio_models if m.name != name]
    if len(cfg.audio_models) == before:
        console.print(f"[red]Unknown audio model: {name}[/red]")
        sys.exit(1)
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed[/green] audio model {name}")


@audio_group.group("voice")
def voice_group() -> None:
    """Manage the named voices that `/v1/audio/speech` resolves."""


@voice_group.command("add")
@click.argument("name")
@click.option(
    "--ref-audio",
    default="",
    help="Reference clip to clone: 3–10 s of clean speech in the target language.",
)
@click.option(
    "--ref-text",
    default="",
    help="Transcript of --ref-audio. Omitting it makes the backend transcribe "
    "the clip with Whisper on first use, which loads a second model onto the GPU.",
)
@click.option(
    "--instruct",
    default="",
    help="Design a voice from attributes instead of cloning, e.g. "
    "'female, low pitch, british accent'. Ignored when --ref-audio is given.",
)
@click.option(
    "--auto",
    is_flag=True,
    help="The model's own voice, with no prompt. Use this for a fine-tuned "
    "model whose speaker is baked into the weights — a clone or design prompt "
    "on top would fight the training.",
)
@click.option("--language", default="", help="Language this voice speaks, e.g. English.")
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Restrict this voice to specific TTS models (repeatable). "
    "Default: available to all of them.",
)
@click.option("--display-name", default="", help="Human-friendly name.")
@click.option(
    "--alias",
    "aliases",
    multiple=True,
    help="Extra match strings (repeatable). Add `alloy` for clients that "
    "hardcode one of OpenAI's voice ids.",
)
@click.pass_context
def voice_add(
    ctx: click.Context,
    name: str,
    ref_audio: str,
    ref_text: str,
    instruct: str,
    auto: bool,
    language: str,
    models: tuple[str, ...],
    display_name: str,
    aliases: tuple[str, ...],
) -> None:
    """Register a named voice.

    Either clone one from a reference clip:

        arc-llama audio voice add glados --ref-audio ref.wav \\
            --ref-text "All right, look. We've both said a lot of things."

    design one from attributes:

        arc-llama audio voice add narrator --instruct "male, low pitch, british accent"

    or name the model's own voice, for a fine-tune that already speaks it:

        arc-llama audio voice add glados --auto
    """
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    try:
        voice = add_voice(
            cfg,
            name=name,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            auto=auto,
            models=list(models),
            display_name=display_name,
            aliases=list(aliases),
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    _save_or_die(cfg, cfg_path)
    mode = "clone" if voice.ref_audio else ("design" if voice.instruct else "auto")
    console.print(f"[green]Registered[/green] voice [bold]{voice.name}[/bold] ({mode})")
    if voice.ref_audio and not voice.ref_text:
        console.print(
            "[yellow]No --ref-text, so the backend will transcribe the clip with "
            "Whisper the first time this voice is used — a second model on the "
            "GPU and a slower first request. Supplying the transcript avoids "
            "both.[/yellow]"
        )


@voice_group.command("list")
@click.pass_context
def voice_list(ctx: click.Context) -> None:
    """List registered voices."""
    cfg = load_config(ctx.obj["config_path"])
    if not cfg.voices:
        console.print(
            "No voices registered. Speech requests will let the model pick a "
            "voice. Add one with `arc-llama audio voice add`."
        )
        return
    table = Table(title="Voices")
    table.add_column("name")
    table.add_column("mode")
    table.add_column("language")
    table.add_column("models")
    table.add_column("aliases")
    table.add_column("reference / attributes")
    for v in cfg.voices:
        if v.ref_audio:
            mode, detail = "clone", Path(v.ref_audio).name
        elif v.instruct:
            mode, detail = "design", v.instruct
        else:
            mode, detail = "auto", "—"
        table.add_row(
            v.name,
            mode,
            v.language or "—",
            ", ".join(v.models) or "all",
            ", ".join(v.aliases) or "—",
            detail,
        )
    console.print(table)


@voice_group.command("rm")
@click.argument("name")
@click.pass_context
def voice_rm(ctx: click.Context, name: str) -> None:
    """Unregister a voice."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.voices)
    cfg.voices = [v for v in cfg.voices if v.name != name]
    if len(cfg.voices) == before:
        console.print(f"[red]Unknown voice: {name}[/red]")
        sys.exit(1)
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed[/green] voice {name}")


@cli.group("upstream")
def upstream_group() -> None:
    """Manage upstream OpenAI-compatible endpoints."""


@upstream_group.command("add")
@click.argument("name")
@click.argument("url")
@click.pass_context
def upstream_add(ctx: click.Context, name: str, url: str) -> None:
    """Register an upstream endpoint. URL should be the base URL
    (e.g. http://127.0.0.1:11434 or http://192.168.1.50:8080)."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    # Validate URL roughly
    if not url.startswith(("http://", "https://")):
        console.print(f"[red]URL must start with http:// or https://: {url}[/red]")
        sys.exit(1)
    # Check for duplicate name
    existing = next((u for u in cfg.upstreams if u.name == name), None)
    if existing is not None:
        console.print(f"[yellow]Upstream '{name}' already exists. Remove it first.[/yellow]")
        sys.exit(1)
    from arc_llama.config import UpstreamConfig

    cfg.upstreams.append(UpstreamConfig(name=name, url=url.rstrip("/")))
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]Added upstream '{name}' at {url}[/green]")


@upstream_group.command("list")
@click.pass_context
def upstream_list(ctx: click.Context) -> None:
    """List registered upstream endpoints."""
    cfg = load_config(ctx.obj["config_path"])
    if not cfg.upstreams:
        console.print("[dim]No upstreams configured.[/dim]")
        return
    table = Table(title="Upstreams")
    table.add_column("Name")
    table.add_column("URL")
    for u in cfg.upstreams:
        table.add_row(u.name, u.url)
    console.print(table)


@upstream_group.command("remove")
@click.argument("name")
@click.pass_context
def upstream_remove(ctx: click.Context, name: str) -> None:
    """Remove an upstream endpoint."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.upstreams)
    cfg.upstreams = [u for u in cfg.upstreams if u.name != name]
    if len(cfg.upstreams) == before:
        console.print(f"[yellow]No upstream named {name!r}.[/yellow]")
        sys.exit(1)
    _save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed upstream '{name}'.[/green]")


# ===========================================================================
# agent
# ===========================================================================


def _state_dir_from_config(cfg: Config) -> Path | None:
    if cfg.paths.state_dir:
        return Path(cfg.paths.state_dir).expanduser()
    return None


@asynccontextmanager
async def _agent_tool_context(cfg: Config, profile: str | None):
    """Load skills and start the active profile's MCP servers for a CLI agent run."""
    load_skills(cfg.paths.skills_dir)
    manager = MCPClientManager(cfg.active_mcp_servers(profile))
    try:
        await manager.start()
        yield
    finally:
        await manager.stop()


async def _prompt_yes_no(prompt: str) -> bool:
    """Prompt the user for a yes/no answer from an async context."""
    loop = asyncio.get_running_loop()
    while True:
        answer = await loop.run_in_executor(None, input, prompt)
        cleaned = answer.strip().lower()
        if cleaned in ("y", "yes"):
            return True
        if cleaned in ("n", "no"):
            return False
        console.print("[dim]Please answer y or n.[/dim]")


def _render_agent_event(event: dict) -> None:
    t = event.get("type")
    if t == "status":
        console.print(f"[dim]# {event.get('message', '')}[/dim]")
    elif t == "plan":
        console.print("[bold cyan]Proposed plan:[/bold cyan]")
        console.print(event.get("content", ""))
    elif t == "assistant":
        content = event.get("content", "")
        if content:
            console.print(content)
    elif t == "tool_call":
        name = event.get("name", "tool")
        args = event.get("arguments", {})
        console.print(f"[bold yellow]▶ {name}[/bold yellow]")
        console.print(f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]")
    elif t == "tool_result":
        name = event.get("name", "tool")
        content = event.get("content", "")
        if event.get("error"):
            console.print(f"[red]✗ {name} failed[/red]")
        else:
            console.print(f"[green]✓ {name} done[/green]")
        console.print(f"[dim]{content}[/dim]")
    elif t == "confirm_required":
        console.print(f"[yellow]⚠ Confirmation required for {event.get('tool', 'tool')}[/yellow]")
    elif t == "checkpoint":
        console.print(f"[dim]Checkpoint saved: {event.get('id', '')}[/dim]")
    elif t == "error":
        console.print(f"[red]Error: {event.get('message', '')}[/red]")
    elif t == "done":
        console.print("[green]Agent finished.[/green]")


@cli.command("agent")
@click.argument("task")
@click.option("--model", "-m", required=True, help="Model id to use.")
@click.option("--root", "-r", default=None, help="Project root (default: agent.root from config).")
@click.option("--auto-confirm", is_flag=True, help="Do not prompt for tool confirmation.")
@click.option("--plan-mode", is_flag=True, help="Generate a plan first and ask for approval.")
@click.option("--max-turns", type=int, default=30, help="Maximum agent turns (default: 30).")
@click.option("--folder", "-f", default="", help="Folder to save the agent transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def agent_cmd(
    ctx: click.Context,
    task: str,
    model: str,
    root: str | None,
    auto_confirm: bool,
    plan_mode: bool,
    max_turns: int,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Run the local coding agent from the terminal.

    Requires a running `arc-llama serve` instance. The agent streams events to
    the terminal and prompts for confirmation before destructive tools.
    """
    cfg = load_config(ctx.obj["config_path"])
    if base_url is None:
        base_url = f"http://{cfg.server.host}:{cfg.server.port}"

    try:
        health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        console.print(f"[red]Cannot reach arc-llama server at {base_url}: {e}[/red]")
        console.print("[dim]Start one with:[/dim] arc-llama serve")
        sys.exit(1)

    root_path = Path(root or cfg.agent.root).expanduser().resolve()
    state_dir = _state_dir_from_config(cfg)
    chat_store = ChatStore(state_dir / "chats" if state_dir else Path(".arc_llama_chats"))
    checkpoint_store = CheckpointStore(
        state_dir / "checkpoints" if state_dir else Path(".arc_llama_checkpoints")
    )

    title = task.strip().split("\n")[0][:80] or "Agent task"
    agent_chat = chat_store.create(str(uuid.uuid4()), title, folder=folder)
    run_id = str(uuid.uuid4())
    transcript: list[ChatMessage] = [ChatMessage(role="user", content=task)]

    async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
        summary = json.dumps(arguments, ensure_ascii=False)[:200]
        return await _prompt_yes_no(f"Allow [bold]{tool}[/bold] {summary}? [y/n] ")

    async def plan_callback(plan_text: str) -> bool:
        return await _prompt_yes_no("Approve plan? [y/n] ")

    async def run() -> None:
        async with _agent_tool_context(cfg, profile):
            async for event in run_agent(
                task=task,
                model=model,
                base_url=base_url,
                root=root_path,
                auto_confirm=auto_confirm,
                confirm_callback=confirm_callback,
                plan_mode=plan_mode,
                plan_callback=plan_callback,
                run_id=run_id,
                checkpoint_store=checkpoint_store,
                max_turns=max_turns,
                chat_store=chat_store,
            ):
                _render_agent_event(event)
                if event.get("type") == "assistant" and event.get("content"):
                    transcript.append(ChatMessage(role="assistant", content=event["content"]))
                elif event.get("type") == "tool_result":
                    name = event.get("name", "tool")
                    content = event.get("content", "")
                    transcript.append(ChatMessage(role="tool", content=f"{name}:\n{content}"))

            chat = chat_store.get(agent_chat.id)
            if chat is not None:
                chat.messages.extend(transcript)
                chat_store.save(chat)
                console.print(
                    f"[dim]Transcript saved: chat {agent_chat.id} in folder '{folder or 'default'}'[/dim]"
                )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("[yellow]Agent run interrupted.[/yellow]")


# ===========================================================================
# code (interactive agent REPL)
# ===========================================================================


@cli.command("code")
@click.option("--model", "-m", required=True, help="Model id to use.")
@click.option("--root", "-r", default=None, help="Project root (default: agent.root from config).")
@click.option("--auto-confirm", is_flag=True, help="Do not prompt for tool confirmation.")
@click.option(
    "--plan-mode", is_flag=True, help="Generate a plan first and ask for approval each turn."
)
@click.option(
    "--max-turns", type=int, default=30, help="Maximum agent turns per user message (default: 30)."
)
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def code_cmd(
    ctx: click.Context,
    model: str,
    root: str | None,
    auto_confirm: bool,
    plan_mode: bool,
    max_turns: int,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Start an interactive coding agent REPL.

    Requires a running `arc-llama serve` instance. Type messages and the agent
    will use tools across multiple turns. Special commands start with `/`.
    """
    cfg = load_config(ctx.obj["config_path"])
    if base_url is None:
        base_url = f"http://{cfg.server.host}:{cfg.server.port}"

    try:
        health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        health.raise_for_status()
    except Exception as e:
        console.print(f"[red]Cannot reach arc-llama server at {base_url}: {e}[/red]")
        console.print("[dim]Start one with:[/dim] arc-llama serve")
        sys.exit(1)

    root_path = Path(root or cfg.agent.root).expanduser().resolve()
    state_dir = _state_dir_from_config(cfg)
    chat_store = ChatStore(state_dir / "chats" if state_dir else Path(".arc_llama_chats"))
    checkpoint_store = CheckpointStore(
        state_dir / "checkpoints" if state_dir else Path(".arc_llama_checkpoints")
    )

    session_chat = chat_store.create(str(uuid.uuid4()), "CLI session", folder=folder)
    run_id = str(uuid.uuid4())

    agent = InteractiveAgent(
        model=model,
        base_url=base_url,
        root=root_path,
        auto_confirm=auto_confirm,
        plan_mode=plan_mode,
        max_turns=max_turns,
        chat_store=chat_store,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
    )

    settings = {
        "model": model,
        "root": str(root_path),
        "folder": folder or "default",
        "auto_confirm": auto_confirm,
        "plan_mode": plan_mode,
        "max_turns": max_turns,
    }

    console.print("[bold green]arc-llama code[/bold green] — interactive agent")
    for key, value in settings.items():
        console.print(f"  [dim]{key}:[/dim] {value}")
    console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    async def confirm_callback(call_id: str, tool: str, arguments: dict) -> bool:
        summary = json.dumps(arguments, ensure_ascii=False)[:200]
        return await _prompt_yes_no(f"Allow [bold]{tool}[/bold] {summary}? [y/n] ")

    async def plan_callback(plan_text: str) -> bool:
        return await _prompt_yes_no("Approve plan? [y/n] ")

    async def save_transcript(messages: list[ChatMessage]) -> None:
        if not messages:
            return
        chat = chat_store.get(session_chat.id)
        if chat is None:
            return
        chat.messages.extend(messages)
        chat_store.save(chat)

    async def repl() -> None:
        nonlocal session_chat, agent
        async with _agent_tool_context(cfg, profile):
            loop = asyncio.get_running_loop()
            while True:
                try:
                    user_input = await loop.run_in_executor(None, input, ">>> ")
                except EOFError:
                    console.print("\n[yellow]Exiting.[/yellow]")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command = user_input[1:].strip()
                    if command in ("quit", "exit"):
                        console.print("[yellow]Goodbye.[/yellow]")
                        break
                    if command == "help":
                        console.print(
                            "[bold]Commands:[/bold]\n"
                            "  /help              show this message\n"
                            "  /quit, /exit       leave the REPL\n"
                            "  /auto              toggle auto-confirm\n"
                            "  /plan              toggle plan mode\n"
                            "  /model <id>        change model\n"
                            "  /root <path>       change project root\n"
                            "  /folder <name>     move transcript to folder\n"
                            "  /max-turns <n>     change max turns per message\n"
                            "  /clear             start a new session chat"
                        )
                        continue
                    if command == "auto":
                        agent.auto_confirm = not agent.auto_confirm
                        console.print(f"[dim]auto_confirm = {agent.auto_confirm}[/dim]")
                        continue
                    if command == "plan":
                        agent.plan_mode = not agent.plan_mode
                        console.print(f"[dim]plan_mode = {agent.plan_mode}[/dim]")
                        continue
                    if command == "clear":
                        await agent.close()
                        session_chat = chat_store.create(
                            str(uuid.uuid4()), "CLI session", folder=folder
                        )
                        agent = InteractiveAgent(
                            model=agent.model,
                            base_url=base_url,
                            root=agent.root,
                            auto_confirm=agent.auto_confirm,
                            plan_mode=agent.plan_mode,
                            max_turns=agent.max_turns,
                            chat_store=chat_store,
                            checkpoint_store=checkpoint_store,
                            run_id=str(uuid.uuid4()),
                        )
                        console.print("[dim]Started a new session chat.[/dim]")
                        continue
                    if command.startswith("model "):
                        agent.model = command[6:].strip() or agent.model
                        console.print(f"[dim]model = {agent.model}[/dim]")
                        continue
                    if command.startswith("root "):
                        new_root = Path(command[5:].strip()).expanduser().resolve()
                        agent.root = new_root
                        console.print(f"[dim]root = {agent.root}[/dim]")
                        continue
                    if command.startswith("folder "):
                        new_folder = command[7:].strip()
                        chat = chat_store.get(session_chat.id)
                        if chat is not None:
                            chat.folder = new_folder
                            chat_store.save(chat)
                        console.print(f"[dim]folder = {new_folder}[/dim]")
                        continue
                    if command.startswith("max-turns "):
                        try:
                            agent.max_turns = int(command[10:].strip())
                            console.print(f"[dim]max_turns = {agent.max_turns}[/dim]")
                        except ValueError:
                            console.print("[red]max-turns requires an integer[/red]")
                        continue
                    console.print(f"[red]Unknown command: /{command}[/red]")
                    continue

                turn_messages: list[ChatMessage] = [ChatMessage(role="user", content=user_input)]
                async for event in agent.chat(
                    user_input,
                    confirm_callback=confirm_callback,
                    plan_callback=plan_callback,
                ):
                    _render_agent_event(event)
                    if event.get("type") == "assistant" and event.get("content"):
                        turn_messages.append(
                            ChatMessage(role="assistant", content=event["content"])
                        )
                    elif event.get("type") == "tool_result":
                        name = event.get("name", "tool")
                        content = event.get("content", "")
                        turn_messages.append(
                            ChatMessage(role="tool", content=f"{name}:\n{content}")
                        )

                await save_transcript(turn_messages)

            await agent.close()

    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted.[/yellow]")


# ===========================================================================
# agent-tui (arcllama)
# ===========================================================================


@cli.command("agent-tui")
@click.option("--model", "-m", default=None, help="Model id to use (default: first available).")
@click.option("--root", "-r", default=None, help="Project root (default: current directory).")
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
@click.pass_context
def agent_tui_cmd(
    ctx: click.Context,
    model: str | None,
    root: str | None,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Launch the interactive arcllama agent TUI."""
    cfg = load_config(ctx.obj["config_path"])
    try:
        # Imported here, not at module scope: agent_tui raises SystemExit at
        # import time when textual is missing, and textual is an optional
        # extra. An eager import takes down every other command with it.
        from arc_llama.agent_tui import run_agent_tui

        run_agent_tui(
            base_url=base_url,
            model=model,
            root=root,
            folder=folder,
            profile=profile,
            config=cfg,
        )
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@click.command(name="arcllama", add_help_option=True)
@click.option("--model", "-m", default=None, help="Model id to use (default: first available).")
@click.option("--root", "-r", default=None, help="Project root (default: current directory).")
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
def arcllama_main(
    model: str | None,
    root: str | None,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Entry point for the `arcllama` command."""
    if not _experimental_agent_enabled():
        console.print(
            "[red]The arcllama agent TUI is experimental. "
            "Set ARC_LLAMA_EXPERIMENTAL_AGENT=1 to enable it.[/red]"
        )
        sys.exit(1)
    cfg = load_config()
    try:
        from arc_llama.agent_tui import run_agent_tui

        run_agent_tui(
            base_url=base_url,
            model=model,
            root=root,
            folder=folder,
            profile=profile,
            config=cfg,
        )
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


# ===========================================================================
# tui
# ===========================================================================


@cli.command("tui")
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Base URL of an arc-llama serve instance (default: http://HOST:PORT from config).",
)
@click.pass_context
def tui_cmd(ctx: click.Context, server_url: str | None) -> None:
    """Launch the terminal UI against a running `arc-llama serve`."""
    if server_url is None:
        cfg = load_config(ctx.obj["config_path"])
        server_url = f"http://{cfg.server.host}:{cfg.server.port}"
    try:
        from arc_llama.tui import run_tui
    except SystemExit as e:
        # The tui module raises SystemExit if textual is missing; surface its message.
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    run_tui(server_url)


def _experimental_agent_enabled() -> bool:
    """Return True if the experimental coding-agent commands should be exposed."""
    return os.environ.get("ARC_LLAMA_EXPERIMENTAL_AGENT", "").lower() in ("1", "true", "yes")


# Hide the experimental agent commands unless the user explicitly opts in.
if not _experimental_agent_enabled():
    for _experimental_agent_cmd in ("agent", "code", "agent-tui"):
        cli.commands.pop(_experimental_agent_cmd, None)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
