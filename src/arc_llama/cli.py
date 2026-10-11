"""arc-llama CLI.

Top-level commands:

  arc-llama run        One-command setup: detect, runtime, model, fit, serve.
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

import json
import logging
import os
import shutil
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import click
import httpx
from rich.console import Console
from rich.table import Table

from arc_llama import __version__
from arc_llama import benchmark as benchmark_mod
from arc_llama.arch import Arch, Backend, aot_arch_for
from arc_llama.binary import detect_backends, detect_llama_server_backend
from arc_llama.cli_agent import register_agent_commands
from arc_llama.cli_diagnostics import register_diagnostic_commands
from arc_llama.cli_models import register_model_commands
from arc_llama.cli_performance import register_performance_commands
from arc_llama.cli_recipes import register_recipe_commands
from arc_llama.cli_runtime import register_runtime_commands
from arc_llama.cli_support import register_support_commands
from arc_llama.cli_upstream import register_upstream_commands
from arc_llama.config import (
    Config,
    ModelConfig,
    default_config_path,
    init_config_from_detection,
    load_config,
)
from arc_llama.detect import DetectedGPU, detect_gpus
from arc_llama.models import (
    add_local_model,
    download_from_hf,
    parse_hf_spec,
)
from arc_llama.platform_checks import (
    rebar_likely_enabled,
)

_IS_WINDOWS = sys.platform == "win32"


def _configure_windows_stdio() -> None:
    """Keep Rich diagnostics printable on legacy Windows consoles."""
    if not _IS_WINDOWS:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_configure_windows_stdio()
console = Console()


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


def _configured_runtime(cfg: Config) -> Path | None:
    """Resolve the configured llama-server without executing it."""
    configured = cfg.paths.llama_server
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(configured)
    return Path(found).resolve() if found else None


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
    """The easiest efficient path from an Intel Arc GPU to local inference."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path or default_config_path()


_diagnostic_command_exports = register_diagnostic_commands(
    cli,
    console=console,
    load_config=lambda path: load_config(path),
    detect_gpus=lambda **kwargs: detect_gpus(**kwargs),
    is_windows=lambda: _IS_WINDOWS,
)
_doctor_marker = _diagnostic_command_exports["_doctor_marker"]
doctor = _diagnostic_command_exports["doctor"]
support_bundle_cmd = _diagnostic_command_exports["support_bundle_cmd"]

_do_scan: Any
_print_gpu_table: Any

_model_command_exports = register_model_commands(
    cli,
    console=console,
    load_config=lambda path: load_config(path),
    save_or_die=lambda cfg, path: _save_or_die(cfg, path),
    add_local_model=lambda *args, **kwargs: add_local_model(*args, **kwargs),
    download_from_hf=lambda *args, **kwargs: download_from_hf(*args, **kwargs),
    detect_gpus=lambda: detect_gpus(),
    do_scan=lambda cfg, paths: _do_scan(cfg, paths),
    print_gpu_table=lambda gpus: _print_gpu_table(gpus),
    httpx_module=httpx,
)
gpus_cmd = _model_command_exports["gpus_cmd"]
_print_gpu_table = _model_command_exports["_print_gpu_table"]
list_models = _model_command_exports["list_models"]
add = _model_command_exports["add"]
_slugify_for_name = _model_command_exports["_slugify_for_name"]
_persist_scan_roots = _model_command_exports["_persist_scan_roots"]
_do_scan = _model_command_exports["_do_scan"]
scan_cmd = _model_command_exports["scan_cmd"]
remove = _model_command_exports["remove"]


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
                "[yellow]No Intel GPUs detected. Check the Intel graphics driver "
                "and Windows Device Manager, then run `arc-llama doctor`.[/yellow]"
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


def _bootstrap_run_config(config_path: Path) -> Config:
    """Load a usable config, detecting Arc GPUs when first-run state is absent."""
    created = not config_path.exists()
    cfg = load_config(config_path)
    if cfg.gpus:
        return cfg

    console.print("[bold blue]1/4[/bold blue] Detecting Intel Arc hardware ...")
    detected = detect_gpus()
    if not detected:
        raise click.ClickException(
            "No Intel Arc GPU was detected. Run `arc-llama doctor` for driver, "
            "permissions, and ReBAR guidance."
        )
    detected_cfg = init_config_from_detection(detected, llama_server_path=None)
    cfg.gpus = detected_cfg.gpus
    if created:
        # Retain Config's platform-aware default paths while using the detected
        # GPU set. Existing config files keep every user setting unchanged.
        cfg.paths = detected_cfg.paths
    _save_or_die(cfg, config_path)
    names = ", ".join(g.name or g.pci_slot for g in cfg.gpus if g.enabled)
    console.print(f"  [green]ready[/green] {names} · config {config_path}")
    return cfg


def _run_backend(
    requested: str | None,
    available: set[Backend],
) -> tuple[str, bool]:
    """Choose a backend for the polished path and report explicit selection."""
    if requested is not None:
        return requested, True
    if Backend.SYCL in available:
        return Backend.SYCL.value, False
    if Backend.VULKAN in available:
        return Backend.VULKAN.value, False
    # A fresh or incomplete machine gets the dependency-light path. Users can
    # opt into SYCL explicitly; an existing recognised SYCL install is kept.
    return Backend.VULKAN.value, False


def _backend_choice_message(requested: str | None, available: set[Backend], selected: str) -> str:
    """Explain the automatic backend choice in one short line."""
    if requested is not None:
        return f"Backend selected explicitly: {selected}."
    if Backend.SYCL in available:
        if Backend.VULKAN in available:
            return "Backend auto-selected: SYCL (Vulkan is also available; use --backend vulkan to override)."
        return "Backend auto-selected: SYCL (the configured runtime advertises SYCL)."
    if Backend.VULKAN in available:
        return "Backend auto-selected: Vulkan (the configured runtime advertises Vulkan)."
    return "Backend selected: Vulkan (portable fallback; a runtime will be installed if needed)."


def _ensure_run_runtime(
    cfg: Config,
    config_path: Path,
    *,
    current: Path | None,
    available: set[Backend],
    backend: str,
    backend_explicit: bool,
    version: str,
    may_install: bool,
) -> Path:
    """Resolve or install a compatible runtime for ``arc-llama run``."""
    if current is not None:
        if Backend(backend) in available or (not available and not backend_explicit):
            changed = cfg.paths.llama_server != str(current)
            cfg.paths.llama_server = str(current)
            for gpu in cfg.gpus:
                if gpu.enabled and gpu.backend != backend:
                    gpu.backend = backend
                    changed = True
            if changed:
                _save_or_die(cfg, config_path)
            console.print(
                f"[bold blue]2/4[/bold blue] Runtime [green]ready[/green] · {backend} · {current}"
            )
            return current

    if not may_install:
        detail = "not installed" if current is None else f"does not provide {backend}"
        raise click.ClickException(
            f"The configured llama-server is {detail}. Remove --no-install-runtime "
            f"or run `arc-llama install-runtime --backend {backend}`."
        )

    console.print(f"[bold blue]2/4[/bold blue] Installing verified {backend} llama-server ...")
    from rich.progress import BarColumn, DownloadColumn, Progress, TaskProgressColumn

    from arc_llama.runtime import RuntimeInstallError, install_runtime

    progress = Progress(BarColumn(), DownloadColumn(), TaskProgressColumn(), console=console)
    state: dict[str, Any] = {"task_id": None}

    def on_progress(done: int, total: int) -> None:
        if state["task_id"] is None:
            state["task_id"] = progress.add_task("runtime", total=total or 1)
        progress.update(state["task_id"], completed=done)

    try:
        with progress:
            result = install_runtime(
                backend=backend,
                version=version,
                cfg=cfg,
                set_default=True,
                config_path=config_path,
                on_progress=on_progress,
            )
    except RuntimeInstallError as exc:
        raise click.ClickException(f"Runtime installation failed: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Runtime installation failed: {exc}") from exc
    console.print(f"  [green]verified[/green] llama.cpp {result.tag} · {result.binary_path}")
    return result.binary_path


def _run_gpu(cfg: Config, requested: str | None) -> str:
    if requested is not None:
        gpu = cfg.find_gpu(requested)
        if gpu is None or not gpu.enabled:
            raise click.ClickException(f"GPU {requested!r} is not enabled in the config.")
        return gpu.pci_slot
    enabled = next((gpu for gpu in cfg.gpus if gpu.enabled), None)
    if enabled is None:
        raise click.ClickException("No enabled Intel GPU is configured.")
    return enabled.pci_slot


def _existing_model_for_path(cfg: Config, path: Path) -> ModelConfig | None:
    resolved = path.resolve()
    for model in cfg.models:
        try:
            if Path(model.path).resolve() == resolved:
                return model
        except OSError:
            continue
    return None


def _choose_registered_model(models: list[ModelConfig]) -> ModelConfig | None:
    """Offer an interactive numbered picker, or return None for automation."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    console.print("[bold]Available models[/bold]")
    for index, model in enumerate(models, start=1):
        path = Path(model.path).expanduser()
        try:
            size = f" · {path.stat().st_size / (1024**3):.1f} GiB"
        except OSError:
            size = ""
        console.print(f"  {index}. {model.display_name or model.name}{size}")
    choice = click.prompt(
        "Choose a model",
        type=click.IntRange(1, len(models)),
        default=1,
    )
    return models[choice - 1]


def _validate_run_source(cfg: Config, source: str | None) -> None:
    """Report mistyped local files before attempting a runtime download."""
    if source is None or cfg.find_model(source) is not None:
        return
    path = Path(source).expanduser()
    looks_local = (path.is_absolute() or bool(PureWindowsPath(source).drive)
                   or source.startswith(("./", "../", "~", ".\\", "..\\"))
                   or (source.lower().endswith(".gguf") and ":" not in source))
    if looks_local and not path.is_file():
        raise click.ClickException(
            f"Local GGUF file not found or not a regular file: {source}. "
            "Check the path and quote it if it contains spaces. "
            "For a download, use a Hugging Face spec such as org/repo:Q4_K_M."
        )


def _prepare_run_model(
    cfg: Config,
    config_path: Path,
    *,
    source: str | None,
    name: str | None,
    gpu_pci_slot: str,
    hf_token: str | None,
) -> ModelConfig:
    """Resolve an existing model, local GGUF, HF spec, or discovered singleton."""
    if source is None:
        added = _do_scan(cfg, [])
        if added:
            _save_or_die(cfg, config_path)
        if len(cfg.models) == 1:
            model = cfg.models[0]
            console.print(f"[bold blue]3/4[/bold blue] Model [green]ready[/green] · {model.name}")
            return model
        if not cfg.models:
            raise click.ClickException(
                "No GGUF model was found. Pass a local file or Hugging Face spec, for "
                "example `arc-llama run /models/qwen.gguf` or "
                "`arc-llama run unsloth/Qwen3-8B-GGUF:Q4_K_M`."
            )
        selected = _choose_registered_model(cfg.models)
        if selected is not None:
            console.print(
                f"[bold blue]3/4[/bold blue] Model [green]ready[/green] · {selected.name}"
            )
            return selected
        choices = ", ".join(model.name for model in cfg.models[:8])
        raise click.ClickException(f"More than one model is registered; choose one: {choices}")

    existing = cfg.find_model(source)
    local = Path(source).expanduser()
    if existing is not None and not local.exists():
        console.print(f"[bold blue]3/4[/bold blue] Model [green]ready[/green] · {existing.name}")
        return existing

    if local.exists():
        path = local.resolve()
        derived_name = name or _slugify_for_name(path.parent.name, path.name)
    elif "/" in source:
        try:
            spec = parse_hf_spec(source)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        target_dir = Path(cfg.paths.models_dir).expanduser() / spec.repo.split("/")[-1]
        console.print(f"[bold blue]3/4[/bold blue] Downloading {spec.repo} → {target_dir}")
        try:
            path = download_from_hf(spec, target_dir=target_dir, token=hf_token)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        derived_name = name or _slugify_for_name(target_dir.name, path.name)
    else:
        raise click.ClickException(
            f"Unknown model {source!r}. Pass a registered name, local GGUF path, "
            "or Hugging Face `org/repo:Q4_K_M` spec."
        )

    already = _existing_model_for_path(cfg, path)
    if already is not None:
        console.print(f"[bold blue]3/4[/bold blue] Model [green]ready[/green] · {already.name}")
        return already
    try:
        model = add_local_model(
            cfg,
            name=derived_name,
            path=str(path),
            gpu_pci_slot=gpu_pci_slot,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    _save_or_die(cfg, config_path)
    console.print(f"  [green]registered[/green] {model.name} · {path.name}")
    return model


def _print_run_readiness(cfg: Config, model: ModelConfig) -> bool:
    """Print the launch contract and return whether the model is estimated to fit."""
    gpu = cfg.find_gpu(model.gpu_pci_slot)
    recipe = model.launch_recipe()
    estimate: int | None = None
    try:
        from arc_llama.router import _estimate_model_vram_mb, estimate_model_vram_quick_mb

        estimate = estimate_model_vram_quick_mb(model)
        if estimate is None:
            estimate = _estimate_model_vram_mb(model)
    except Exception as exc:  # noqa: BLE001 - readiness is useful even without an estimate
        logging.getLogger("arc_llama.cli").debug("VRAM readiness estimate failed: %s", exc)

    try:
        file_gib = Path(model.path).stat().st_size / (1024**3)
    except OSError as exc:
        raise click.ClickException(f"Model file is not readable: {model.path}") from exc
    kv = (
        recipe.cache_type_k.value
        if recipe.cache_type_k == recipe.cache_type_v
        else f"{recipe.cache_type_k.value}/{recipe.cache_type_v.value}"
    )
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Model", f"{model.display_name or model.name} ({file_gib:.1f} GiB)")
    table.add_row("GPU", (gpu.name or gpu.pci_slot) if gpu is not None else model.gpu_pci_slot)
    table.add_row("Recipe", f"ctx {recipe.ctx:,} · KV {kv} · backend {gpu.backend if gpu else '?'}")
    fits = True
    if estimate is not None and gpu is not None and gpu.vram_mb:
        headroom = gpu.vram_mb - estimate
        fits = headroom >= 0
        if fits and headroom >= max(768, int(gpu.vram_mb * 0.08)):
            fit_text = f"[green]comfortable[/green] · {estimate:,}/{gpu.vram_mb:,} MiB"
        elif fits:
            fit_text = f"[yellow]tight[/yellow] · {estimate:,}/{gpu.vram_mb:,} MiB"
        else:
            fit_text = f"[red]too large[/red] · {estimate:,}/{gpu.vram_mb:,} MiB"
        table.add_row("Estimated fit", fit_text)
    else:
        table.add_row("Estimated fit", "[yellow]unknown; llama-server will verify at load[/yellow]")
    console.print("[bold blue]4/4[/bold blue] Launch plan")
    console.print(table)
    return fits


@cli.command("run")
@click.argument("source", required=False)
@click.option("--name", default=None, help="Name to register a new model under.")
@click.option("--gpu", "gpu_pci_slot", default=None, help="Enabled GPU PCI slot to use.")
@click.option(
    "--backend",
    type=click.Choice([Backend.VULKAN.value, Backend.SYCL.value]),
    default=None,
    help="Runtime backend. Fresh installs default to portable Vulkan.",
)
@click.option(
    "--runtime-version",
    default="latest",
    show_default=True,
    help="llama.cpp release tag used when a runtime must be installed.",
)
@click.option(
    "--install-runtime/--no-install-runtime",
    default=True,
    help="Automatically install a compatible verified runtime when needed.",
)
@click.option("--hf-token", default=None, help="Hugging Face token for gated repositories.")
@click.option("--host", default=None, help="Override the OpenAI server host.")
@click.option("--port", type=int, default=None, help="Override the OpenAI server port.")
@click.option(
    "--auto-tune/--no-auto-tune",
    default=None,
    help="Override background tuning for this run.",
)
@click.option(
    "--setup-only",
    is_flag=True,
    help="Prepare and validate everything, print the launch plan, then exit.",
)
@click.pass_context
def run_cmd(
    ctx: click.Context,
    source: str | None,
    name: str | None,
    gpu_pci_slot: str | None,
    backend: str | None,
    runtime_version: str,
    install_runtime: bool,
    hf_token: str | None,
    host: str | None,
    port: int | None,
    auto_tune: bool | None,
    setup_only: bool,
) -> None:
    """Go from an Arc GPU and MODEL/GGUF/HF spec to a ready inference API.

    With no SOURCE, uses the only registered/discovered model. This command
    composes first-run detection, verified runtime installation, model
    registration, fit validation, and ``serve``; the lower-level commands
    remain available for explicit control.
    """
    config_path: Path = ctx.obj["config_path"]
    cfg = _bootstrap_run_config(config_path)
    _validate_run_source(cfg, source)
    current_runtime = _configured_runtime(cfg)
    available_backends = detect_backends(current_runtime) if current_runtime is not None else set()
    selected_backend, backend_explicit = _run_backend(backend, available_backends)
    console.print(
        f"  [dim]{_backend_choice_message(backend, available_backends, selected_backend)}[/dim]"
    )
    _ensure_run_runtime(
        cfg,
        config_path,
        current=current_runtime,
        available=available_backends,
        backend=selected_backend,
        backend_explicit=backend_explicit,
        version=runtime_version,
        may_install=install_runtime,
    )
    selected_gpu = _run_gpu(cfg, gpu_pci_slot)
    model = _prepare_run_model(
        cfg,
        config_path,
        source=source,
        name=name,
        gpu_pci_slot=selected_gpu,
        hf_token=hf_token,
    )
    if gpu_pci_slot is not None and model.gpu_pci_slot != selected_gpu:
        model.gpu_pci_slot = selected_gpu
        _save_or_die(cfg, config_path)
    model_gpu = cfg.find_gpu(model.gpu_pci_slot)
    if model_gpu is not None and model_gpu.backend != selected_backend:
        model_gpu.backend = selected_backend
        _save_or_die(cfg, config_path)
    if not _print_run_readiness(cfg, model):
        raise click.ClickException(
            "This recipe is estimated to exceed GPU VRAM. Choose a smaller "
            "quantization/model or reduce context before serving."
        )

    serve_host = host or cfg.server.host
    serve_port = port or cfg.server.port
    display_host = "127.0.0.1" if serve_host in ("0.0.0.0", "::") else serve_host
    console.print()
    console.print(
        f"[bold green]Arc inference is ready[/bold green] · model [bold]{model.name}[/bold]"
    )
    console.print(f"  OpenAI base URL  [cyan]http://{display_host}:{serve_port}/v1[/cyan]")
    console.print(f"  Web UI           [cyan]http://{display_host}:{serve_port}/[/cyan]")
    if setup_only:
        console.print(
            "  [dim]Setup-only complete. Run the same command without --setup-only to serve.[/dim]"
        )
        return

    console.print(
        "  [dim]Press Ctrl+C to stop. The first request loads the selected model.[/dim]\n"
    )
    ctx.invoke(
        serve,
        host=host,
        port=port,
        profile=None,
        admin_token=None,
        scan=False,
        auto_tune=auto_tune,
    )


@cli.command("setup")
@click.argument("source", required=False)
@click.option("--name", default=None, help="Name to register a new model under.")
@click.option("--gpu", "gpu_pci_slot", default=None, help="Enabled GPU PCI slot to use.")
@click.option(
    "--backend",
    type=click.Choice([Backend.VULKAN.value, Backend.SYCL.value]),
    default=None,
    help="Runtime backend. Fresh installs default to portable Vulkan.",
)
@click.option(
    "--runtime-version",
    default="latest",
    show_default=True,
    help="llama.cpp release tag used when a runtime must be installed.",
)
@click.option(
    "--install-runtime/--no-install-runtime",
    default=True,
    help="Automatically install a compatible verified runtime when needed.",
)
@click.option("--hf-token", default=None, help="Hugging Face token for gated repositories.")
@click.option("--host", default=None, help="Override the OpenAI server host in the plan.")
@click.option("--port", type=int, default=None, help="Override the OpenAI server port in the plan.")
@click.pass_context
def setup_cmd(
    ctx: click.Context,
    source: str | None,
    name: str | None,
    gpu_pci_slot: str | None,
    backend: str | None,
    runtime_version: str,
    install_runtime: bool,
    hf_token: str | None,
    host: str | None,
    port: int | None,
) -> None:
    """Prepare and validate Arc inference without starting the server.

    This is the beginner-friendly name for ``run --setup-only``. It detects
    the Intel GPU, installs a verified runtime when needed, registers or finds
    the requested model, checks the fit estimate, and prints the URLs that the
    subsequent ``arc-llama run`` command will serve.
    """
    ctx.invoke(
        run_cmd,
        source=source,
        name=name,
        gpu_pci_slot=gpu_pci_slot,
        backend=backend,
        runtime_version=runtime_version,
        install_runtime=install_runtime,
        hf_token=hf_token,
        host=host,
        port=port,
        auto_tune=None,
        setup_only=True,
    )


@cli.command("serve")
@click.option(
    "--host",
    default=None,
    envvar="ARC_LLAMA_HOST",
    help="Override server host (env: ARC_LLAMA_HOST).",
)
@click.option(
    "--port",
    type=int,
    default=None,
    envvar="ARC_LLAMA_PORT",
    help="Override server port (env: ARC_LLAMA_PORT).",
)
@click.option(
    "--profile",
    default=None,
    help="Active integration profile name.",
)
@click.option(
    "--admin-token",
    default=None,
    help="Bearer token required for admin endpoints (also ARC_LLAMA_ADMIN_TOKEN).",
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
@click.option(
    "--lan",
    is_flag=True,
    help="Serve on every network interface. Remote clients need an API key; "
    "one is created if none exists.",
)
@click.option("--yes", "assume_yes", is_flag=True, help="Skip the --lan confirmation.")
@click.pass_context
def serve(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    profile: str | None,
    admin_token: str | None,
    scan: bool,
    auto_tune: bool | None,
    lan: bool,
    assume_yes: bool,
) -> None:
    """Run the OpenAI-compatible router."""
    cfg = load_config(ctx.obj["config_path"])
    if lan:
        if host and host not in ("0.0.0.0", "::"):
            raise click.UsageError("--lan binds every interface; drop --host or use --host alone.")
        host = "0.0.0.0"
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

    if lan:
        _prepare_lan(cfg, port or cfg.server.port, assume_yes)

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
        "Admin endpoints require 'Authorization: Bearer <token>'.[/dim]"
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


def _lan_addresses() -> list[str]:
    """Best-guess LAN IPv4 addresses (no packets are sent)."""
    import socket

    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1: routing lookup only
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = str(info[4][0])
            if not addr.startswith("127.") and addr not in found:
                found.append(addr)
    except OSError:
        pass
    return found


def _prepare_lan(cfg: Config, port: int, assume_yes: bool) -> None:
    """Confirm network exposure and make sure remote clients need a key."""
    from arc_llama.api_keys import ApiKeyStore

    console.print(
        "[bold yellow]LAN mode:[/bold yellow] the API and web UI will be reachable "
        "from other devices on your network."
    )
    if not assume_yes and not click.confirm("Expose arc-llama to your network?", default=False):
        raise click.Abort()
    store = ApiKeyStore.for_state_dir(cfg.paths.state_dir)
    if not store:
        _key, plaintext = store.create("lan")
        console.print(
            "  Created API key [bold]lan[/bold] (shown once, store it now):\n"
            f"    [cyan]{plaintext}[/cyan]"
        )
    else:
        console.print(f"  {len(store)} API key(s) active; manage them with arc-llama keys.")
    for addr in _lan_addresses():
        console.print(f"  From other devices: [cyan]http://{addr}:{port}/chat[/cyan]")
    console.print(
        "  [dim]Remote clients send 'Authorization: Bearer <key>'; this machine "
        "needs no key.[/dim]"
    )


# ===========================================================================
# benchmark / tune
# ===========================================================================


def _server_url_from(ctx: click.Context, server_url: str | None) -> str:
    if server_url:
        return server_url.rstrip("/")
    cfg = load_config(ctx.obj["config_path"])
    return f"http://{cfg.server.host}:{cfg.server.port}"


def _emit_recipe_submission(ctx: click.Context, cfg: Any, report: Any) -> None:
    """Write a community-registry submission file and print the PR link.

    Explicit opt-in (tune --share). Produces a JSON document the registry
    repo's CI can validate and a pre-filled GitHub PR URL — no token, no
    network call from here.
    """
    from arc_llama import workload as workload_mod
    from arc_llama.recipe_share import (
        build_pr_body,
        llama_server_build_identity,
        share_fingerprint,
        submission_document,
        validate_submission,
    )

    m = cfg.find_model(report.model)
    if m is None:
        return
    gpu = cfg.find_gpu(m.gpu_pci_slot)
    model_class = getattr(m, "kv_class", "default") or "default"
    fp = share_fingerprint(
        gpu_arch=(gpu.arch if gpu else "unknown"),
        backend=(gpu.backend if gpu else "sycl"),
        model_class=model_class,
        workload_key=workload_mod.fingerprint_key(cfg.workload),
        tune_schema_version=3,
        vram_mb=(gpu.vram_mb if gpu is not None and gpu.vram_mb is not None else 0),
    )
    provenance = llama_server_build_identity(cfg.paths.llama_server)
    best = report.best
    doc = submission_document(
        fingerprint=fp,
        recipe=report.best_edits,
        prompt_eval_tok_s=(getattr(best, "prompt_eval_tok_s", None) if best else None),
        generation_tok_s=(getattr(best, "generation_tok_s", None) if best else None),
        gpu_name=(gpu.name if gpu else ""),
        arc_llama_version=__version__,
        provenance=provenance,
    )
    problems = validate_submission(doc)
    if problems:
        console.print("[red]Submission failed validation (not shared):[/red]")
        for p in problems:
            console.print(f"  - {p}")
        return

    out_dir = Path(ctx.obj["config_path"]).parent / "shared-recipes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fp[:16]}.json"
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    title = f"Recipe: {(gpu.arch if gpu else '?')}/{model_class} ({fp[:12]})"
    from urllib.parse import quote

    pr_url = (
        "https://github.com/offbyonebit/arc-llama-recipes/compare/new-recipe..."
        f"submit-{fp[:12]}?expand=1&title={quote(title)}&body={quote(build_pr_body(doc))}"
    )
    console.print("[bold green]Recipe ready to share[/bold green]")
    console.print(f"  submission file: {out_path}")
    console.print(f"  open this URL to submit it:\n  {pr_url}")
    console.print(
        "[dim]Submission is manual and opt-in. The file stays on disk until you do.[/dim]"
    )


_support_command_exports = register_support_commands(
    cli, console=console, load_config=lambda path: load_config(path)
)
keys_group = _support_command_exports["keys_group"]
_key_store = _support_command_exports["_key_store"]
keys_create = _support_command_exports["keys_create"]
keys_list = _support_command_exports["keys_list"]
keys_revoke = _support_command_exports["keys_revoke"]
plugin_group = _support_command_exports["plugin_group"]
plugin_new = _support_command_exports["plugin_new"]
plugin_list = _support_command_exports["plugin_list"]


_performance_command_exports = register_performance_commands(
    cli,
    console=console,
    benchmark_mod=benchmark_mod,
    httpx=httpx,
    load_config=lambda path: load_config(path),
    server_url_from=lambda ctx, server_url: _server_url_from(ctx, server_url),
    print_tune_status_table=lambda cfg: _print_tune_status_table(cfg),
    emit_recipe_submission=lambda ctx, cfg, report: _emit_recipe_submission(ctx, cfg, report),
    save_or_die=lambda cfg, path: _save_or_die(cfg, path),
)
benchmark_cmd = _performance_command_exports["benchmark_cmd"]
tune_cmd = _performance_command_exports["tune_cmd"]
speculative_cmd = _performance_command_exports["speculative_cmd"]


# ===========================================================================
# recipes (community registry)
# ===========================================================================


register_recipe_commands(
    cli,
    console=console,
    benchmark_mod=benchmark_mod,
    httpx=httpx,
    load_config=lambda path: load_config(path),
    server_url_from=_server_url_from,
)


_runtime_command_exports = register_runtime_commands(
    cli,
    console=console,
    load_config=lambda path: load_config(path),
    save_or_die=lambda cfg, path: _save_or_die(cfg, path),
)
runtime_group = _runtime_command_exports["runtime_group"]
_installed_runtimes = _runtime_command_exports["_installed_runtimes"]
runtime_list_cmd = _runtime_command_exports["runtime_list_cmd"]
runtime_use_cmd = _runtime_command_exports["runtime_use_cmd"]
runtime_update_cmd = _runtime_command_exports["runtime_update_cmd"]
runtime_rollback_cmd = _runtime_command_exports["runtime_rollback_cmd"]
install_runtime_cmd = _runtime_command_exports["install_runtime_cmd"]


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


register_upstream_commands(cli, console=console, save_or_die=_save_or_die)


# ===========================================================================
# agent commands
# ===========================================================================


register_agent_commands(cli, console=console)


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
