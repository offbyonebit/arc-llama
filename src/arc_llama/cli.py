"""arc-llama CLI.

Top-level commands:

  arc-llama init       Auto-detect GPUs and write an initial config.
  arc-llama doctor     Diagnose the local environment (drivers, oneAPI, perms).
  arc-llama list       List registered models and their state.
  arc-llama gpus       List detected Intel GPUs.
  arc-llama add        Register a model — local file or HF download.
  arc-llama remove     Remove a model from the config.
  arc-llama serve      Run the OpenAI-compatible router.
  arc-llama tui        Launch the terminal UI.
  arc-llama systemd    Print a systemd --user service unit for `arc-llama serve`.

A small static web UI is bundled and served at `/` on the same port as
`arc-llama serve` — open it in a browser for a model-picker + load/stop view.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from arc_llama import __version__
from arc_llama.config import (
    Config,
    default_config_path,
    init_config_from_detection,
    load_config,
)
from arc_llama.detect import detect_gpus, lspci_intel_gpus
from arc_llama.models import (
    add_local_model,
    discover_ggufs,
    download_from_hf,
    parse_hf_spec,
    register_discovered,
)

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
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
    """Find a usable llama-server binary, in order of preference."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates += [
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
    "-c", "--config", "config_path",
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

@cli.command()
@click.option(
    "--llama-server",
    type=click.Path(),
    default=None,
    help="Path to your built llama-server binary (SYCL backend).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
@click.option(
    "--scan/--no-scan", default=True,
    help="After init, walk scan paths for .gguf files and auto-register them (default: on).",
)
@click.option(
    "--scan-path", "scan_paths", multiple=True, type=click.Path(),
    help="Extra directory to walk for GGUFs. Repeatable.",
)
@click.pass_context
def init(
    ctx: click.Context,
    llama_server: str | None,
    force: bool,
    scan: bool,
    scan_paths: tuple[str, ...],
) -> None:
    """Detect GPUs and write a starter config; auto-register any GGUFs found."""
    config_path: Path = ctx.obj["config_path"]
    if config_path.exists() and not force:
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("Use --force to overwrite, or edit it directly.")
        sys.exit(1)
    gpus = detect_gpus()
    if not gpus:
        console.print("[red]No Intel GPUs detected.[/red]")
        console.print("Run [bold]arc-llama doctor[/bold] for a diagnosis.")
        sys.exit(2)
    server_path = _resolve_llama_server(llama_server)
    cfg = init_config_from_detection(gpus, llama_server_path=server_path)
    if scan_paths:
        cfg.paths.scan_paths = list(scan_paths)
    _save_or_die(cfg, config_path)
    console.print(f"[green]Wrote config to {config_path}[/green]")
    _print_gpu_table(gpus)
    if cfg.paths.llama_server == "llama-server":
        console.print(
            "[yellow]llama-server binary not found; set 'paths.llama_server' "
            "in the config or pass --llama-server.[/yellow]"
        )
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

@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Diagnose the local environment."""
    config_path: Path = ctx.obj["config_path"]
    console.print("[bold]arc-llama doctor[/bold]\n")

    # Kernel + driver
    console.print(f"  kernel:        {os.uname().release}")
    has_xe = Path("/sys/module/xe").exists()
    has_i915 = Path("/sys/module/i915").exists()
    console.print(f"  xe driver:     {'loaded' if has_xe else 'not loaded'}")
    console.print(f"  i915 driver:   {'loaded' if has_i915 else 'not loaded'}")

    # GPU detection (enrich=True so clinfo populates VRAM where xe doesn't via sysfs)
    gpus = detect_gpus(enrich=True)
    if gpus:
        console.print(f"\n  detected {len(gpus)} Intel GPU(s):")
        for g in gpus:
            vram = f"{g.vram_gb} GB" if g.vram_gb else "VRAM unknown"
            console.print(
                f"    - {g.name} @ {g.pci_slot}  ({g.arch.value}, "
                f"driver={g.driver or '—'}, {vram})"
            )
            if g.notes:
                for n in g.notes:
                    console.print(f"        note: {n}")
    else:
        console.print("\n  [red]no Intel GPUs detected via sysfs[/red]")
        raw = lspci_intel_gpus()
        if raw:
            console.print("\n  raw lspci output for Intel display devices:")
            for line in raw.splitlines():
                console.print(f"    {line}")
        else:
            console.print("    lspci shows no Intel display devices either.")

    # External tools
    console.print("\n  external tools:")
    for tool in ("clinfo", "sycl-ls", "intel_gpu_top", "nvtop", "lspci"):
        path = shutil.which(tool)
        console.print(f"    {tool:<14} {path or '— missing —'}")

    # Permissions
    console.print("\n  user groups:")
    try:
        out = subprocess.run(["id", "-nG"], capture_output=True, text=True, timeout=2)
        groups = out.stdout.split()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        groups = []
    for needed in ("render", "video"):
        ok = needed in groups
        marker = "[green]ok[/green]" if ok else "[yellow]missing[/yellow]"
        console.print(f"    {needed:<14} {marker}")
    if "render" not in groups or "video" not in groups:
        console.print(
            "    [yellow]→ add yourself with `sudo usermod -aG render,video $USER` "
            "and re-login.[/yellow]"
        )

    # oneAPI
    oneapi_setvars = Path("/opt/intel/oneapi/setvars.sh")
    console.print("\n  oneAPI:")
    if oneapi_setvars.exists():
        console.print(f"    setvars.sh:   {oneapi_setvars}")
    else:
        console.print(
            "    [yellow]/opt/intel/oneapi/setvars.sh missing — install Intel "
            "oneAPI Base Toolkit if you're building llama.cpp from source.[/yellow]"
        )

    # Config
    console.print("\n  config:")
    if config_path.exists():
        console.print(f"    [green]found[/green] at {config_path}")
    else:
        console.print(
            f"    [yellow]missing[/yellow] at {config_path} — run "
            f"[bold]arc-llama init[/bold]."
        )


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
    """List registered models."""
    cfg = load_config(ctx.obj["config_path"])
    if not cfg.models:
        console.print(
            "[yellow]No models registered. "
            "Run [bold]arc-llama add ...[/bold].[/yellow]"
        )
        return
    table = Table(title="Models")
    table.add_column("Name")
    table.add_column("GPU")
    table.add_column("Port")
    table.add_column("ctx")
    table.add_column("KV")
    table.add_column("Spec")
    table.add_column("Path")
    for m in cfg.models:
        r = m.recipe or {}
        kv = f"{r.get('cache_type_k','f16')}/{r.get('cache_type_v','f16')}"
        spec = r.get("spec_type", "—")
        if r.get("ubatch_size"):
            spec += f" (ub={r['ubatch_size']})"
        table.add_row(
            m.name,
            m.gpu_pci_slot,
            str(m.port),
            str(r.get("ctx", "?")),
            kv,
            spec,
            m.path,
        )
    console.print(table)


# ===========================================================================
# add
# ===========================================================================

@cli.command("add")
@click.argument("source")
@click.option("--name", default=None, help="Short name (default: derived from source).")
@click.option(
    "--gpu", "gpu_pci_slot", default=None,
    help="PCI slot of the GPU to bind to (default: first enabled GPU).",
)
@click.option(
    "--port", type=int, default=None,
    help="Backend port for this model's llama-server (default: auto).",
)
@click.option("--ctx", "ctx_override", type=int, default=None, help="Override context length.")
@click.option(
    "--kv", "kv_type", type=click.Choice(["f16", "q8_0", "q5_1", "q4_0"]),
    default=None,
    help="Override KV cache type (applies to both K and V).",
)
@click.option("--display-name", default="", help="Human-friendly name.")
@click.option(
    "--kv-class",
    type=click.Choice(["default", "moe_a3b", "qwen3_27b_dense", "gemma_swa"]),
    default="default",
    help="KV-class hint, used for VRAM estimation.",
)
@click.option("--alias", "aliases", multiple=True, help="Extra match strings (repeatable).")
@click.option(
    "--spec-type", "spec_type", default=None,
    help="Speculative decoding type (e.g. draft-mtp). Auto-detected for MTP models.",
)
@click.option(
    "--ubatch-size", "ubatch_size", type=int, default=None,
    help="Ubatch size (-ub). Auto-set to 8 for MTP models.",
)
@click.option(
    "--from-hf", is_flag=True,
    help="Treat SOURCE as a Hugging Face spec (`org/repo` or `org/repo:Q4_K_M`).",
)
@click.option("--hf-token", default=None, help="HF token for gated repos.")
@click.pass_context
def add(
    ctx: click.Context,
    source: str,
    name: str | None,
    gpu_pci_slot: str | None,
    port: int | None,
    ctx_override: int | None,
    kv_type: str | None,
    display_name: str,
    kv_class: str,
    aliases: tuple[str, ...],
    spec_type: str | None,
    ubatch_size: int | None,
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
    if ubatch_size is not None:
        overrides["ubatch_size"] = ubatch_size

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
    console.print(f"[green]Registered {mc.name}[/green] on {gpu_pci_slot}, port {mc.port}")


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
    "--gpu", "gpu_pci_slot", default=None,
    help="Bind newly discovered models to this PCI slot (default: first enabled GPU).",
)
@click.option(
    "--persist/--no-persist", default=True,
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
        console.print(
            "[yellow]No GGUFs found.[/yellow] Scanned: " + ", ".join(scanned)
        )
        return
    try:
        added = register_discovered(cfg, found, gpu_pci_slot=gpu_pci_slot)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    if not added:
        console.print(
            f"[dim]Found {len(found)} GGUF(s); all already registered.[/dim]"
        )
        return
    if persist:
        _save_or_die(cfg, cfg_path)
    console.print(
        f"[green]Registered {len(added)} new model(s):[/green] "
        + ", ".join(m.name for m in added)
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
# ===========================================================================

@cli.command("serve")
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", type=int, default=None, help="Override server port.")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Run the OpenAI-compatible router."""
    cfg = load_config(ctx.obj["config_path"])
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    if not cfg.models:
        console.print(
            "[yellow]No models registered yet — `arc-llama add` something first.[/yellow]"
        )
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        sys.exit(1)
    from arc_llama.server import create_app
    app = create_app(cfg)

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
        os.kill(os.getpid(), signum)

    for s in (_signal.SIGTERM, _signal.SIGINT):
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
    "--server", "server_url", default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option("--prompt-tokens", type=int, default=512, show_default=True,
              help="Approximate prompt length for the prompt-eval measurement.")
@click.option("--gen-tokens", type=int, default=128, show_default=True,
              help="Tokens to generate for the generation measurement.")
@click.option(
    "--sweep-ctx", default=None,
    help="Comma-separated ctx values to sweep (e.g. 8192,16384,32768). "
         "Each is benchmarked against every --kv value.",
)
@click.option(
    "--kv", "kv_types", multiple=True,
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
    sweep_ctx: str | None,
    kv_types: tuple[str, ...],
    as_json: bool,
) -> None:
    """Measure prompt-eval and generation tok/s for MODEL.

    Requires a running `arc-llama serve` — measurements go through the
    router so they use the exact SYCL env and recipe your requests get.
    """
    import asyncio
    import json as _json

    from arc_llama.benchmark import (
        benchmark_model,
        benchmark_sweep,
        print_result,
        print_sweep_table,
    )

    url = _server_url_from(ctx, server_url)
    cfg = load_config(ctx.obj["config_path"])
    if sweep_ctx:
        try:
            ctx_values = [int(v) for v in sweep_ctx.split(",") if v.strip()]
        except ValueError:
            console.print(f"[red]--sweep-ctx must be comma-separated integers: {sweep_ctx!r}[/red]")
            sys.exit(1)
        kvs = list(kv_types) or ["f16", "q8_0"]
        results = asyncio.run(benchmark_sweep(
            url, model,
            ctx_values=ctx_values, kv_types=kvs,
            prompt_tokens=prompt_tokens, gen_tokens=gen_tokens, cfg=cfg,
        ))
        if as_json:
            click.echo(_json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print_sweep_table(results)
        sys.exit(1 if all(r.error for r in results) else 0)
    result = asyncio.run(benchmark_model(
        url, model, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens, cfg=cfg,
    ))
    if as_json:
        click.echo(_json.dumps(result.to_dict(), indent=2))
    else:
        print_result(result)
    sys.exit(1 if result.error else 0)


@cli.command("tune")
@click.argument("model")
@click.option(
    "--server", "server_url", default=None,
    help="Base URL of a running `arc-llama serve` (default: http://HOST:PORT from config).",
)
@click.option(
    "--target", type=click.Choice(["balanced", "generation", "prompt"]),
    default="balanced", show_default=True,
    help="What to optimise: generation tok/s, prompt-eval tok/s, or both.",
)
@click.option("--prompt-tokens", type=int, default=1024, show_default=True)
@click.option("--gen-tokens", type=int, default=128, show_default=True)
@click.option(
    "--apply/--dry-run", "apply_", default=True,
    help="Write the winning config into the model's recipe (default) or restore the original.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of tables.")
@click.pass_context
def tune_cmd(
    ctx: click.Context,
    model: str,
    server_url: str | None,
    target: str,
    prompt_tokens: int,
    gen_tokens: int,
    apply_: bool,
    as_json: bool,
) -> None:
    """Find the fastest recipe for MODEL by measuring, then persist it.

    Staged sweep over KV cache type, ubatch size, and flash attention —
    roughly 6–9 measured configs, each paying one model reload. Expect
    ~10 minutes on a Battlemage-class card. Requires a running
    `arc-llama serve`.
    """
    import asyncio
    import json as _json
    from dataclasses import asdict

    from arc_llama.tune import print_report, tune_model

    url = _server_url_from(ctx, server_url)
    cfg = load_config(ctx.obj["config_path"])
    report = asyncio.run(tune_model(
        url, model,
        target=target, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
        apply=apply_, cfg=cfg,
    ))
    if as_json:
        click.echo(_json.dumps(asdict(report), indent=2, default=str))
    else:
        print_report(report)
    sys.exit(1 if report.error else 0)


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
# tui
# ===========================================================================

@cli.command("tui")
@click.option(
    "--server", "server_url",
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


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
