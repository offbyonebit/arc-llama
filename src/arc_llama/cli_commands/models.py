"""CLI commands: gpus, list, add, scan, remove."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx
from rich.table import Table

from arc_llama.arch import Backend
from arc_llama.binary import detect_llama_server_backend
from arc_llama.cli_bindings import (
    add_local_model,
    download_from_hf,
    load_config,
    print_gpu_table,
    save_or_die,
    slugify_for_name,
)
from arc_llama.detect import detect_gpus
from arc_llama.models import discover_ggufs, parse_hf_spec, register_discovered

from .common import console


@click.command("gpus")
def gpus_cmd() -> None:
    """List detected Intel GPUs."""
    gpus = detect_gpus()
    if not gpus:
        console.print("[red]No Intel GPUs detected.[/red]")
        sys.exit(2)
    print_gpu_table(gpus)


@click.command("list")
@click.pass_context
def list_models_cmd(ctx: click.Context) -> None:
    """List registered models and which one is currently loaded."""
    cfg = load_config(ctx.obj["config_path"])
    if not cfg.models:
        console.print(
            "[yellow]No models registered. "
            "Run [bold]arc-llama add ...[/bold].[/yellow]"
        )
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
        kv = f"{recipe.get('cache_type_k','f16')}/{recipe.get('cache_type_v','f16')}"
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


@click.command("add")
@click.argument("source")
@click.option("--name", default=None, help="Short name (default: derived from source).")
@click.option(
    "--gpu", "gpu_pci_slot", default=None,
    help="PCI slot of the GPU to bind to (default: first enabled GPU).",
)
@click.option(
    "--backend", "backend", default=None,
    type=click.Choice([Backend.SYCL.value, Backend.VULKAN.value]),
    help="Compute backend for this GPU (default: the GPU's configured backend, usually sycl).",
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
    type=click.Choice([
        "default", "moe_a3b", "qwen3_dense", "qwen3_27b_dense",
        "qwen2_5", "gemma_swa", "phi4", "llama3", "deepseek_r1_distill",
    ]),
    default="default",
    help="KV-class hint, used for VRAM estimation.",
)
@click.option("--alias", "aliases", multiple=True, help="Extra match strings (repeatable).")
@click.option(
    "--spec-type", "spec_type", default=None,
    help="Speculative decoding type (e.g. draft-mtp). Auto-detected for models "
         "with embedded MTP heads or a sidecar draft GGUF.",
)
@click.option(
    "--spec-draft-model", "spec_draft_model", default=None,
    help="Path to a sidecar speculative-draft GGUF (--spec-draft-model). "
         "Auto-detected from a sibling mtp-/draft- file when present.",
)
@click.option(
    "--spec-draft-ngl", "spec_draft_ngl", type=int, default=None,
    help="GPU layers for the draft model (--spec-draft-ngl); default 999.",
)
@click.option(
    "--ubatch-size", "ubatch_size", type=int, default=None,
    help="Ubatch size (-ub). Left unset by default; llama.cpp picks its own.",
)
@click.option(
    "--from-hf", is_flag=True,
    help="Treat SOURCE as a Hugging Face spec (`org/repo` or `org/repo:Q4_K_M`).",
)
@click.option("--hf-token", default=None, help="HF token for gated repos.")
@click.pass_context
def add_cmd(
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
    from_hf: bool,
    hf_token: str | None,
) -> None:
    """Register a model. SOURCE is either a local GGUF path or a HF spec."""
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

    if backend is not None:
        gpu_cfg = cfg.find_gpu(gpu_pci_slot)
        if gpu_cfg is not None:
            gpu_cfg.backend = backend

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
        derived_name = name or slugify_for_name(target_dir.name, path.name)
    else:
        path = local_candidate.resolve()
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            sys.exit(1)
        derived_name = name or slugify_for_name(path.parent.name, path.name)

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

    save_or_die(cfg, cfg_path)
    console.print(f"[green]Registered {mc.name}[/green] on {gpu_pci_slot}, port {mc.port}")


def _do_scan(cfg, extra_paths: list[Path]) -> list:
    found = discover_ggufs(cfg, extra_paths=extra_paths)
    if not found:
        return []
    return register_discovered(cfg, found)


@click.command("scan")
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
        save_or_die(cfg, cfg_path)
    console.print(
        f"[green]Registered {len(added)} new model(s):[/green] "
        + ", ".join(m.name for m in added)
    )
    if not persist:
        console.print("[dim]--no-persist: config NOT saved.[/dim]")


@click.command("remove")
@click.argument("name")
@click.pass_context
def remove_cmd(ctx: click.Context, name: str) -> None:
    """Remove a model from the config (does NOT delete the GGUF)."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.models)
    cfg.models = [m for m in cfg.models if m.name != name]
    if len(cfg.models) == before:
        console.print(f"[yellow]No model named {name!r}.[/yellow]")
        sys.exit(1)
    save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed {name}.[/green]")


__all__ = [
    "gpus_cmd",
    "list_models_cmd",
    "add_cmd",
    "scan_cmd",
    "remove_cmd",
]
