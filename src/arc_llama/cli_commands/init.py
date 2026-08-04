"""CLI command: init."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from arc_llama.binary import detect_llama_server_backend
from arc_llama.cli_bindings import (
    detect_gpus,
    init_config_from_detection,
    print_gpu_table,
    resolve_llama_server,
    save_or_die,
)
from arc_llama.models import discover_ggufs, register_discovered

from .common import console, gather_workload_profile


@click.command()
@click.option(
    "--llama-server",
    type=click.Path(),
    default=None,
    help="Path to your built llama-server binary (SYCL or Vulkan backend).",
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
def init_cmd(
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
    from arc_llama.cli_commands.common import _IS_WINDOWS

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
    server_path = resolve_llama_server(llama_server)
    server_bin = Path(server_path).expanduser()
    runtime_missing = not server_bin.exists()
    if runtime_missing and llama_server is not None:
        console.print(f"[red]llama-server binary not found: {server_path}[/red]")
        sys.exit(3)

    bin_backend = None
    if runtime_missing:
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
            console.print(
                f"[dim]Detected llama-server backend: {bin_backend.value}[/dim]"
            )

    cfg = init_config_from_detection(
        gpus, llama_server_path=None if runtime_missing else server_path
    )
    if bin_backend is not None:
        for gpu_cfg in cfg.gpus:
            gpu_cfg.backend = bin_backend.value
    if scan_paths:
        cfg.paths.scan_paths = list(scan_paths)
    gather_workload_profile(cfg, workload_context, workload_style, workload_priority)
    save_or_die(cfg, config_path)
    console.print(f"[green]Wrote config to {config_path}[/green]")
    print_gpu_table(gpus)
    if scan:
        found = discover_ggufs(cfg, extra_paths=[Path(p) for p in scan_paths])
        if found:
            added = register_discovered(cfg, found)
            save_or_die(cfg, config_path)
            console.print(
                f"[green]Auto-registered {len(added)} model(s):[/green] "
                + ", ".join(m.name for m in added)
            )
        else:
            console.print(
                "[dim]No GGUFs found in scan paths. "
                "Drop one in `paths.models_dir` or pass --scan-path next time.[/dim]"
            )


__all__ = ["init_cmd"]
