"""CLI command: install-runtime."""
from __future__ import annotations

import platform as _platform
from pathlib import Path

import click
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)

from arc_llama.config import load_config
from arc_llama.runtime import RuntimeInstallError, install_runtime

from .common import console


@click.command("install-runtime")
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
def install_runtime_cmd(
    ctx: click.Context,
    backend: str,
    runtime_version: str,
    dest: Path | None,
    set_default: bool,
    force: bool,
) -> None:
    """Download a prebuilt llama-server so you can skip building llama.cpp."""
    cfg = load_config(ctx.obj["config_path"])
    console.print(
        f"[bold]Fetching {backend} llama-server[/bold] "
        f"(llama.cpp {runtime_version}) ..."
    )

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
        raise click.exceptions.Exit(1) from e
    except Exception as e:
        console.print(f"[red]install-runtime failed:[/red] {e}")
        raise click.exceptions.Exit(1) from e

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


__all__ = ["install_runtime_cmd"]
