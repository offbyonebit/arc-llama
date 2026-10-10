"""Registration for local llama.cpp runtime management commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from arc_llama.config import Config


def register_runtime_commands(
    cli: click.Group, *, console: Console, load_config: Any, save_or_die: Any
) -> dict[str, Any]:
    """Register runtime inspection, selection, installation, and rollback commands."""
    # ===========================================================================
    # install-runtime
    # ===========================================================================


    @cli.group("runtime")
    def runtime_group() -> None:
        """Inspect and select installed llama.cpp runtimes."""


    def _installed_runtimes(cfg: Config) -> list[tuple[str, str, Path, dict[str, Any]]]:
        """Return (tag, backend, binary, marker) for complete local installs."""
        root = Path(cfg.paths.state_dir).expanduser() / "runtime"
        found: list[tuple[str, str, Path, dict[str, Any]]] = []
        if not root.is_dir():
            return found
        for install_dir in sorted(root.glob("llama-*-*")):
            if not install_dir.is_dir():
                continue
            marker_path = install_dir / ".arc-llama-runtime.json"
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                marker = {}
            binary = next(
                (path for path in install_dir.rglob("llama-server*") if path.is_file()),
                None,
            )
            if binary is None:
                continue
            tag = str(marker.get("tag") or install_dir.name.split("-")[1])
            backend = str(marker.get("backend") or install_dir.name.rsplit("-", 1)[-1])
            found.append((tag, backend, binary, marker))
        return found


    @runtime_group.command("list")
    @click.pass_context
    def runtime_list_cmd(ctx: click.Context) -> None:
        """List complete runtimes installed in Arc Llama's state directory."""
        cfg = load_config(ctx.obj["config_path"])
        runtimes = _installed_runtimes(cfg)
        if not runtimes:
            console.print("[yellow]No installed Arc Llama runtimes found.[/yellow]")
            return
        active = str(Path(cfg.paths.llama_server).expanduser().resolve())
        table = Table(title="Installed llama.cpp runtimes")
        table.add_column("Tag")
        table.add_column("Backend")
        table.add_column("Active")
        table.add_column("Path")
        for tag, backend, binary, _marker in runtimes:
            table.add_row(tag, backend, "yes" if binary.resolve() == active else "", str(binary))
        console.print(table)


    @runtime_group.command("use")
    @click.argument("tag")
    @click.option("--backend", type=click.Choice(["vulkan", "sycl"]), default=None)
    @click.pass_context
    def runtime_use_cmd(ctx: click.Context, tag: str, backend: str | None) -> None:
        """Select an already-installed runtime by release tag."""
        cfg_path: Path = ctx.obj["config_path"]
        cfg = load_config(cfg_path)
        candidates = [item for item in _installed_runtimes(cfg) if item[0] == tag]
        if backend is not None:
            candidates = [item for item in candidates if item[1] == backend]
        if not candidates:
            suffix = f"/{backend}" if backend else ""
            raise click.ClickException(f"No installed runtime found for {tag}{suffix}.")
        if len(candidates) > 1:
            choices = ", ".join(item[1] for item in candidates)
            raise click.ClickException(f"Multiple runtimes found for {tag}: {choices}; pass --backend.")
        selected_tag, selected_backend, binary, _marker = candidates[0]
        cfg.paths.llama_server = str(binary)
        for gpu in cfg.gpus:
            if gpu.enabled:
                gpu.backend = selected_backend
        save_or_die(cfg, cfg_path)
        console.print(f"[green]Selected[/green] {selected_tag}/{selected_backend} · {binary}")


    @runtime_group.command("update")
    @click.option(
        "--backend",
        type=click.Choice(["vulkan", "sycl"]),
        default=None,
        help="Backend to fetch (default: the enabled GPU's backend).",
    )
    @click.option(
        "--runtime-version",
        "runtime_version",
        default="latest",
        show_default=True,
        help="llama.cpp release tag (e.g. b10092) or 'latest'.",
    )
    @click.option("--canary-model", default=None, help="Registered model to test with.")
    @click.option(
        "--no-canary",
        is_flag=True,
        help="Skip the trial launch; only check the candidate's flags.",
    )
    @click.option("--dry-run", is_flag=True, help="Install and test, but keep the current runtime.")
    @click.pass_context
    def runtime_update_cmd(
        ctx: click.Context,
        backend: str | None,
        runtime_version: str,
        canary_model: str | None,
        no_canary: bool,
        dry_run: bool,
    ) -> None:
        """Install a newer llama.cpp beside the current one and switch only if it passes.

        The candidate must support every flag your recipes use, then start one
        registered model (the smallest, or --canary-model) and answer a short
        prompt. Otherwise the current runtime stays selected. Stop
        `arc-llama serve` first if it has a model loaded: the canary respects the
        single-resident lock.
        """
        from arc_llama.runtime import RuntimeInstallError
        from arc_llama.runtime_update import update_runtime

        cfg_path: Path = ctx.obj["config_path"]
        cfg = load_config(cfg_path)
        console.print(f"[bold]Checking llama.cpp {runtime_version}[/bold] (current: {cfg.paths.llama_server})")
        try:
            result = update_runtime(
                cfg,
                cfg_path,
                backend=backend,
                version=runtime_version,
                canary_model=canary_model,
                canary=not no_canary,
                dry_run=dry_run,
            )
        except RuntimeInstallError as exc:
            raise click.ClickException(f"download failed: {exc}") from exc
        if result.status == "up_to_date":
            console.print(f"[green]Already on {result.tag}.[/green]")
            return
        console.print(f"  candidate: {result.candidate} ({result.tag})")
        if result.canary is not None:
            marker = "[green]passed[/green]" if result.canary.ok else "[red]failed[/red]"
            timing = f" in {result.canary.seconds:.0f}s" if result.canary.seconds else ""
            console.print(f"  canary:    {marker} on {result.canary.model}{timing}: {result.canary.detail}")
        if result.status == "rejected":
            console.print("[red]Kept the current runtime.[/red] The candidate stays installed:")
            for problem in result.problems:
                console.print(f"  - {problem}")
            sys.exit(1)
        if result.status == "dry_run":
            console.print("[green]Candidate passed.[/green] Dry run: config unchanged.")
            console.print(f"Select it with: [bold]arc-llama runtime use {result.tag}[/bold]")
            return
        console.print(f"[green]Switched to {result.tag}.[/green] Restart arc-llama serve to use it.")
        console.print("Undo with: [bold]arc-llama runtime rollback[/bold]")


    @runtime_group.command("rollback")
    @click.pass_context
    def runtime_rollback_cmd(ctx: click.Context) -> None:
        """Return to the runtime that the last `runtime update` replaced."""
        from arc_llama.runtime_update import rollback_runtime

        cfg_path: Path = ctx.obj["config_path"]
        cfg = load_config(cfg_path)
        restored = rollback_runtime(cfg, cfg_path)
        if restored is None:
            raise click.ClickException("No earlier runtime to roll back to.")
        console.print(f"[green]Rolled back[/green] {restored[0]} -> {restored[1]}")
        console.print("Restart arc-llama serve to use it.")


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



    return {
        "runtime_group": runtime_group,
        "_installed_runtimes": _installed_runtimes,
        "runtime_list_cmd": runtime_list_cmd,
        "runtime_use_cmd": runtime_use_cmd,
        "runtime_update_cmd": runtime_update_cmd,
        "runtime_rollback_cmd": runtime_rollback_cmd,
        "install_runtime_cmd": install_runtime_cmd,
    }
