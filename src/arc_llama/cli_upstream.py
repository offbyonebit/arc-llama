"""CLI commands for managing OpenAI-compatible upstream endpoints."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from arc_llama.config import Config, UpstreamConfig, load_config


def register_upstream_commands(
    cli: click.Group,
    *,
    console: Console,
    save_or_die: Callable[[Config, Path], None],
) -> None:
    """Register upstream commands on the main CLI group."""

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
        if not url.startswith(("http://", "https://")):
            console.print(f"[red]URL must start with http:// or https://: {url}[/red]")
            sys.exit(1)
        existing = next((upstream for upstream in cfg.upstreams if upstream.name == name), None)
        if existing is not None:
            console.print(f"[yellow]Upstream '{name}' already exists. Remove it first.[/yellow]")
            sys.exit(1)

        cfg.upstreams.append(UpstreamConfig(name=name, url=url.rstrip("/")))
        save_or_die(cfg, cfg_path)
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
        for upstream in cfg.upstreams:
            table.add_row(upstream.name, upstream.url)
        console.print(table)

    @upstream_group.command("remove")
    @click.argument("name")
    @click.pass_context
    def upstream_remove(ctx: click.Context, name: str) -> None:
        """Remove an upstream endpoint."""
        cfg_path: Path = ctx.obj["config_path"]
        cfg = load_config(cfg_path)
        before = len(cfg.upstreams)
        cfg.upstreams = [upstream for upstream in cfg.upstreams if upstream.name != name]
        if len(cfg.upstreams) == before:
            console.print(f"[yellow]No upstream named {name!r}.[/yellow]")
            sys.exit(1)
        save_or_die(cfg, cfg_path)
        console.print(f"[green]Removed upstream '{name}'.[/green]")
