"""CLI command group: upstream."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.table import Table

from arc_llama.config import UpstreamConfig, load_config

from .common import console, save_or_die


@click.group("upstream")
def upstream_group() -> None:
    """Manage upstream OpenAI-compatible endpoints."""


@upstream_group.command("add")
@click.argument("name")
@click.argument("url")
@click.pass_context
def upstream_add_cmd(ctx: click.Context, name: str, url: str) -> None:
    """Register an upstream endpoint."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    if not url.startswith(("http://", "https://")):
        console.print(f"[red]URL must start with http:// or https://: {url}[/red]")
        sys.exit(1)
    existing = next((u for u in cfg.upstreams if u.name == name), None)
    if existing is not None:
        console.print(f"[yellow]Upstream '{name}' already exists. Remove it first.[/yellow]")
        sys.exit(1)
    cfg.upstreams.append(UpstreamConfig(name=name, url=url.rstrip("/")))
    save_or_die(cfg, cfg_path)
    console.print(f"[green]Added upstream '{name}' at {url}[/green]")


@upstream_group.command("list")
@click.pass_context
def upstream_list_cmd(ctx: click.Context) -> None:
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
def upstream_remove_cmd(ctx: click.Context, name: str) -> None:
    """Remove an upstream endpoint."""
    cfg_path: Path = ctx.obj["config_path"]
    cfg = load_config(cfg_path)
    before = len(cfg.upstreams)
    cfg.upstreams = [u for u in cfg.upstreams if u.name != name]
    if len(cfg.upstreams) == before:
        console.print(f"[yellow]No upstream named {name!r}.[/yellow]")
        sys.exit(1)
    save_or_die(cfg, cfg_path)
    console.print(f"[green]Removed upstream '{name}'.[/green]")


__all__ = [
    "upstream_group",
    "upstream_add_cmd",
    "upstream_list_cmd",
    "upstream_remove_cmd",
]
