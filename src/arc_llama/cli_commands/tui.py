"""CLI command: tui."""
from __future__ import annotations

import sys

import click

from arc_llama.cli_bindings import load_config

from .common import console


@click.command("tui")
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
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    run_tui(server_url)


__all__ = ["tui_cmd"]
