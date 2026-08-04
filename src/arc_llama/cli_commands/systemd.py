"""CLI command: systemd."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from arc_llama.cli_commands.common import _IS_WINDOWS, console


@click.command("systemd")
@click.option("--service-name", default="arc-llama.service")
@click.option("--description", default="arc-llama OpenAI-compatible router")
@click.option("--write", is_flag=True, help="Write the unit to ~/.config/systemd/user/")
def systemd_unit_cmd(service_name: str, description: str, write: bool) -> None:
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


__all__ = ["systemd_unit_cmd"]
