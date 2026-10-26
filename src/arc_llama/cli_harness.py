"""`arc-llama install-harness`: install OpenCode or Hermes for Work."""

from __future__ import annotations

from typing import Any

import click
from rich.console import Console


def register_harness_commands(cli: click.Group, *, console: Console, load_config: Any) -> None:
    @cli.command("install-harness")
    @click.argument("name", type=click.Choice(["opencode", "hermes"]))
    @click.option("--latest", is_flag=True, help="Install the newest upstream release instead of the tested version.")
    @click.option("--remove", "remove_version", metavar="VERSION", help="Remove a version Arc Llama installed.")
    @click.option("--list", "list_only", is_flag=True, help="List versions Arc Llama installed.")
    @click.pass_context
    def install_harness_cmd(ctx: click.Context, name: str, latest: bool, remove_version: str | None,
                            list_only: bool) -> None:
        """Download and select a Work harness (OpenCode or Hermes Agent).

        Installs go under Arc Llama's state directory and are verified by
        SHA-256 before use. Your own OpenCode or Hermes installation is not
        changed; this one is selected for Work in the config.
        """
        from arc_llama import harness_install as hi

        config_path = ctx.obj["config_path"]
        cfg = load_config(config_path)
        if list_only:
            items = hi.managed_installs(cfg, name)
            if not items:
                console.print(f"No {hi.LABELS[name]} installed by Arc Llama.")
            selected = getattr(cfg.agent, f"{name}_bin")
            for item in items:
                mark = "*" if item["path"] == selected else " "
                tested = "tested" if item.get("tested") else "untested"
                console.print(f"{mark} {item['version']}  ({tested})  {item['path']}")
            return
        if remove_version:
            if not hi.remove_harness(cfg, name, remove_version, config_path=config_path):
                raise click.ClickException(f"{hi.LABELS[name]} {remove_version} was not installed by Arc Llama.")
            console.print(f"Removed {hi.LABELS[name]} {remove_version}.")
            return

        offer = hi.offer(cfg, name)
        if not offer["installable"]:
            raise click.ClickException(f"{hi.LABELS[name]} cannot be installed on this platform by Arc Llama.")
        version = "latest" if latest else "pinned"
        console.print(
            f"Installing {hi.LABELS[name]} {'(latest, untested)' if latest else offer['version']} "
            f"from {offer['source']} ({offer['license']}, {offer['size_note']}) into {offer['location']}"
        )
        last = {"step": ""}

        def progress(step: str, done: int | None, total: int | None) -> None:
            if step != last["step"]:
                console.print(f"  {step}")
                last["step"] = step

        try:
            result = hi.install_harness(cfg, name, version=version, config_path=config_path, progress=progress)
        except hi.HarnessInstallError as exc:
            raise click.ClickException(str(exc)) from exc
        note = "" if result.tested else " (untested with this Arc Llama version)"
        console.print(f"[green]Installed {hi.LABELS[name]} {result.version}{note}[/green]: {result.binary}")
        console.print("Work uses it now. Restart `arc-llama serve` if it is already running.")
