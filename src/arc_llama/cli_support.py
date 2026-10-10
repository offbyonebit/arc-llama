"""Registration for API-key and plugin management commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table


def register_support_commands(
    cli: click.Group, *, console: Console, load_config: Any
) -> dict[str, Any]:
    """Register API-key and plugin support commands."""
    # ===========================================================================
    # keys
    # ===========================================================================


    @cli.group("keys")
    def keys_group() -> None:
        """Manage API keys for remote clients."""


    def _key_store(ctx: click.Context):
        from arc_llama.api_keys import ApiKeyStore

        cfg = load_config(ctx.obj["config_path"])
        return ApiKeyStore.for_state_dir(cfg.paths.state_dir)


    @keys_group.command("create")
    @click.argument("name")
    @click.pass_context
    def keys_create(ctx: click.Context, name: str) -> None:
        """Create a key. It is printed once and only its hash is stored."""
        try:
            key, plaintext = _key_store(ctx).create(name)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="NAME") from exc
        console.print(f"[green]Created key {key.name!r}[/green] (id {key.id}). Store it now:")
        click.echo(plaintext)
        console.print(
            "[dim]Remote clients must now send it as 'Authorization: Bearer <key>'. "
            "A running server picks it up after restart.[/dim]"
        )


    @keys_group.command("list")
    @click.pass_context
    def keys_list(ctx: click.Context) -> None:
        """List keys with their usage."""
        import datetime as _dt

        keys = _key_store(ctx).list()
        if not keys:
            console.print("[dim]No API keys. Remote access is open until you create one.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        for column in ("id", "name", "created", "last used", "requests"):
            table.add_column(column)

        def when(value: object) -> str:
            if not isinstance(value, (int, float)):
                return "never"
            return _dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")

        for key in keys:
            table.add_row(
                str(key["id"]),
                str(key["name"]),
                when(key["created_at"]),
                when(key["last_used_at"]),
                str(key["requests"]),
            )
        console.print(table)


    @keys_group.command("revoke")
    @click.argument("key_id")
    @click.pass_context
    def keys_revoke(ctx: click.Context, key_id: str) -> None:
        """Revoke a key by id."""
        if not _key_store(ctx).revoke(key_id):
            raise click.ClickException(f"No key with id {key_id!r}.")
        console.print(f"[green]Revoked {key_id}.[/green]")



    # ===========================================================================
    # plugin
    # ===========================================================================


    @cli.group("plugin")
    def plugin_group() -> None:
        """Create and inspect arc-llama plugins."""


    @plugin_group.command("new")
    @click.argument("name")
    @click.option(
        "--dir",
        "target_dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Directory to create (default: ./arc-llama-NAME).",
    )
    def plugin_new(name: str, target_dir: Path | None) -> None:
        """Generate a working plugin package with a test."""
        from arc_llama.plugin_scaffold import validate_plugin_name, write_scaffold

        try:
            validate_plugin_name(name)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="NAME") from exc
        target = target_dir or Path.cwd() / f"arc-llama-{name.replace('_', '-')}"
        try:
            written = write_scaffold(name, target)
        except FileExistsError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        console.print(f"[green]Created plugin {name!r} in {target}[/green]")
        for path in written:
            console.print(f"  {path.relative_to(target)}")
        console.print(
            f"\nNext: [bold]cd {target} && pip install -e '.[dev]' && pytest[/bold], "
            "then restart [bold]arc-llama serve[/bold]."
        )


    @plugin_group.command("list")
    @click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
    def plugin_list(as_json: bool) -> None:
        """List installed plugins and whether they load."""
        from arc_llama.plugin_api import PLUGIN_API_VERSION
        from arc_llama.plugins import plugin_health

        health = plugin_health()
        if as_json:
            click.echo(json.dumps({"api_version": PLUGIN_API_VERSION, "plugins": health}, indent=2))
            return
        console.print(f"plugin API {PLUGIN_API_VERSION}")
        if not health:
            console.print("[dim]No plugins installed.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        for column in ("name", "status", "version", "requires API", "error"):
            table.add_column(column)
        for entry in health:
            table.add_row(
                str(entry.get("name", "")),
                str(entry.get("status", "")),
                str(entry.get("version", "")),
                str(entry.get("requires_api", "")),
                str(entry.get("error", "")),
            )
        console.print(table)



    return {
        "keys_group": keys_group,
        "_key_store": _key_store,
        "keys_create": keys_create,
        "keys_list": keys_list,
        "keys_revoke": keys_revoke,
        "plugin_group": plugin_group,
        "plugin_new": plugin_new,
        "plugin_list": plugin_list,
    }
