"""CLI command: serve."""
from __future__ import annotations

import atexit
import os
import signal as _signal
import sys

import click

from arc_llama.cli_bindings import load_config, save_or_die
from arc_llama.models import discover_ggufs, register_discovered

from .common import (
    console,
    print_autotune_banner,
    print_serve_banner,
)


@click.command("serve")
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", type=int, default=None, help="Override server port.")
@click.option(
    "--profile",
    default=None,
    help="Active MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--admin-token",
    default=None,
    help=(
        "Bearer token required for admin endpoints and auto_confirm agent runs "
        "(overrides config; also settable via ARC_LLAMA_ADMIN_TOKEN)."
    ),
)
@click.option(
    "--scan/--no-scan", "scan", default=True,
    help="Auto-register any new GGUFs found in models_dir/scan_paths on startup "
         "(default: on). Drop a model in and it just appears.",
)
@click.option(
    "--auto-tune/--no-auto-tune", "auto_tune", default=None,
    help="Enable background auto-tuning (default: from config tune.auto).",
)
@click.pass_context
def serve_cmd(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    profile: str | None,
    admin_token: str | None,
    scan: bool,
    auto_tune: bool | None,
) -> None:
    """Run the OpenAI-compatible router."""
    cfg = load_config(ctx.obj["config_path"])
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    if profile:
        cfg.agent.profile = profile
    if admin_token:
        cfg.server.admin_token = admin_token
    if auto_tune is not None:
        cfg.tune.auto = auto_tune

    if scan and cfg.gpus:
        try:
            found = discover_ggufs(cfg, extra_paths=[])
            added = register_discovered(cfg, found)
        except Exception as e:  # noqa: BLE001 - discovery must not block serve
            added = []
            console.print(f"[yellow]Startup scan failed: {e}[/yellow]")
        if added:
            save_or_die(cfg, ctx.obj["config_path"])
            console.print(
                f"[green]Auto-registered {len(added)} new model(s):[/green] "
                + ", ".join(m.name for m in added)
            )

    print_autotune_banner(cfg)
    if not cfg.models:
        console.print(
            "[yellow]No models registered yet — drop a GGUF in "
            f"{cfg.paths.models_dir} or run `arc-llama add`.[/yellow]"
        )
    if cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]Binding to {cfg.server.host!r}, not loopback -- make sure "
            "admin_token is set to something you control (it was auto-generated "
            "if you never set one).[/yellow]"
        )
    token_source = (
        "ARC_LLAMA_ADMIN_TOKEN environment variable"
        if os.environ.get("ARC_LLAMA_ADMIN_TOKEN")
        else f"config file ({ctx.obj['config_path']})"
    )
    console.print(
        f"[dim]Admin authentication is enabled via {token_source}. "
        "Admin endpoints and auto_confirm agent runs require "
        "'Authorization: Bearer <token>'.[/dim]"
    )
    print_serve_banner(cfg)
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        sys.exit(1)
    from arc_llama.server import create_app
    app = create_app(cfg, config_path=ctx.obj["config_path"])

    def _shutdown_subprocesses() -> None:
        rt = getattr(app.state, "router", None)
        if rt is None:
            return
        for srv in rt._servers.values():
            try:
                srv.stop()
            except Exception:
                pass

    atexit.register(_shutdown_subprocesses)

    def _on_signal(signum: int, _frame) -> None:  # noqa: ANN001
        _shutdown_subprocesses()
        _signal.signal(signum, _signal.SIG_DFL)
        if sys.platform == "win32":
            sys.exit(0)
        else:
            os.kill(os.getpid(), signum)

    for s in (getattr(_signal, "SIGTERM", None), _signal.SIGINT):
        if s is None:
            continue
        try:
            _signal.signal(s, _on_signal)
        except (OSError, ValueError):
            pass

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


__all__ = ["serve_cmd"]
