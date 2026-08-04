"""arc-llama CLI.

Top-level commands:

  arc-llama init       Auto-detect GPUs and write an initial config.
  arc-llama doctor     Diagnose the local environment (drivers, oneAPI, perms).
  arc-llama list       List registered models and their state.
  arc-llama gpus       List detected Intel GPUs.
  arc-llama add        Register a model — local file or HF download.
  arc-llama remove     Remove a model from the config.
  arc-llama serve      Run the OpenAI-compatible router.
  arc-llama benchmark  Measure prompt-eval / generation tok/s for a model.
  arc-llama tune       Staged greedy autotune; persist the winning recipe.
  arc-llama tui        Launch the server management TUI.
  arc-llama systemd    Print a systemd --user service unit for `arc-llama serve`.

The agent/coding-assistant commands (`agent`, `code`, `agent-tui`) are
experimental and only appear when ARC_LLAMA_EXPERIMENTAL_AGENT=1 is set.

A small static web UI is bundled and served at `/` on the same port as
`arc-llama serve` — open it in a browser for a model-picker + load/stop view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx

from arc_llama import __version__
from arc_llama.cli_commands.common import (
    console,
    experimental_agent_enabled,
    print_gpu_table,
    resolve_llama_server,
    save_or_die,
    setup_logging,
    slugify_for_name,
)
from arc_llama.config import (
    default_config_path,
    init_config_from_detection,
    load_config,
)
from arc_llama.detect import detect_gpus
from arc_llama.models import add_local_model, download_from_hf


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
@click.option(
    "-c", "--config", "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to config.toml (default: $XDG_CONFIG_HOME/arc-llama/config.toml).",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: Path | None) -> None:
    """Plug-and-play llama.cpp runtime for Intel Arc GPUs."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path or default_config_path()


# Legacy re-exports for backward compatibility.
main = cli


@click.command(name="arcllama", add_help_option=True)
@click.option("--model", "-m", default=None, help="Model id to use (default: first available).")
@click.option("--root", "-r", default=None, help="Project root (default: current directory).")
@click.option("--folder", "-f", default="", help="Folder to save the session transcript chat.")
@click.option(
    "--profile",
    default=None,
    help="MCP profile name (overrides agent.profile in config).",
)
@click.option(
    "--base-url",
    default=None,
    help="arc-llama server base URL (default: http://HOST:PORT from config).",
)
def arcllama_main(
    model: str | None,
    root: str | None,
    folder: str,
    profile: str | None,
    base_url: str | None,
) -> None:
    """Entry point for the `arcllama` command."""
    cfg = load_config()
    try:
        from arc_llama.agent_tui import run_agent_tui

        run_agent_tui(
            base_url=base_url,
            model=model,
            root=root,
            folder=folder,
            profile=profile,
            config=cfg,
        )
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


# Re-export private helpers that tests / downstream code patch or import.
_load_config = load_config
_default_config_path = default_config_path
_detect_gpus = detect_gpus
_init_config_from_detection = init_config_from_detection  # noqa: F841
_resolve_llama_server = resolve_llama_server  # noqa: F841
_print_gpu_table = print_gpu_table  # noqa: F841
_save_or_die = save_or_die  # noqa: F841
_slugify_for_name = slugify_for_name  # noqa: F841
_download_from_hf = download_from_hf  # noqa: F841
_add_local_model = add_local_model  # noqa: F841
_experimental_agent_enabled = experimental_agent_enabled  # noqa: F841
httpx = httpx  # noqa: F841 - tests patch arc_llama.cli.httpx.get

# Import command modules last so they can import this package without cycles.
from arc_llama.cli_commands import (  # noqa: E402
    agent,
    bench_tune,
    doctor,
    init,
    install_runtime,
    models,
    mtp_info,
    serve,
    systemd,
    tui,
    upstream,
)

# Register commands.
cli.add_command(init.init_cmd)
cli.add_command(doctor.doctor_cmd)
cli.add_command(models.gpus_cmd)
cli.add_command(models.list_models_cmd)
cli.add_command(models.add_cmd)
cli.add_command(models.scan_cmd)
cli.add_command(models.remove_cmd)
cli.add_command(serve.serve_cmd)
cli.add_command(bench_tune.benchmark_cmd)
cli.add_command(bench_tune.tune_cmd)
cli.add_command(install_runtime.install_runtime_cmd)
cli.add_command(mtp_info.mtp_info_cmd)
cli.add_command(systemd.systemd_unit_cmd)
cli.add_command(upstream.upstream_group)
cli.add_command(agent.agent_cmd)
cli.add_command(agent.code_cmd)
cli.add_command(agent.agent_tui_cmd)
cli.add_command(tui.tui_cmd)


# Hide the experimental agent commands unless the user explicitly opts in.
if not experimental_agent_enabled():
    for _experimental_agent_cmd in ("agent", "code", "agent-tui"):
        cli.commands.pop(_experimental_agent_cmd, None)


__all__ = [
    "cli",
    "main",
    "arcllama_main",
    "load_config",
    "default_config_path",
    "detect_gpus",
    "save_or_die",
    "slugify_for_name",
    "download_from_hf",
    "add_local_model",
    "experimental_agent_enabled",
]
