"""CLI command: mtp-info."""
from __future__ import annotations

from pathlib import Path

import click

from arc_llama.cli_commands.common import console
from arc_llama.gguf_meta import mtp_info


@click.command("mtp-info")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def mtp_info_cmd(path: Path) -> None:
    """Inspect a GGUF file for MTP-relevant metadata."""
    info = mtp_info(path)
    console.print(f"[bold]GGUF:[/bold] {info['path']}")
    console.print(f"  architecture:          {info['architecture']}")
    console.print(f"  block_count:           {info['block_count']}")
    console.print(f"  nextn_predict_layers:  {info['nextn_predict_layers']}")
    console.print(f"  has_mtp_heads:         {info['has_mtp_heads']}")
    console.print(f"  is_hybrid_ssm:         {info['is_hybrid_ssm']}")


__all__ = ["mtp_info_cmd"]
