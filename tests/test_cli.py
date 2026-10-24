"""Tests for arc_llama.cli — arcllama entry point."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import patch

from click.testing import CliRunner

from arc_llama.cli import arcllama_main
from arc_llama.config import Config


def test_arcllama_main_loads_config_and_passes_to_tui():
    cfg = Config()
    cfg.server.host = "127.0.0.1"
    cfg.server.port = 11437

    with (
        patch("arc_llama.cli._experimental_agent_enabled", return_value=True),
        patch("arc_llama.cli.load_config", return_value=cfg) as mock_load_config,
        # Patched at its source, not on arc_llama.cli: the import is lazy, so
        # there is no module-level name on cli to patch.
        patch("arc_llama.agent_tui.run_agent_tui") as mock_run,
    ):
        runner = CliRunner()
        result = runner.invoke(arcllama_main, ["--model", "qwen"])

    assert result.exit_code == 0, result.output
    mock_load_config.assert_called_once_with()
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["config"] is cfg
    assert kwargs["model"] == "qwen"


def test_cli_imports_without_textual():
    """A plain `pip install arc-llama` has no textual; every command must still run.

    textual is an optional [tui] extra, and agent_tui raises SystemExit at
    import time when it is missing. cli.py imported agent_tui at module scope
    from ee59071 until this was fixed, which took down *every* command,
    including `--version` and `serve`. Both 0.4.0 and 0.5.0 shipped to PyPI
    that way and were unusable on a clean install.

    Runs in a subprocess so blocking textual cannot leak into other tests.
    """
    code = textwrap.dedent(
        """
        import sys

        class BlockTextual:
            def find_spec(self, name, path=None, target=None):
                if name == "textual" or name.startswith("textual."):
                    raise ImportError("simulated: textual is not installed")
                return None

        sys.meta_path.insert(0, BlockTextual())
        import arc_llama.cli  # must not raise
        print("IMPORT_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"cli import failed without textual:\n{result.stderr}"
    assert "IMPORT_OK" in result.stdout
