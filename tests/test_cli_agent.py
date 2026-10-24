"""Tests for the `arc-llama agent` CLI command."""

from __future__ import annotations

import importlib
import os

import arc_llama.cli

os.environ["ARC_LLAMA_EXPERIMENTAL_AGENT"] = "1"
importlib.reload(arc_llama.cli)

from click.testing import CliRunner  # noqa: E402

from arc_llama.cli import cli  # noqa: E402


def test_agent_command_requires_model() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["agent", "do something"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "--model" in result.output


def test_agent_command_fails_without_server(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[server]\nhost = "127.0.0.1"\nport = 1\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "agent", "do something", "--model", "test"],
    )
    assert result.exit_code != 0
    assert "Cannot reach arc-llama server" in result.output
