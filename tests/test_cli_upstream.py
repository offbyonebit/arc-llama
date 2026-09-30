from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.config import Config, load_config


def test_upstream_commands_add_list_and_remove(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    Config().save(config_path)
    runner = CliRunner()
    common = ["--config", str(config_path), "upstream"]

    added = runner.invoke(cli, [*common, "add", "local", "http://localhost:11434/"])
    assert added.exit_code == 0, added.output
    assert load_config(config_path).upstreams[0].url == "http://localhost:11434"

    listed = runner.invoke(cli, [*common, "list"])
    assert listed.exit_code == 0, listed.output
    assert "local" in listed.output
    assert "http://localhost:11434" in listed.output

    removed = runner.invoke(cli, [*common, "remove", "local"])
    assert removed.exit_code == 0, removed.output
    assert load_config(config_path).upstreams == []
