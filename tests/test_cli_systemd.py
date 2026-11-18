from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from arc_llama.cli import cli


@pytest.mark.skipif(sys.platform == "win32", reason="systemd is unavailable on Windows")
@pytest.mark.parametrize(
    ("executable", "quoted"),
    [
        ("/home/me/My Apps/venv/bin/arc-llama", '"/home/me/My Apps/venv/bin/arc-llama"'),
        (r'/home/me/My "Apps"\venv/bin/arc-llama', r'"/home/me/My \"Apps\"\\venv/bin/arc-llama"'),
    ],
)
def test_systemd_print_quotes_executable(monkeypatch, tmp_path, executable, quoted):
    monkeypatch.setattr("arc_llama.cli.shutil.which", lambda name: executable)

    result = CliRunner().invoke(cli, ["-c", str(tmp_path / "config.toml"), "systemd"])

    assert result.exit_code == 0, result.output
    assert f"ExecStart={quoted} serve" in result.output.splitlines()


@pytest.mark.skipif(sys.platform == "win32", reason="systemd is unavailable on Windows")
def test_systemd_write_matches_printed_unit_as_utf8(monkeypatch, tmp_path):
    executable = "/home/me/My Apps/venv/bin/arc-llama"
    monkeypatch.setattr("arc_llama.cli.shutil.which", lambda name: executable)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    args = ["-c", str(tmp_path / "config.toml"), "systemd", "--description", "Café 東京"]
    runner = CliRunner()

    printed = runner.invoke(cli, args)
    written = runner.invoke(cli, [*args, "--write"])

    assert printed.exit_code == 0, printed.output
    assert written.exit_code == 0, written.output
    target = tmp_path / ".config" / "systemd" / "user" / "arc-llama.service"
    unit = target.read_bytes().decode("utf-8")
    assert printed.output == unit + "\n"
    assert f'ExecStart="{executable}" serve' in unit.splitlines()
    assert "Description=Café 東京" in unit.splitlines()


def test_systemd_windows_early_exit(monkeypatch, tmp_path):
    monkeypatch.setattr("arc_llama.cli._IS_WINDOWS", True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = CliRunner().invoke(
        cli, ["-c", str(tmp_path / "config.toml"), "systemd", "--write"]
    )

    assert result.exit_code == 1
    assert "systemd is not available on Windows." in result.output
    assert not (tmp_path / ".config" / "systemd").exists()
