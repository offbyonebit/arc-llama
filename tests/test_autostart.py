"""No scheduler is touched: all Windows subprocess calls are controlled doubles."""
from __future__ import annotations

import base64
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import arc_llama.autostart as auto
import arc_llama.cli as cli_mod
from arc_llama.cli import cli

SID = "S-1-5-21-111-222-333-1001"
NS = {"t": auto.NS}


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    python = tmp_path / "Program Files" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"fixture")
    windowed = python.with_name("pythonw.exe")
    windowed.write_bytes(b"fixture")
    system = tmp_path / "Windows" / "System32"
    config = tmp_path / "Config Folder & Unicode Ω" / "config.toml"
    config.parent.mkdir()
    config.write_text("[server]\nport = 11437\n", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(auto.sys, "executable", str(python))
    monkeypatch.setenv("SystemRoot", str(system.parent))
    monkeypatch.setattr(auto, "default_state_dir", lambda: tmp_path / "State Folder")
    calls = []
    state = {"xml": None, "query_error": None, "name": None, "definition": None}

    def run(args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["input"] == b"" and kwargs["capture_output"] is True
        assert kwargs["check"] is False and kwargs["timeout"] == 30
        assert not kwargs.get("shell")
        if args[0] == str(system / "whoami.exe"):
            assert args[1:] == ["/user", "/fo", "csv", "/nh"]
            return subprocess.CompletedProcess(args, 0, f'"fixture\\user","{SID}"\r\n'.encode(), b"")
        assert args[0] == str(system / "schtasks.exe")
        if args[1] == "/Query":
            if state["query_error"] or state["xml"] is None:
                return subprocess.CompletedProcess(args, 1, b"", (state["query_error"] or "Task not found").encode())
            return subprocess.CompletedProcess(args, 0, state["xml"], b"")
        if args[1] == "/Create":
            if state["xml"] is not None and "/F" not in args:
                return subprocess.CompletedProcess(args, 1, b"", b"Task already exists")
            state["definition"] = args[5]
            state["name"] = args[3]
            state["xml"] = Path(args[5]).read_bytes()
            return subprocess.CompletedProcess(args, 0, b"Created", b"")
        assert args[1] == "/Delete" and args[3] == state["name"]
        state["xml"] = None
        return subprocess.CompletedProcess(args, 0, b"Deleted", b"")

    monkeypatch.setattr(auto.subprocess, "run", run)

    def invoke(action, path=config):
        return CliRunner().invoke(cli, ["--config", str(path), "autostart", action])

    return SimpleNamespace(python=python, windowed=windowed, system=system, config=config,
                           calls=calls, state=state, invoke=invoke)


def test_enable_uses_current_user_non_elevated_logon_and_quoted_paths(scheduler):
    s = scheduler
    result = s.invoke("enable")
    assert result.exit_code == 0, result.output
    name = s.state["name"]
    assert name.startswith("ArcLlama-")
    assert s.calls[1][0] == [str(s.system / "schtasks.exe"), "/Query", "/TN", name, "/XML"]
    assert s.calls[2][0] == [str(s.system / "schtasks.exe"), "/Create", "/TN", name, "/XML", s.state["definition"]]
    assert len(s.calls) == 3 and not Path(s.state["definition"]).exists()
    root = ET.fromstring(s.state["xml"])
    assert root.findtext("t:Triggers/t:LogonTrigger/t:UserId", namespaces=NS) == SID
    assert root.findtext("t:Principals/t:Principal/t:UserId", namespaces=NS) == SID
    assert root.findtext("t:Principals/t:Principal/t:LogonType", namespaces=NS) == "InteractiveToken"
    assert root.findtext("t:Principals/t:Principal/t:RunLevel", namespaces=NS) == "LeastPrivilege"
    assert root.findtext("t:Settings/t:ExecutionTimeLimit", namespaces=NS) == "PT0S"
    assert root.findtext("t:Settings/t:MultipleInstancesPolicy", namespaces=NS) == "IgnoreNew"
    assert root.findtext("t:Actions/t:Exec/t:Command", namespaces=NS) == str(s.windowed)
    args = root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=NS)
    assert f'--config "{s.config.resolve()}"' in args and "-m arc_llama.autostart" in args
    assert f'"{s.windowed}"' in result.output and "next Windows login" in result.output


def test_status_and_disable_query_actual_owned_definition(scheduler):
    s = scheduler
    assert s.invoke("enable").exit_code == 0
    name = s.state["name"]
    s.calls.clear()
    status = s.invoke("status")
    assert status.exit_code == 0 and "exists, enabled (managed)" in status.output
    assert str(s.windowed) in status.output and "--config" in status.output
    assert len(s.calls) == 2
    s.calls.clear()
    result = s.invoke("disable")
    assert result.exit_code == 0 and s.state["xml"] is None
    assert s.calls[-1][0] == [str(s.system / "schtasks.exe"), "/Delete", "/TN", name, "/F"]
    assert len(s.calls) == 3


def test_reenable_updates_only_owned_task(scheduler):
    s = scheduler
    assert s.invoke("enable").exit_code == 0
    s.calls.clear()
    assert s.invoke("enable").exit_code == 0
    assert s.calls[-1][0][-1] == "/F"


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_foreign_task_is_never_changed(scheduler, action):
    s = scheduler
    s.state["xml"] = auto._xml(s.config, SID, "unrelated task", "unrelated.exe", "")
    original = s.state["xml"]
    result = s.invoke(action)
    assert result.exit_code == 1 and "unmanaged" in result.output
    assert len(s.calls) == 2 and s.state["xml"] == original


def test_unreadable_task_is_not_overwritten_or_deleted(scheduler):
    s = scheduler
    s.state["xml"] = auto._xml(s.config, SID, "unrelated", "unrelated.exe", "")
    original = s.state["xml"]
    s.state["query_error"] = "Access denied"
    result = s.invoke("enable")
    assert result.exit_code == 1 and "/F" not in s.calls[-1][0]
    assert s.state["xml"] == original
    s.calls.clear()
    result = s.invoke("disable")
    assert "Nothing removed" in result.output and len(s.calls) == 2
    assert s.state["xml"] == original


def test_missing_windowed_python_uses_hidden_waiting_powershell(scheduler, tmp_path):
    s = scheduler
    s.windowed.unlink()
    powershell = s.system / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    powershell.parent.mkdir(parents=True)
    powershell.write_bytes(b"fixture")
    config = tmp_path / "O'Brien & Team" / "custom config.toml"
    config.parent.mkdir()
    result = s.invoke("enable", config)
    assert result.exit_code == 0, result.output
    root = ET.fromstring(s.state["xml"])
    assert root.findtext("t:Actions/t:Exec/t:Command", namespaces=NS) == str(powershell)
    args = root.findtext("t:Actions/t:Exec/t:Arguments", namespaces=NS)
    assert args.startswith("-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand ")
    script = base64.b64decode(args.split()[-1]).decode("utf-16-le")
    assert str(s.python).replace("'", "''") in script
    assert "O''Brien & Team" in script and "-WindowStyle Hidden -Wait -PassThru" in script
    assert "-m arc_llama.autostart" in script and "-WorkingDirectory" in script


def test_status_accepts_utf16_without_bom_and_reports_disabled(scheduler):
    s = scheduler
    assert s.invoke("enable").exit_code == 0
    root = ET.fromstring(s.state["xml"])
    root.find("t:Settings/t:Enabled", NS).text = "false"
    s.state["xml"] = ET.tostring(root, encoding="unicode").encode("utf-16-le")
    result = s.invoke("status")
    assert result.exit_code == 0 and "exists, disabled" in result.output


@pytest.mark.parametrize("action", ["enable", "disable", "status"])
def test_non_windows_prints_systemd_hint_without_subprocess(monkeypatch, action):
    monkeypatch.setattr(cli_mod, "_IS_WINDOWS", False)
    monkeypatch.setattr(auto.subprocess, "run", lambda *a, **k: pytest.fail("non-Windows scheduler call"))
    result = CliRunner().invoke(cli, ["autostart", action])
    assert result.exit_code == 0 and "arc-llama systemd" in result.output


def test_background_entry_sets_streams_and_config_before_serve(tmp_path, monkeypatch):
    path = tmp_path / "config folder" / "config.toml"
    log = tmp_path / "state" / "startup.log"
    seen = []

    def main():
        seen.append(list(sys.argv))
        assert sys.stdout is sys.stderr and not sys.stdout.isatty()
        assert sys.stdin.read() == ""
        print("controlled background startup")

    monkeypatch.setattr(cli_mod, "main", main)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", None)
        patch.setattr(sys, "stderr", None)
        auto._serve(["--config", str(path), "--log-file", str(log)])
        assert sys.stdout is None and sys.stderr is None
    assert seen == [["arc-llama", "--config", str(path), "serve"]]
    assert "controlled background startup" in log.read_text(encoding="utf-8")
