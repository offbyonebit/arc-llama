"""Per-user Windows login startup and the console-free server entry point."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import locale
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import click

from arc_llama.config import default_state_dir

NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
OWNER = "Arc Llama login startup v1: "


def _system32(name: str) -> Path:
    return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name


def _text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if b"\0" in data[:64]:
        return data.decode("utf-16-le", errors="replace")
    return data.decode(locale.getpreferredencoding(False), errors="replace")


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, input=b"", capture_output=True, check=False, timeout=30,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _check(result: subprocess.CompletedProcess[bytes]) -> None:
    if result.returncode:
        raise ValueError(_text(result.stderr or result.stdout).strip() or "Windows command failed")


def _identity(config: Path) -> tuple[str, str, str]:
    result = _run([str(_system32("whoami.exe")), "/user", "/fo", "csv", "/nh"])
    _check(result)
    rows = list(csv.reader(_text(result.stdout).splitlines()))
    sid = rows[0][1].strip() if rows and len(rows[0]) > 1 else ""
    if not re.fullmatch(r"S-\d+(?:-\d+)+", sid):
        raise ValueError("Could not determine the current Windows user SID")
    digest = hashlib.sha256((sid + "\n" + os.path.normcase(str(config))).encode()).hexdigest()
    return sid, "ArcLlama-" + digest[:16], OWNER + digest


def _command(config: Path, log: Path) -> tuple[str, str]:
    python = Path(sys.executable).absolute()
    windowed = python.with_name("pythonw.exe")
    args = subprocess.list2cmdline(["-m", "arc_llama.autostart", "--config", str(config), "--log-file", str(log)])
    if windowed.is_file():
        return str(windowed), args
    powershell = _system32("WindowsPowerShell") / "v1.0" / "powershell.exe"
    if not powershell.is_file():
        raise ValueError("pythonw.exe is missing and Windows PowerShell is unavailable for hidden startup")
    # Preserve native argument quoting inside literal PowerShell strings.
    exe = str(python).replace("'", "''")
    native_args = args.replace("'", "''")
    directory = str(config.parent if config.parent.is_dir() else Path.home()).replace("'", "''")
    script = f"$p = Start-Process -FilePath '{exe}' -ArgumentList '{native_args}' -WorkingDirectory '{directory}' -WindowStyle Hidden -Wait -PassThru -ErrorAction Stop; exit $p.ExitCode"
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return str(powershell), f"-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand {encoded}"


def _xml(config: Path, sid: str, marker: str, command: str, arguments: str) -> bytes:
    root = ET.Element("Task", {"xmlns": NS, "version": "1.2"})
    ET.SubElement(ET.SubElement(root, "RegistrationInfo"), "Description").text = marker
    trigger = ET.SubElement(ET.SubElement(root, "Triggers"), "LogonTrigger")
    ET.SubElement(trigger, "Enabled").text = "true"
    ET.SubElement(trigger, "UserId").text = sid
    principal = ET.SubElement(ET.SubElement(root, "Principals"), "Principal", {"id": "User"})
    for key, value in (("UserId", sid), ("LogonType", "InteractiveToken"), ("RunLevel", "LeastPrivilege")):
        ET.SubElement(principal, key).text = value
    settings = ET.SubElement(root, "Settings")
    for key, value in (("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                       ("StopIfGoingOnBatteries", "false"), ("ExecutionTimeLimit", "PT0S"), ("Enabled", "true")):
        ET.SubElement(settings, key).text = value
    action = ET.SubElement(ET.SubElement(root, "Actions", {"Context": "User"}), "Exec")
    for key, value in (("Command", command), ("Arguments", arguments),
                       ("WorkingDirectory", str(config.parent if config.parent.is_dir() else Path.home()))):
        ET.SubElement(action, key).text = value
    return ET.tostring(root, encoding="utf-16", xml_declaration=True)


def register_autostart_commands(cli: click.Group, *, is_windows: Callable[[], bool]) -> None:
    @cli.command("autostart")
    @click.argument("action", type=click.Choice(["enable", "disable", "status"]))
    @click.pass_context
    def autostart(ctx: click.Context, action: str) -> None:
        """Start at Windows login for the current user; no elevated task or password."""
        if not is_windows():
            click.echo("Windows login startup is unavailable here. Use arc-llama systemd.")
            return
        try:
            config = Path(ctx.obj["config_path"]).expanduser().resolve()
            sid, name, marker = _identity(config)
            tool = str(_system32("schtasks.exe"))
            query = _run([tool, "/Query", "/TN", name, "/XML"])
            task = ET.fromstring(_text(query.stdout)) if query.returncode == 0 else None
            owned = task is not None and task.findtext("t:RegistrationInfo/t:Description", namespaces={"t": NS}) == marker
            if action == "status":
                if task is None:
                    click.echo("Startup task not found or unavailable: " + _text(query.stderr or query.stdout).strip())
                    return
                enabled = task.findtext("t:Settings/t:Enabled", "true", {"t": NS}) != "false"
                enabled = enabled and task.findtext("t:Triggers/t:LogonTrigger/t:Enabled", "true", {"t": NS}) != "false"
                command = task.findtext("t:Actions/t:Exec/t:Command", "", {"t": NS})
                args = task.findtext("t:Actions/t:Exec/t:Arguments", "", {"t": NS})
                click.echo(f"{name}: exists, {'enabled' if enabled else 'disabled'} ({'managed' if owned else 'unmanaged'}).")
                click.echo("Command: " + subprocess.list2cmdline([command]) + " " + args)
                return
            if task is not None and not owned:
                raise ValueError("Refusing to change an unmanaged task with this name")
            if action == "disable":
                if task is None:
                    click.echo("Nothing removed; no readable startup task was found. " + _text(query.stderr or query.stdout).strip())
                    return
                _check(_run([tool, "/Delete", "/TN", name, "/F"]))
                click.echo("Removed the login startup task; an already running server continues until stopped.")
                return
            log = default_state_dir() / "autostart" / (name + ".log")
            command, args = _command(config, log)
            with tempfile.TemporaryDirectory(prefix="arc-autostart-") as folder:
                definition = Path(folder) / "task.xml"
                definition.write_bytes(_xml(config, sid, marker, command, args))
                # A failed query never authorizes overwriting an unreadable task.
                _check(_run([tool, "/Create", "/TN", name, "/XML", str(definition), *(["/F"] if owned else [])]))
            click.echo(f"Enabled {name} for your next Windows login.")
            click.echo("Command: " + subprocess.list2cmdline([command]) + " " + args)
            click.echo(f"Log: {log}")
        except (OSError, subprocess.SubprocessError, ValueError, ET.ParseError) as exc:
            raise click.ClickException(str(exc)) from exc


def _serve(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Background server entry point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-file", required=True, type=Path)
    args = parser.parse_args(argv)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8", buffering=1) as log, open(os.devnull, encoding="utf-8") as stdin:
        previous = sys.stdin, sys.stdout, sys.stderr, sys.argv
        sys.stdin, sys.stdout, sys.stderr = stdin, log, log
        sys.argv = ["arc-llama", "--config", args.config, "serve"]
        try:
            # pythonw has no standard streams; configure them before Rich and uvicorn import.
            from arc_llama.cli import main

            main()
        finally:
            sys.stdin, sys.stdout, sys.stderr, sys.argv = previous


if __name__ == "__main__":
    _serve()
