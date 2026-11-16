"""Check a distribution's real clean-install first answer on an Arc host.

Examples:
  python scripts/first_run_check.py --dist dist/arc_llama-1.3.0-py3-none-any.whl --work-dir /large-disk
  python scripts/first_run_check.py --version 1.3.0 --work-dir D:\\checks
  python scripts/first_run_check.py --version 1.3.0 --dry-run

Only the Python standard library is required by this runner. Real runs download
a runtime and model, and need the B60 check on both Linux and Windows before ship.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--dist", type=Path, help="Local wheel or source distribution")
    source.add_argument("--version", help="Exact arc-llama version to install from PyPI")
    result.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()), help="Parent on a drive with room for runtime/model")
    result.add_argument("--log-dir", type=Path, help="Parent for permanent per-run logs (default: WORK/first-run-logs)")
    result.add_argument("--timeout", type=int, default=1800, help="Setup deadline in seconds")
    result.add_argument("--dry-run", action="store_true", help="Write the isolated plan; no venv, install, network or GPU use")
    return result


def isolated_env(root: Path, inherited: dict[str, str]) -> dict[str, str]:
    # Only OS execution/network plumbing survives; credentials, app options,
    # Python paths and installer configs from the calling shell do not.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL",
               "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
    env = {k: v for k, v in inherited.items() if k.upper() in allowed}
    # A real user's llama-server on PATH would bypass the runtime download.
    runtime_names = ("llama-server", "llama-server.exe")
    clean_path = [entry for entry in inherited.get("PATH", "").split(os.pathsep)
                  if Path(entry).is_absolute() and not any((Path(entry) / name).is_file() for name in runtime_names)]
    env["PATH"] = os.pathsep.join([str(root / "venv" / ("Scripts" if os.name == "nt" else "bin")), *clean_path])
    dirs = {
        "HOME": "home", "USERPROFILE": "home", "APPDATA": "home/AppData/Roaming",
        "LOCALAPPDATA": "home/AppData/Local", "XDG_CONFIG_HOME": "home/.config",
        "XDG_CACHE_HOME": "home/.cache", "XDG_DATA_HOME": "home/.local/share",
        "XDG_STATE_HOME": "home/.local/state", "XDG_RUNTIME_DIR": "runtime",
        "XDG_DATA_DIRS": "data-dirs", "XDG_CONFIG_DIRS": "config-dirs",
        "HERMES_HOME": "hermes", "HF_HOME": "huggingface", "TEMP": "tmp", "TMP": "tmp",
    }
    for name in inherited:
        if name.upper().startswith("XDG_"):
            dirs.setdefault(name.upper(), "xdg/" + name.lower())
    for name, relative in dirs.items():
        folder = root / relative
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1",
               HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="0")
    return env


def request_json(url: str, body: dict[str, Any] | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    # Loopback must never go through an inherited corporate proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as response:
        return json.load(response)


def stop_server(process: subprocess.Popen, base: str, token: str | None) -> None:
    if process.poll() is not None:
        return
    try:
        request_json(base + "/admin/stop-all", {}, token)
    except (OSError, ValueError):
        pass
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        if os.name == "nt":
            # Built-in Windows utility; scope termination to our own process tree.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def first_answer(base: str, process: subprocess.Popen, timeout: int, log) -> str:
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Launcher exited with {process.returncode}; see application.log")
        try:
            request_json(base + "/health")
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Health deadline expired; see application.log") from None
            time.sleep(1)
    token = request_json(base + "/admin/session-token").get("admin_token")
    last_phase = None
    started = False
    while time.monotonic() < deadline:
        state = request_json(base + "/admin/onboarding", token=token)
        phase = (state.get("phase"), state.get("discovery", {}).get("status"))
        if phase != last_phase:
            log.write(f"Setup: {phase}\n")
            log.flush()
            last_phase = phase
        if state.get("error"):
            raise RuntimeError(f"Setup failed: {state['error']}")
        if not state.get("gpu"):
            raise RuntimeError("No supported GPU found; run this check on the Arc machine")
        if not started and (state.get("runtime_ready") or state.get("model")):
            raise RuntimeError("A runtime/model was already available before setup; clean-install isolation failed")
        if not started and state.get("starter"):
            request_json(base + "/admin/onboarding", {"action": "start", "starter": state["starter"]["id"]}, token)
            started = True
        elif not started and phase[1] in {"unavailable", "no_match", "hardware", "integrated"}:
            raise RuntimeError(f"No starter available: {state.get('discovery')}")
        if state.get("phase") == "ready" and state.get("model"):
            response = request_json(base + "/v1/chat/completions", {
                "model": state["model"], "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                "stream": False, "max_tokens": 128,
            }, token)
            reply = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(reply, str) or not reply.strip():
                raise RuntimeError("Real chat request returned an empty answer")
            return reply.strip()
        if process.poll() is not None:
            raise RuntimeError("Launcher exited during setup; see application.log")
        time.sleep(1)
    raise RuntimeError("Setup deadline expired; downloads/runtime/driver details are in application.log")


def run(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.version and not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", args.version):
        raise ValueError("--version must be an exact package version")
    if args.dist:
        args.dist = args.dist.resolve()
        if not args.dist.is_file() or not str(args.dist).endswith((".whl", ".tar.gz", ".zip", ".tar.bz2", ".tgz")):
            raise ValueError("--dist must point to an existing wheel or sdist")
    parent = args.work_dir.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="arc-first-run-", dir=parent))
    logs = (args.log_dir.resolve() if args.log_dir else parent / "first-run-logs") / root.name
    try:
        return _run_isolated(args, root, logs)
    except (Exception, KeyboardInterrupt) as exc:
        try:
            logs.mkdir(parents=True, mode=0o700, exist_ok=True)
            summary_path = logs / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            summary.update(passed=False, error=f"{type(exc).__name__}: {exc}", work_dir=str(root), logs=str(logs))
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass
        print(f"FAIL: {type(exc).__name__}: {exc}\nKept work: {root}\nLogs: {logs}")
        return 1


def _run_isolated(args: argparse.Namespace, root: Path, logs: Path) -> int:
    logs.mkdir(parents=True, mode=0o700)
    env = isolated_env(root, dict(os.environ))
    hooks = root / "hooks"
    hooks.mkdir()
    shutil.copyfile(Path(__file__).with_name("first_run_url_audit.py"), hooks / "sitecustomize.py")
    env["PYTHONPATH"] = str(hooks)
    env["ARC_FIRST_RUN_URL_LOG"] = str(logs / "urls.jsonl")
    executable = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    package = str(args.dist) if args.dist else "arc-llama==" + args.version
    port = 11437  # Dry-run placeholder; only a real run reserves an available port.
    if not args.dry_run:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    commands = [[sys.executable, "-m", "venv", str(root / "venv")],
                [str(executable), "-m", "pip", "install", package],
                [str(executable), "-m", "arc_llama", "start", "--no-open", "--port", str(port)]]
    plan = {"work_dir": str(root), "logs": str(logs), "commands": commands,
            "isolated_paths": {k: v for k, v in env.items() if k not in os.environ or os.environ.get(k) != v},
            "url_audit": "httpx/requests/pip vendored requests/urllib; HF Xet and hf_transfer disabled to keep transfers observable"}
    (logs / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"DRY RUN: no installation, network, GPU or chat check performed. Plan: {logs / 'plan.json'}")
        shutil.rmtree(root)
        return 0
    process = None
    token = None
    base = f"http://127.0.0.1:{port}"
    passed = False
    summary: dict[str, Any] = {"passed": False, "work_dir": str(root), "logs": str(logs)}
    try:
        with (logs / "install.log").open("w", encoding="utf-8") as install:
            for command in commands[:2]:
                subprocess.run(command, env=env, cwd=root, stdin=subprocess.DEVNULL,
                               stdout=install, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
        with (logs / "application.log").open("w", encoding="utf-8") as application, \
                (logs / "check.log").open("w", encoding="utf-8") as check:
            process = subprocess.Popen(commands[2], env=env, cwd=root, stdin=subprocess.DEVNULL,
                                       stdout=application, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                                       start_new_session=os.name != "nt")
            reply = first_answer(base, process, args.timeout, check)
            token = request_json(base + "/admin/session-token").get("admin_token")
            stop_server(process, base, token)
            rows = [json.loads(line) for line in (logs / "urls.jsonl").read_text().splitlines()]
            if not rows:
                raise RuntimeError("No external URLs captured; URL audit did not validate the first run")
            failed = [row for row in rows if not 200 <= row["status"] < 400]
            for row in rows:
                check.write(f"HTTP {row['status']} {row['url']} {row['error']}\n")
            if failed:
                raise RuntimeError(f"{len(failed)} external fetch(es) failed; see urls.jsonl and check.log")
            summary.update(passed=True, answer=reply, external_fetches=len(rows))
            passed = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            try:
                stop_server(process, base, token)
            except Exception as exc:
                passed = False
                summary.update(passed=False, cleanup_error=str(exc))
        (logs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if passed:
        shutil.rmtree(root)
        print(f"PASS: clean install, guided setup, real chat answer and URL audit. Logs: {logs}")
        return 0
    print(f"FAIL: {summary.get('error', summary.get('cleanup_error'))}\nKept work: {root}\nLogs: {logs}")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
