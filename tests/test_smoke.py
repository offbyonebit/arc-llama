"""End-to-end inference smoke test.

This test is skipped by default. To run it locally on a machine with a model
and GPU:

    ARC_LLAMA_SMOKE_MODEL=qwen2.5-0.5b-instruct-q4_k_m pytest tests/test_smoke.py

It starts a real `arc-llama serve`, loads the requested model, sends a chat
completion, and verifies a non-empty response. It will not run in CI because
GitHub Actions has no Intel Arc GPU.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_SKIP_REASON = "Set ARC_LLAMA_SMOKE_MODEL to run inference smoke test"


def _real_home() -> str:
    """Return the real user's home, ignoring any test-isolated HOME env var."""
    if sys.platform == "win32":
        return str(Path.home())
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, ImportError):
        return os.path.expanduser("~")


def _wait_for_server(
    url: str,
    proc: subprocess.Popen[str],
    timeout: float = 180.0,
) -> None:
    import httpx

    deadline = time.time() + timeout
    last_error = "no response"
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"arc-llama serve exited with code {proc.returncode}\n"
                f"stdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-8000:]}"
            )
        try:
            r = httpx.get(f"{url}/v1/models", timeout=5)
            if r.status_code == 200:
                return
            last_error = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(
        f"arc-llama serve did not become ready at {url} after {timeout:.0f}s; "
        f"last error: {last_error}"
    )


@pytest.mark.skipif(not os.environ.get("ARC_LLAMA_SMOKE_MODEL"), reason=_SKIP_REASON)
def test_inference_smoke() -> None:
    model = os.environ["ARC_LLAMA_SMOKE_MODEL"]
    config_path = os.environ.get("ARC_LLAMA_SMOKE_CONFIG")
    if not config_path:
        from arc_llama.config import default_config_path

        config_path = str(default_config_path())
    server_url = os.environ.get("ARC_LLAMA_SMOKE_URL", "http://127.0.0.1:11436")

    real_home = _real_home()
    env = os.environ.copy()
    env["HOME"] = real_home
    env["USERPROFILE"] = real_home
    env["XDG_CONFIG_HOME"] = os.path.join(real_home, ".config")
    env["APPDATA"] = env["XDG_CONFIG_HOME"]
    env["LOCALAPPDATA"] = os.path.join(real_home, "AppData", "Local")
    # Make sure the subprocess uses the same Python/venv as the test runner.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "arc_llama.cli",
            "-c",
            config_path,
            "serve",
            "--no-scan",
            "--no-auto-tune",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        _wait_for_server(server_url, proc)

        import httpx

        r = httpx.post(
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 64,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        message = data["choices"][0]["message"]
        output = message.get("content") or message.get("reasoning_content")
        assert isinstance(output, str) and output, data

        with httpx.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with hello"}],
                "max_tokens": 32,
                "stream": True,
            },
            timeout=120,
        ) as stream:
            stream.raise_for_status()
            lines = [line for line in stream.iter_lines() if line.startswith("data: ")]
        assert any(line != "data: [DONE]" for line in lines), lines
        assert lines[-1] == "data: [DONE]", lines[-5:]
    finally:
        if sys.platform == "win32":
            # Terminating the Python parent alone leaves llama-server.exe
            # behind on Windows. Kill the full process tree by PID.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
