"""Login startup runs arc-llama under pythonw.exe, which has no console.

Children started from there get a visible console of their own unless
CREATE_NO_WINDOW is set (closing it kills the model), and CTRL_BREAK can't
reach them, so a stop must go straight to taskkill instead of sitting out
the whole drain. These run on any OS by faking the Windows pieces.
"""
from __future__ import annotations

import subprocess

import pytest

from arc_llama import launcher as launcher_mod
from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.launcher import LlamaServer, build_plan

NEW_GROUP = 0x00000200
NO_WINDOW = 0x08000000


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setattr(launcher_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", NEW_GROUP, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", NO_WINDOW, raising=False)
    return monkeypatch


def _server():
    plan = build_plan(
        Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
        ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
        GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage"),
    )
    return LlamaServer(plan)


def _start_flags(monkeypatch, tmp_path, has_console):
    monkeypatch.setattr(launcher_mod, "_has_console", lambda: has_console)
    srv = _server()
    seen = {}

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    try:
        srv.start(log_dir=tmp_path / "logs")
    finally:
        srv.process = None
        srv.stop()
    return seen["creationflags"]


def test_no_console_parent_hides_child_console(win, tmp_path):
    assert _start_flags(win, tmp_path, has_console=False) == NEW_GROUP | NO_WINDOW


def test_console_parent_keeps_shared_console_for_ctrl_break(win, tmp_path):
    assert _start_flags(win, tmp_path, has_console=True) == NEW_GROUP


def _stop(monkeypatch, *, break_fails, drain=30.0):
    calls = []

    class FakeProc:
        pid = 4242
        dead = False

        def poll(self):
            return 0 if self.dead else None

        def send_signal(self, sig):
            calls.append(("signal", sig))
            if break_fails:
                raise OSError(6, "The handle is invalid")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if not self.dead:
                raise subprocess.TimeoutExpired("llama-server", timeout)

    proc = FakeProc()

    def fake_run(cmd, **kwargs):
        calls.append(("run", cmd, kwargs.get("creationflags")))
        proc.dead = True
        return subprocess.CompletedProcess(cmd, 0)

    srv = _server()
    monkeypatch.setattr(subprocess, "run", fake_run)
    srv.process = proc
    srv.stop(drain_seconds=drain)
    return calls


def test_failed_ctrl_break_skips_the_drain_and_hides_taskkill(win):
    calls = _stop(win, break_fails=True, drain=30.0)
    waits = [c for c in calls if c[0] == "wait"]
    assert waits[0] == ("wait", 0), calls
    runs = [c for c in calls if c[0] == "run"]
    assert runs and runs[0][1][:4] == ["taskkill", "/F", "/T", "/PID"]
    assert runs[0][2] == NO_WINDOW


def test_delivered_ctrl_break_still_gets_the_full_drain(win):
    calls = _stop(win, break_fails=False, drain=30.0)
    assert ("wait", 30.0) in calls, calls


def test_has_console_defaults_true_off_windows(monkeypatch):
    # No ctypes.windll on Linux/macOS: never add CREATE_NO_WINDOW there.
    import ctypes

    monkeypatch.delattr(ctypes, "windll", raising=False)
    assert launcher_mod._has_console() is True
