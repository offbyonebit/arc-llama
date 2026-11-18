"""Recover from a visible but ABI-incompatible oneAPI runtime."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arc_llama.platform_checks import sycl_runtime_error_hint

SYMBOL_ERROR = (
    "llama-server: symbol lookup error: /runtime/libggml-sycl.so: undefined symbol: "
    "_ZN4sycl3_V16detail19defaultAsyncHandlerE"
)


def test_runtime_pin():
    from arc_llama.runtime_pin import LLAMA_CPP_BUILD

    assert LLAMA_CPP_BUILD == "b12395"


@pytest.mark.parametrize("platform_name", ["linux", "win32"])
@pytest.mark.parametrize("force", [False, True])
def test_setvars_force_argument(tmp_path, monkeypatch, platform_name, force):
    import arc_llama.platform_checks as checks

    setvars = tmp_path / ("setvars.bat" if platform_name == "win32" else "setvars.sh")
    setvars.touch()

    def run(cmd, **kwargs):
        script = Path(cmd[-1]).read_text() if platform_name == "win32" else cmd[-1]
        assert (" --force" in script) is force
        output = "FORCED=yes\n" if platform_name == "win32" else b"FORCED=yes\0"
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(checks.sys, "platform", platform_name)
    monkeypatch.setattr(subprocess, "run", run)
    assert checks.source_setvars(setvars, force=force) == {"FORCED": "yes"}


@pytest.mark.parametrize("output,matched", [
    (SYMBOL_ERROR, True),
    ("symbol lookup error: libsycl.so: undefined symbol: other", True),
    ("symbol lookup error: libother.so: undefined symbol: other", False),
    ("libsycl.so: permission denied", False),
    ("unrelated error", False),
])
def test_runtime_hint(output, matched):
    hint = sycl_runtime_error_hint(output)
    assert bool(hint) is matched
    if matched:
        assert hint == (
            "This llama.cpp build needs the oneAPI 2026.2 SYCL runtime (builds after b12243). "
            "Install oneAPI 2026.2 beside 2026.1, or set paths.oneapi_setvars to its setvars.sh, "
            "or run with --runtime-version b12359 on a 2026.2 runtime / b12243 on 2026.1."
        )


@pytest.mark.parametrize("probe", ["build", "caps"])
@pytest.mark.parametrize("recovers", [True, False])
def test_symbol_error_forces_oneapi_and_retries_once(tmp_path, monkeypatch, caplog, probe, recovers):
    import arc_llama.platform_checks as checks
    import arc_llama.server_caps as caps_mod
    from arc_llama.benchmark import llama_cpp_build

    binary = tmp_path / "runtime with spaces.exe"
    binary.write_bytes(b"fixture")
    calls, sources = [], []
    monkeypatch.setattr(checks, "oneapi_runtime_env_needed", lambda: False)
    monkeypatch.setattr(checks, "oneapi_setvars_path", lambda: tmp_path / "setvars.sh")

    def source(path, *, force=False):
        sources.append((path, force))
        return {"LD_LIBRARY_PATH": "oneapi-2026.2"}

    def run(cmd, **kwargs):
        calls.append(kwargs.get("env"))
        if len(calls) == 1 or not recovers:
            return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=SYMBOL_ERROR)
        output = "version: 1.0.0-dev (build 12395, commit abc1234)" if probe == "build" else "--flash-attn on off auto"
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(checks, "source_setvars", source)
    monkeypatch.setattr(subprocess, "run", run)
    if probe == "build":
        assert llama_cpp_build(str(binary)) == ("b12395 (abc1234)" if recovers else None)
    else:
        caps = caps_mod.probe_server_caps(str(binary), env={"PRESERVED": "yes"})
        assert caps.probed is recovers
        assert caps.probe_error == ("" if recovers else SYMBOL_ERROR)
        assert calls[1]["PRESERVED"] == "yes"
    assert len(calls) == 2
    assert sources == [(tmp_path / "setvars.sh", True)]
    assert calls[1]["LD_LIBRARY_PATH"] == "oneapi-2026.2"
    if not recovers:
        assert sycl_runtime_error_hint(SYMBOL_ERROR) in caplog.text


def test_doctor_reports_sycl_runtime_hint(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import arc_llama.cli_diagnostics as diagnostics
    import arc_llama.server_caps as caps_mod
    from arc_llama.arch import Backend
    from arc_llama.cli import cli
    from arc_llama.config import Config

    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"fixture")
    cfg = Config()
    cfg.paths.llama_server = str(binary)
    cfg.save(tmp_path / "config.toml")
    monkeypatch.setattr(diagnostics, "detect_backends", lambda _: {Backend.SYCL})
    monkeypatch.setattr(diagnostics, "detect_llama_server_backend", lambda _: Backend.SYCL)
    monkeypatch.setattr(caps_mod, "probe_server_caps", lambda _: caps_mod.ServerCaps(probe_error=SYMBOL_ERROR))
    result = CliRunner().invoke(cli, ["-c", str(tmp_path / "config.toml"), "doctor"])
    assert result.exit_code == 2, result.output
    assert "sycl_runtime" in result.output
    assert "oneAPI 2026.2 SYCL runtime" in " ".join(result.output.split())
