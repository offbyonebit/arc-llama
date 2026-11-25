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
            "Install Intel's 2026.2 runtime packages (see docs/gpu-setup.md), or set "
            "paths.oneapi_setvars to a 2026.2 setvars.sh, and make sure LD_LIBRARY_PATH isn't "
            "loading 2026.1 first. To stay on 2026.1, use --runtime-version b12243."
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
    monkeypatch.setattr(caps_mod, "probe_server_caps", lambda *_a, **_k: caps_mod.ServerCaps(probe_error=SYMBOL_ERROR))
    result = CliRunner().invoke(cli, ["-c", str(tmp_path / "config.toml"), "doctor"])
    assert result.exit_code == 2, result.output
    assert "sycl_runtime" in result.output
    assert "oneAPI 2026.2 SYCL runtime" in " ".join(result.output.split())


@pytest.mark.parametrize("probe_error", [SYMBOL_ERROR, ""])
def test_linux_sycl_install_reports_the_runtime_check(tmp_path, monkeypatch, probe_error):
    from click.testing import CliRunner
    from conftest import make_config

    from arc_llama import runtime, server_caps
    from arc_llama.cli import cli

    config_path = tmp_path / "config.toml"
    make_config(tmp_path).save(config_path)
    binary = tmp_path / "llama-server"
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(runtime, "install_runtime", lambda **_kw: runtime.RuntimeInstallResult(
        binary, None, "sycl", "b12395", tmp_path, False,
    ))
    monkeypatch.setattr(server_caps, "probe_server_caps", lambda *_a, **_k: server_caps.ServerCaps(
        probe_error=probe_error,
    ))
    result = CliRunner().invoke(
        cli, ["--config", str(config_path), "install-runtime", "--backend", "sycl"]
    )
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())  # Rich wraps long lines.
    if probe_error:
        assert "needs the oneAPI 2026.2 SYCL runtime" in output
    else:
        assert "source setvars.sh" in output


@pytest.mark.parametrize("configured", [True, False])
def test_probe_retries_with_configured_setvars_first(tmp_path, monkeypatch, configured):
    import arc_llama.platform_checks as checks
    import arc_llama.server_caps as caps_mod

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fixture")
    discovered = tmp_path / "discovered" / "setvars.sh"
    chosen = tmp_path / "oneapi 2026.2" / ("setvars.bat" if checks.sys.platform == "win32" else "setvars.sh")
    chosen.parent.mkdir()
    chosen.touch()
    sources, calls = [], []
    monkeypatch.setattr(checks, "oneapi_runtime_env_needed", lambda: False)
    monkeypatch.setattr(checks, "oneapi_setvars_path", lambda: discovered)
    monkeypatch.setattr(checks, "source_setvars", lambda path, *, force=False: sources.append(path) or {"X": "1"})

    def run(cmd, **kwargs):
        calls.append(kwargs.get("env"))
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=SYMBOL_ERROR)
        return subprocess.CompletedProcess(cmd, 0, stdout="--flash-attn on off auto", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    caps = caps_mod.probe_server_caps(str(binary), oneapi_setvars=str(chosen) if configured else None)
    assert caps.probed
    assert sources == [chosen if configured else discovered]


def test_configured_setvars_helps_a_missing_library(tmp_path, monkeypatch):
    import arc_llama.platform_checks as checks
    from arc_llama.server_caps import _env_with_oneapi

    chosen = tmp_path / "setvars.sh"
    chosen.touch()
    monkeypatch.setattr(checks, "oneapi_runtime_env_needed", lambda: False)
    monkeypatch.setattr(checks, "source_setvars", lambda path, *, force=False: {"FROM": str(path)})
    assert _env_with_oneapi({}, oneapi_setvars=str(chosen)) == {"FROM": str(chosen)}
    assert _env_with_oneapi({}, oneapi_setvars=str(tmp_path / "missing.sh")) is None


def test_install_and_doctor_pass_the_configured_setvars(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from conftest import make_config

    import arc_llama.cli_diagnostics as diagnostics
    from arc_llama import runtime, server_caps
    from arc_llama.arch import Backend
    from arc_llama.cli import cli

    chosen = tmp_path / "setvars.sh"
    chosen.touch()
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fixture")
    cfg = make_config(tmp_path)
    cfg.paths.llama_server = str(binary)
    cfg.paths.oneapi_setvars = str(chosen)
    config_path = tmp_path / "config.toml"
    cfg.save(config_path)
    seen = []
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(runtime, "install_runtime", lambda **_kw: runtime.RuntimeInstallResult(
        binary, None, "sycl", "b12395", tmp_path, False,
    ))
    monkeypatch.setattr(server_caps, "probe_server_caps", lambda _b, **kw: seen.append(kw) or server_caps.ServerCaps())
    monkeypatch.setattr(diagnostics, "detect_backends", lambda _: {Backend.SYCL})
    monkeypatch.setattr(diagnostics, "detect_llama_server_backend", lambda _: Backend.SYCL)
    runner = CliRunner()
    runner.invoke(cli, ["--config", str(config_path), "install-runtime", "--backend", "sycl"])
    doctor = runner.invoke(cli, ["--config", str(config_path), "doctor"])
    assert seen == [{"oneapi_setvars": str(chosen)}] * 2
    assert str(chosen) in "".join(doctor.output.split())


async def test_compatibility_check_names_the_sycl_runtime_problem(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    import arc_llama.model_compatibility as compat
    from arc_llama.config import Config, ModelConfig

    compat._PROBE_CACHE.clear()
    monkeypatch.setattr(compat, "probe_output", AsyncMock(return_value=(127, SYMBOL_ERROR)))
    assert await compat.probe_architecture("runtime", "identity", "llama") == "sycl_runtime"
    assert not compat._PROBE_CACHE

    file = tmp_path / "m.gguf"
    file.write_bytes(b"GGUF")
    cfg = Config()
    cfg.models = [ModelConfig("local", str(file), 18080, "gpu")]
    monkeypatch.setattr(compat, "runtime_identity", lambda _: ("/runtime", "fingerprint"))
    monkeypatch.setattr(compat, "header_architecture", lambda _: "llama")
    result = await compat.assess_compatibility(cfg, {"name": "local"})
    assert result["label"] == "SYCL runtime ABI mismatch"
    assert result["status"] == "incompatible"
    assert result["action"] == sycl_runtime_error_hint(SYMBOL_ERROR)


async def test_compatibility_probe_uses_the_launch_env_without_a_model_env(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    import arc_llama.launcher as launcher
    import arc_llama.model_compatibility as compat
    from arc_llama.config import Config, GPUConfig, ModelConfig

    file = tmp_path / "m.gguf"
    file.write_bytes(b"GGUF")
    cfg = Config()
    cfg.gpus = [GPUConfig(pci_slot="0000:03:00.0", sycl_index=0, arch="battlemage")]
    cfg.models = [ModelConfig("local", str(file), 18080, "0000:03:00.0")]
    monkeypatch.setattr(compat, "runtime_identity", lambda _: ("/runtime", "fingerprint"))
    monkeypatch.setattr(compat, "header_architecture", lambda _: "llama")
    monkeypatch.setattr(launcher, "build_model_env", lambda *_a: {"FROM": "launch"})
    probe = AsyncMock(return_value="sycl_runtime")
    monkeypatch.setattr(compat, "probe_architecture", probe)
    result = await compat.assess_compatibility(cfg, {"name": "local"})
    assert probe.call_args.kwargs["env"] == {"FROM": "launch"}
    assert result["label"] == "SYCL runtime ABI mismatch"
