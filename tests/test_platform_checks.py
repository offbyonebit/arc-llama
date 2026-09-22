"""Tests for host/driver competitive-inference checks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arc_llama.platform_checks import (
    format_bytes,
    max_memory_bar_bytes,
    oneapi_runtime_env_needed,
    oneapi_setvars_path,
    parse_kernel_version,
    rebar_likely_enabled,
    source_setvars,
)

_skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="fakes a Linux /sys/bus/pci/devices/<slot> resource file; the colon "
    "in the slot dir name is illegal on Windows and there's no sysfs to fake there",
)


def _write_resource(path: Path, lines: list[str]) -> Path:
    sysfs = path / "0000:03:00.0"
    sysfs.mkdir(parents=True)
    (sysfs / "resource").write_text("\n".join(lines) + "\n")
    return sysfs


class TestMaxMemoryBar:
    @_skip_on_windows
    def test_reads_largest_mem_bar(self, tmp_path: Path):
        # 256 MiB BAR (typical without ReBAR) + tiny IO region
        sysfs = _write_resource(
            tmp_path,
            [
                "0x00000000f0000000 0x00000000f0ffffff 0x0000000000040200",  # 16 MiB
                "0x00000000e0000000 0x00000000efffffff 0x0000000000040200",  # 256 MiB
                "0x000000000000e000 0x000000000000e0ff 0x0000000000000101",  # IO
            ],
        )
        size = max_memory_bar_bytes(sysfs)
        assert size == 256 * 1024 * 1024

    @_skip_on_windows
    def test_large_rebar_aperture(self, tmp_path: Path):
        # 16 GiB aperture
        start = 0x0000380000000000
        end = start + 16 * 1024**3 - 1
        sysfs = _write_resource(
            tmp_path,
            [f"0x{start:016x} 0x{end:016x} 0x000000000014020e"],
        )
        size = max_memory_bar_bytes(sysfs)
        assert size == 16 * 1024**3


class TestRebarHeuristic:
    @_skip_on_windows
    def test_small_bar_is_off(self, tmp_path: Path):
        sysfs = _write_resource(
            tmp_path,
            ["0x00000000e0000000 0x00000000efffffff 0x0000000000040200"],  # 256 MiB
        )
        assert rebar_likely_enabled(sysfs, vram_mb=24 * 1024) is False

    @_skip_on_windows
    def test_large_bar_is_on(self, tmp_path: Path):
        start = 0x0000380000000000
        end = start + 24 * 1024**3 - 1
        sysfs = _write_resource(
            tmp_path,
            [f"0x{start:016x} 0x{end:016x} 0x000000000014020e"],
        )
        assert rebar_likely_enabled(sysfs, vram_mb=24 * 1024) is True

    def test_missing_resource_unknown(self, tmp_path: Path):
        empty = tmp_path / "nodev"
        empty.mkdir()
        assert rebar_likely_enabled(empty) is None


class TestHelpers:
    def test_format_bytes(self):
        assert format_bytes(256 * 1024 * 1024) == "256 MiB"
        assert "GiB" in format_bytes(16 * 1024**3)

    def test_parse_kernel_version(self):
        assert parse_kernel_version("6.14.0-generic") == (6, 14)
        assert parse_kernel_version("5.15.0") == (5, 15)
        assert parse_kernel_version("bogus") is None


def _setvars_name() -> str:
    return "setvars.bat" if sys.platform == "win32" else "setvars.sh"


class TestOneapiSetvarsPath:
    def test_finds_env_oneapi_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path / "oneapi"))
        monkeypatch.setenv("CMPLR_ROOT", "")
        setvars = tmp_path / "oneapi" / _setvars_name()
        setvars.parent.mkdir(parents=True)
        setvars.write_text("# fake")
        assert oneapi_setvars_path() == setvars

    def test_finds_cmplr_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        oneapi_root = tmp_path / "oneapi"
        setvars = oneapi_root / _setvars_name()
        setvars.parent.mkdir(parents=True)
        setvars.write_text("# fake")
        cmplr = oneapi_root / "compiler" / "2026.1"
        cmplr.mkdir(parents=True)
        monkeypatch.setenv("CMPLR_ROOT", str(cmplr))
        monkeypatch.setenv("ONEAPI_ROOT", "")
        assert oneapi_setvars_path() == setvars

    def test_prefers_env_over_standard_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        env_root = tmp_path / "env-oneapi"
        env_setvars = env_root / _setvars_name()
        env_setvars.parent.mkdir(parents=True)
        env_setvars.write_text("# fake")
        monkeypatch.setenv("ONEAPI_ROOT", str(env_root))
        monkeypatch.setenv("CMPLR_ROOT", "")
        result = oneapi_setvars_path()
        assert result is not None
        assert result == env_setvars


def _runtime_lib_name() -> str:
    return "libsvml.dll" if sys.platform == "win32" else "libsvml.so"


class TestOneapiRuntimeEnvNeeded:
    def test_no_oneapi_dirs_returns_true(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path / "no-such-oneapi"))
        monkeypatch.setenv("CMPLR_ROOT", "")
        # Ensure PATH doesn't accidentally find libs on this test runner.
        monkeypatch.setenv("PATH", "")
        # Ensure ldconfig doesn't accidentally find libs on this test runner.
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: None)
        assert oneapi_runtime_env_needed() is True

    def test_finds_lib_in_oneapi_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        lib_dir = tmp_path / "lib" / "intel64"
        lib_dir.mkdir(parents=True)
        (lib_dir / _runtime_lib_name()).write_text("")
        monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))
        monkeypatch.setenv("CMPLR_ROOT", "")
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: None)
        assert oneapi_runtime_env_needed() is False


class TestSourceSetvars:
    def test_parses_sourced_env(self, tmp_path: Path):
        if sys.platform == "win32":
            setvars = tmp_path / "setvars.bat"
            setvars.write_text(
                r"@echo off" + "\n"
                r"set ONEAPI_ROOT=C:\Intel\oneAPI" + "\n"
                r"set LD_LIBRARY_PATH=C:\Intel\oneAPI\lib" + "\n"
                r"set PATH=C:\Intel\oneAPI\bin;%PATH%" + "\n"
            )
            env = source_setvars(setvars)
            assert env["ONEAPI_ROOT"] == r"C:\Intel\oneAPI"
            assert env["LD_LIBRARY_PATH"] == r"C:\Intel\oneAPI\lib"
            path = next(
                (value for key, value in env.items() if key.casefold() == "path"),
                None,
            )
            assert path is not None
            assert r"C:\Intel\oneAPI\bin" in path
        else:
            setvars = tmp_path / "setvars.sh"
            setvars.write_text(
                "export ONEAPI_ROOT=/opt/intel/oneapi\n"
                "export LD_LIBRARY_PATH=/opt/intel/oneapi/lib/intel64\n"
                "export PATH=/opt/intel/oneapi/bin:$PATH\n"
            )
            env = source_setvars(setvars)
            assert env["ONEAPI_ROOT"] == "/opt/intel/oneapi"
            assert env["LD_LIBRARY_PATH"] == "/opt/intel/oneapi/lib/intel64"
            # PATH should reflect expansion of the (empty) inherited PATH.
            assert "/opt/intel/oneapi/bin" in env["PATH"]
            # Bash bookkeeping variables should be stripped.
            assert "PWD" not in env
            assert "SHLVL" not in env

    def test_missing_script_returns_empty(self, tmp_path: Path):
        if sys.platform == "win32":
            assert source_setvars(tmp_path / "no-such-setvars.bat") == {}
        else:
            assert source_setvars(tmp_path / "no-such-setvars.sh") == {}
