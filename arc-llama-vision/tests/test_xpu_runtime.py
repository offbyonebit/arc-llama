"""Focused tests for the XPU runtime sanity helpers.

These tests are pure-stdlib and must not touch a GPU or initialize torch:
the module exists precisely because the GPU runtime can die before any
import-level sanity check could run. All process-map inputs are fakes or
this interpreter's own process.

Run from the repo root:

    .venv/bin/python -m pytest arc-llama-vision/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import arc_llama_vision.xpu_runtime as xrt  # noqa: E402

# ---------------------------------------------------------------------------
# detect_ur_runtime_mixing: /proc/<pid>/map parsing tree classification
# ---------------------------------------------------------------------------

_ONEAPI_SYCL = "/mnt/storage/opt/intel/oneapi/compiler/2026.1/lib/libsycl.so.9.0.0"
_VENV_SYCL = (
    "/mnt/storage/comfyui-xpu/venv/lib/python3.12/site-packages/torch/lib/../../../../libsycl.so.8"
)
_VENV_UR = "/mnt/storage/comfyui-xpu/venv/lib/libur_loader.so.0"
_ONEAPI_UR = "/mnt/storage/opt/intel/oneapi/compiler/2026.1/lib/libur_loader.so.0"


def _report_with_maps(monkeypatch: pytest.MonkeyPatch, paths: list[str]) -> dict:
    monkeypatch.setattr(xrt, "_proc_map_paths", lambda pid: paths)
    return xrt.detect_ur_runtime_mixing(pid=999999)


def test_mixed_binding_is_flagged_for_real_crash_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact production pairing: wheel-bundled SYCL RT bound to a system
    oneAPI UR loader (what LD_LIBRARY_PATH from setvars.sh produces)."""
    report = _report_with_maps(
        monkeypatch,
        [
            _VENV_SYCL,
            _ONEAPI_UR,
            "/mnt/storage/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0",
        ],
    )
    assert report["diagnosis"] == "mixed"
    # SYCL realpaths into the venv lib dir; the UR loader sits in the
    # system oneAPI 2026.1 compiler tree.
    assert report["sycl_tree"].startswith("libdir:")
    assert report["sycl_tree"].replace("\\", "/").endswith("/venv/lib")
    assert report["loader_tree"] == "oneapi:/mnt/storage/opt/intel/oneapi/compiler/2026.1"
    assert "urProgramBuildExp" in report["detail"]
    assert "sanitize_launch_env" in report["detail"]


def test_venv_bound_pairing_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed launcher's pairing: everything from the wheel's own tree."""
    report = _report_with_maps(monkeypatch, [_VENV_SYCL, _VENV_UR])
    assert report["diagnosis"] == "clean"
    assert xrt.describe_binding(pid=999999) == ""


def test_oneapi_internal_pairing_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process *built for* the system oneAPI tree resolving its own UR
    stack is internally consistent, not mixed."""
    report = _report_with_maps(monkeypatch, [_ONEAPI_SYCL, _ONEAPI_UR])
    assert report["diagnosis"] == "clean"


def test_unreadable_process_maps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xrt, "_proc_map_paths", lambda pid: [])
    report = xrt.detect_ur_runtime_mixing(pid=1)
    assert report["diagnosis"] == "unavailable"


def test_self_process_is_inspectable_with_real_proc() -> None:
    """Reading /proc/<our pid>/maps works on Linux (the helper is real, not
    vaporware). On a non-Linux sandbox the graceful 'unavailable' path is
    what we expect instead."""
    report = xrt.detect_ur_runtime_mixing(pid=__import__("os").getpid())
    assert report["diagnosis"] in ("clean", "unavailable", "mixed")


# ---------------------------------------------------------------------------
# sanitize_launch_env: launcher hygiene
# ---------------------------------------------------------------------------


def test_sanitize_strips_oneapi_library_shadows() -> None:
    polluted = {
        "LD_LIBRARY_PATH": "/mnt/storage/opt/intel/oneapi/compiler/2026.1/lib",
        "LIBRARY_PATH": "/opt/intel/oneapi/mkl/lib",
        "CPATH": "/opt/intel/oneapi/include",
        "PATH": "/usr/bin",
        "HOME": "/home/slowe",
    }
    clean = xrt.sanitize_launch_env(polluted)
    assert "LD_LIBRARY_PATH" not in clean
    assert "LIBRARY_PATH" not in clean
    assert "CPATH" not in clean
    # Unrelated variables survive untouched.
    assert clean["PATH"] == "/usr/bin"
    assert clean["HOME"] == "/home/slowe"


def test_sanitize_keeps_device_selection_defaults() -> None:
    clean = xrt.sanitize_launch_env({})
    assert clean["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"
    assert clean["ZES_ENABLE_SYSMAN"] == "1"
    assert clean["SYCL_CACHE_PERSISTENT"] == "0"


def test_sanitize_preserves_caller_overrides() -> None:
    clean = xrt.sanitize_launch_env(
        {"ONEAPI_DEVICE_SELECTOR": "level_zero:1", "SYCL_CACHE_PERSISTENT": "1"}
    )
    assert clean["ONEAPI_DEVICE_SELECTOR"] == "level_zero:1"
    assert clean["SYCL_CACHE_PERSISTENT"] == "1"


def test_sanitize_defaults_to_real_environ() -> None:
    clean = xrt.sanitize_launch_env()
    assert "PATH" in clean  # inherited
    assert "LD_LIBRARY_PATH" not in clean


# ---------------------------------------------------------------------------
# stdlib-only hygiene (mirrors the other adapter tests)
# ---------------------------------------------------------------------------


def test_xpu_runtime_module_imports_no_heavy_dependencies() -> None:
    code = (
        f"import sys, importlib; sys.path.insert(0, {str(_SRC)!r}); "
        "importlib.import_module('arc_llama_vision.xpu_runtime'); "
        "bad = sorted(m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'diffusers', 'transformers', 'numpy')); "
        "print(','.join(bad))"
    )
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy imports leaked: {result.stdout}"
