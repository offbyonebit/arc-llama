"""Reading llama-server's startup log for the device a model actually loaded on."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.launcher import (
    LlamaServer,
    build_plan,
    startup_device_summary,
    startup_device_warning,
)

LOGS = Path(__file__).parent / "fixtures" / "startup-logs"
B60 = "Arc Pro B60"


def _log(name: str) -> str:
    return (LOGS / name).read_text(encoding="utf-8")


def test_gpu_offloaded_load_does_not_warn():
    assert startup_device_warning(_log("vulkan-b60-offloaded.txt"), B60) is None


def test_zero_layers_offloaded_warns_cpu():
    warning = startup_device_warning(_log("cpu-fallback.txt"), B60)
    assert warning is not None
    assert "0/65 layers" in warning and "CPU" in warning


def test_cpu_only_model_buffers_warn_without_offload_line():
    text = "load_tensors:   CPU_Mapped model buffer size = 18553.77 MiB\n"
    assert "CPU memory" in (startup_device_warning(text, B60) or "")


def test_other_vendor_device_warns():
    warning = startup_device_warning(_log("wrong-device-nvidia.txt"), B60)
    assert warning is not None
    assert "Vulkan0 (NVIDIA GeForce RTX 4060 Ti)" in warning


def test_intel_igpu_instead_of_configured_arc_warns():
    text = (
        "llama_model_load_from_file_impl: using device Vulkan0 "
        "(Intel(R) Graphics (ARL)) (unknown id) - 7000 MiB free\n"
        "load_tensors: offloaded 65/65 layers to GPU\n"
    )
    assert "Intel(R) Graphics (ARL)" in (startup_device_warning(text, B60) or "")


def test_other_arc_model_warns_but_sycl_b60_does_not():
    a770 = "llama_model_load_from_file_impl: using device SYCL0 (Intel(R) Arc(TM) A770 Graphics) - 15000 MiB free\n"
    b60 = "llama_model_load_from_file_impl: using device SYCL0 (Intel(R) Arc(TM) Pro B60 Graphics) - 23000 MiB free\n"
    assert startup_device_warning(a770, B60) is not None
    assert startup_device_warning(b60, B60) is None


def test_fallback_gpu_names_and_split_loads_do_not_warn_falsely():
    two = (
        "llama_model_load_from_file_impl: using device Vulkan0 (Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)) (0000:03:00.0) - 23000 MiB free\n"
        "llama_model_load_from_file_impl: using device Vulkan1 (Intel(R) Arc(tm) A770 Graphics (DG2)) (0000:04:00.0) - 15000 MiB free\n"
    )
    assert startup_device_warning(two, B60) is None
    assert startup_device_warning(_log("vulkan-b60-offloaded.txt"), "Arc B-series (variant)") is None
    assert startup_device_warning(_log("vulkan-b60-offloaded.txt"), "") is None


def test_log_without_device_lines_does_not_warn():
    assert startup_device_warning(_log("no-device-lines.txt"), B60) is None
    assert startup_device_warning("", B60) is None


def test_draft_model_on_cpu_does_not_warn():
    text = _log("vulkan-b60-offloaded.txt") + "load_tensors: offloaded 0/29 layers to GPU\n"
    assert startup_device_warning(text, B60) is None


def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LlamaServer:
    plan = build_plan(
        Config(paths=type("P", (), {"llama_server": "/bin/llama-server"})()),
        ModelConfig(name="m", path="/m.gguf", port=18080, gpu_pci_slot="00:00.0"),
        GPUConfig(pci_slot="00:00.0", sycl_index=0, arch="battlemage", name=B60),
    )
    srv = LlamaServer(plan, name="m")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # An earlier run's CPU fallback stays in the appended log; it must be ignored.
    (log_dir / "m.log").write_text(_log("cpu-fallback.txt"), encoding="utf-8")

    class FakeProc:
        pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    srv.start(log_dir=log_dir)
    return srv


async def _healthy(self, url):
    return type("R", (), {"status_code": 200, "json": lambda self: {"status": "ok"}})()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture", "warns"),
    [("vulkan-b60-offloaded.txt", False), ("cpu-fallback.txt", True), ("no-device-lines.txt", False)],
)
async def test_wait_ready_warns_from_this_runs_log_only(tmp_path, monkeypatch, caplog, fixture, warns):
    import httpx

    srv = _server(tmp_path, monkeypatch)
    try:
        srv._log_file.write(_log(fixture).encode())
        srv._log_file.flush()
        monkeypatch.setattr(httpx.AsyncClient, "get", _healthy)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="arc_llama.launcher"):
            assert await srv.wait_ready(timeout=2.0) is True
        warned = [
            r for r in caplog.records if r.levelno == logging.WARNING and r.getMessage().startswith("[m]")
        ]
        assert bool(warned) is warns
    finally:
        srv.process = None
        srv.stop()


@pytest.mark.asyncio
async def test_unreadable_log_never_fails_the_load(tmp_path, monkeypatch):
    import httpx

    srv = _server(tmp_path, monkeypatch)
    try:
        srv._log_path = tmp_path / "missing.log"
        monkeypatch.setattr(httpx.AsyncClient, "get", _healthy)
        assert await srv.wait_ready(timeout=2.0) is True
    finally:
        srv.process = None
        srv.stop()


def test_summary_names_device_and_layers():
    assert startup_device_summary(_log("vulkan-b60-offloaded.txt")) == (
        " on Vulkan1: Intel(R) Arc(TM) Pro B60 Graphics, 33/33 layers on GPU"
    )
    assert startup_device_summary(_log("cpu-fallback.txt")) == ", 0/65 layers on GPU"
    assert startup_device_summary(_log("no-device-lines.txt")) is None


@pytest.mark.asyncio
async def test_wait_ready_logs_the_device_at_info(tmp_path, monkeypatch, caplog):
    import httpx

    srv = _server(tmp_path, monkeypatch)
    try:
        srv._log_file.write(_log("vulkan-b60-offloaded.txt").encode())
        srv._log_file.flush()
        monkeypatch.setattr(httpx.AsyncClient, "get", _healthy)
        with caplog.at_level(logging.INFO, logger="arc_llama.launcher"):
            assert await srv.wait_ready(timeout=2.0) is True
        assert (
            "Loaded m on Vulkan1: Intel(R) Arc(TM) Pro B60 Graphics, 33/33 layers on GPU"
            in caplog.messages
        )
    finally:
        srv.process = None
        srv.stop()
