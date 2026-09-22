from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arc_llama.config import Config, GPUConfig, ModelConfig


@pytest.fixture
def make_sysfs_gpu(tmp_path: Path):
    """Factory fixture that creates fake sysfs PCI device entries.

    Simulates Linux's /sys/bus/pci/devices/<slot> layout, which only exists
    on Linux. The slot name contains colons (e.g. "0000:03:00.0"), which are
    illegal in Windows path components, so this is skipped there rather than
    rewritten — there's no Windows sysfs equivalent to fake.
    """
    if sys.platform == "win32":
        pytest.skip("simulates Linux-only /sys/bus/pci layout")

    def _make(slot: str, device_id: int = 0xE211, vram_bytes: int | None = None, driver: str = "xe"):
        base = tmp_path / "sys" / "bus" / "pci" / "devices" / slot
        base.mkdir(parents=True)
        (base / "vendor").write_text("0x8086\n")
        (base / "device").write_text(f"0x{device_id:04X}\n")
        (base / "class").write_text("0x030000\n")
        if driver:
            drv = base / "driver"
            drv.mkdir()
            (drv / "name").write_text(f"{driver}\n")
            # symlink from driver to device is created by the kernel; we don't need it
        else:
            # No driver bound
            pass
        if vram_bytes is not None:
            # Create a fake drm card with VRAM info
            drm = base / "drm" / "card0"
            drm.mkdir(parents=True)
            (drm / "device").mkdir(parents=True, exist_ok=True)
            (drm / "device" / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
            (drm / "device" / "mem_info_vram_used").write_text("0\n")
            # detect.py also looks at the PCI device path directly
            (base / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
        return base
    return _make


def make_config(tmp_path: Path, *, single_resident: bool = True) -> Config:
    cfg = Config()
    cfg.server.single_resident = single_resident
    cfg.paths.llama_server = "/usr/bin/llama-server"
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.gpus = [
        GPUConfig(
            pci_slot="0000:03:00.0",
            sycl_index=0,
            arch="battlemage",
            vram_mb=24576,
            name="Arc Pro B60",
        ),
        GPUConfig(
            pci_slot="0000:04:00.0",
            sycl_index=1,
            arch="alchemist",
            vram_mb=16384,
            name="Arc A770",
        ),
    ]
    cfg.models = [
        ModelConfig(
            name="qwen",
            display_name="Qwen 3",
            path=str(tmp_path / "models" / "Qwen3-7B-Q4_K_M.gguf"),
            port=18080,
            gpu_pci_slot="0000:03:00.0",
            aliases=["qwen.gguf"],
            recipe={
                "ctx": 8192,
                "n_gpu_layers": 999,
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        ),
        ModelConfig(
            name="gemma",
            display_name="Gemma",
            path=str(tmp_path / "models" / "gemma-3-4b-Q4_K_M.gguf"),
            port=18081,
            gpu_pci_slot="0000:04:00.0",
            aliases=["gemma.gguf"],
            recipe={
                "ctx": 8192,
                "n_gpu_layers": 999,
                "parallel": 1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        ),
    ]
    return cfg


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path: Path, monkeypatch):
    """Redirect XDG dirs to a temp path so the suite never touches the developer's real config."""
    config_home = tmp_path / ".config"
    config_home.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(config_home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))


@pytest.fixture(autouse=True)
def _isolate_fake_servers_from_host_preflight(monkeypatch):
    """Most router tests use nonexistent model/runtime paths by design.

    Host prerequisite checks have focused coverage in test_preflight.py.
    Individual integration tests can replace this stub when they need to
    assert Router's ordering around a preflight failure.
    """
    monkeypatch.setattr("arc_llama.router.preflight_launch", lambda *_args: None)


@pytest.fixture
def base_config(tmp_path: Path) -> Config:
    """A populated Config using temp paths, suitable for CLI tests."""
    return make_config(tmp_path)
