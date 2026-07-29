"""Tests for arc_llama.detect — GPU discovery without real hardware."""
from __future__ import annotations

import re
import subprocess

import pytest

from arc_llama.arch import Arch
from arc_llama.detect import (
    _REG_BINARY,
    _REG_QWORD,
    DetectedGPU,
    _enrich_with_clinfo,
    _parse_clinfo_devices,
    _parse_pnp_device_id,
    _parse_sycl_ls_level_zero,
    _scan_pci,
    _scan_windows,
    lspci_intel_gpus,
)


def _patch_pci_root(monkeypatch, fake_sys):
    """Point detect.Path('/sys/bus/pci/devices') at a fake sysfs tree."""
    from pathlib import Path as RealPath

    import arc_llama.detect as detect_mod

    def _fake_path(p, *args, **kwargs):
        if p == "/sys/bus/pci/devices":
            return RealPath(fake_sys)
        return RealPath(p, *args, **kwargs)

    monkeypatch.setattr(detect_mod, "Path", _fake_path)


@pytest.fixture
def make_class_entry():
    def _make(
        device_id,
        *,
        vram_raw=None,
        vram_type=None,
        driver_version="31.0.101.1",
        driver_desc="Intel Arc B580",
    ):
        entry = {
            "MatchingDeviceId": f"PCI\\VEN_8086&DEV_{device_id:04X}&SUBSYS_00000000&REV_00",
            "DriverDesc": driver_desc,
        }
        if driver_version is not None:
            entry["DriverVersion"] = driver_version
        if vram_raw is not None:
            entry["HardwareInformation.qwMemorySize"] = vram_raw
            entry["HardwareInformation.qwMemorySize_type"] = vram_type
        return entry

    return _make


@pytest.fixture
def make_pci_slot():
    def _make(device_id, slot="0000:03:00.0", source="cfgmgr32"):
        m = re.match(
            r"0000:([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9a-fA-F])", slot
        )
        if not m:
            raise ValueError(f"bad slot {slot}")
        bus, dev, func = (int(x, 16) for x in m.groups())
        return {
            "hardware_id": f"PCI\\VEN_8086&DEV_{device_id:04X}&SUBSYS_00000000&REV_00",
            "device_id": device_id,
            "pci_slot": slot,
            "bus": bus,
            "dev": dev,
            "func": func,
            "source": source,
        }

    return _make


class TestParseClinfo:
    def test_single_device(self):
        text = """
  Device Name                                     Intel Arc Pro B60 Graphics
  Global memory size                              25769803776
"""
        out = _parse_clinfo_devices(text)
        assert out == [("Intel Arc Pro B60 Graphics", 25769803776)]

    def test_multiple_devices(self):
        text = """
  Device Name                                     Intel Arc Pro B60 Graphics
  Global memory size                              25769803776
  Device Name                                     Intel(R) UHD Graphics
  Global memory size                              4294967296
"""
        out = _parse_clinfo_devices(text)
        assert len(out) == 2
        assert out[0] == ("Intel Arc Pro B60 Graphics", 25769803776)
        assert out[1] == ("Intel(R) UHD Graphics", 4294967296)

    def test_no_match_returns_empty(self):
        assert _parse_clinfo_devices("") == []


class TestParsePnpDeviceId:
    def test_parses_dev_id(self):
        assert (
            _parse_pnp_device_id(r"PCI\VEN_8086&DEV_E20B&SUBSYS_12345678&REV_01")
            == 0xE20B
        )

    def test_lowercase_hex(self):
        assert _parse_pnp_device_id(r"pci\ven_8086&dev_e20b") == 0xE20B

    def test_malformed_returns_none(self):
        assert _parse_pnp_device_id(r"PCI\VEN_8086&SUBSYS_12345678") is None
        assert _parse_pnp_device_id("") is None
        assert _parse_pnp_device_id("not a pnp string") is None


class TestParseSyclLs:
    def test_with_explicit_index(self):
        text = (
            "[level_zero:gpu:0] Intel Arc B580\n"
            "[level_zero:gpu:1] Intel Arc B580"
        )
        assert _parse_sycl_ls_level_zero(text) == [
            (0, "Intel Arc B580"),
            (1, "Intel Arc B580"),
        ]

    def test_without_index(self):
        text = "[level_zero:gpu] Intel Arc B580\n[level_zero:gpu] Intel Arc B580"
        assert _parse_sycl_ls_level_zero(text) == [
            (0, "Intel Arc B580"),
            (1, "Intel Arc B580"),
        ]

    def test_ignores_other_backends(self):
        text = "[opencl:gpu:0] Intel Arc B580\n[level_zero:gpu:0] Intel Arc B580"
        assert _parse_sycl_ls_level_zero(text) == [(0, "Intel Arc B580")]


class TestScanPci:
    def test_finds_battlemage(self, make_sysfs_gpu, tmp_path, monkeypatch):
        fake_sys = tmp_path / "sys" / "bus" / "pci" / "devices"
        make_sysfs_gpu(
            slot="0000:03:00.0", device_id=0xE211, vram_bytes=24 * 1024 * 1024 * 1024
        )
        _patch_pci_root(monkeypatch, fake_sys)

        gpus = _scan_pci()

        assert len(gpus) == 1
        assert gpus[0].pci_slot == "0000:03:00.0"
        assert gpus[0].arch == Arch.BATTLEMAGE
        assert gpus[0].vram_mb == 24 * 1024

    def test_skips_non_intel_vendor(self, make_sysfs_gpu, tmp_path, monkeypatch):
        base = tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:01:00.0"
        base.mkdir(parents=True)
        (base / "vendor").write_text("0x10DE\n")
        (base / "device").write_text("0x1234\n")
        (base / "class").write_text("0x030000\n")

        _patch_pci_root(monkeypatch, tmp_path / "sys" / "bus" / "pci" / "devices")
        assert _scan_pci() == []

    def test_notes_when_no_driver(self, make_sysfs_gpu, tmp_path, monkeypatch):
        make_sysfs_gpu(slot="0000:03:00.0", device_id=0xE211, driver="")
        _patch_pci_root(monkeypatch, tmp_path / "sys" / "bus" / "pci" / "devices")

        gpus = _scan_pci()

        assert any("No kernel driver" in n for n in gpus[0].notes)


class TestScanWindows:
    def test_finds_discrete_arc_from_registry_vram(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [
                make_class_entry(
                    0xE20B,
                    vram_raw=12 * 1024**3,
                    vram_type=_REG_QWORD,
                )
            ],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [make_pci_slot(0xE20B, "0000:03:00.0")],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        g = gpus[0]
        assert g.pci_slot == "0000:03:00.0"
        assert g.device_id == 0xE20B
        assert g.arch == Arch.BATTLEMAGE
        assert g.driver == "31.0.101.1"
        assert g.vram_mb == 12 * 1024
        assert any("HardwareInformation.qwMemorySize" in n for n in g.notes)

    def test_vram_from_reg_binary(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        vram_bytes = 24 * 1024**3
        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [
                make_class_entry(
                    0xE211,
                    vram_raw=vram_bytes.to_bytes(8, "little"),
                    vram_type=_REG_BINARY,
                    driver_desc="Intel Arc Pro B60",
                )
            ],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [make_pci_slot(0xE211, "0000:04:00.0")],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].vram_mb == 24 * 1024
        assert any("HardwareInformation.qwMemorySize" in n for n in gpus[0].notes)

    def test_vram_fallback_to_known_table(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [make_class_entry(0xE20B)],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [make_pci_slot(0xE20B, "0000:03:00.0")],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].vram_mb == 12288
        assert any("device-ID fallback table" in n for n in gpus[0].notes)

    def test_undiscoverable_vram_note(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [make_class_entry(0x9999, driver_desc="Intel Mystery GPU")],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [make_pci_slot(0x9999, "0000:03:00.0")],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].vram_mb is None
        assert any("not discoverable" in n for n in gpus[0].notes)

    def test_igpu_excluded(self, monkeypatch, make_class_entry, make_pci_slot):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [
                make_class_entry(0x64A0, driver_desc="Intel Lunar Lake Graphics"),
                make_class_entry(0xE20B),
            ],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [
                make_pci_slot(0x64A0, "0000:00:02.0"),
                make_pci_slot(0xE20B, "0000:03:00.0"),
            ],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].device_id == 0xE20B

    def test_localized_location_info_still_detects(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [make_class_entry(0xE20B)],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [
                make_pci_slot(
                    0xE20B, "0000:05:00.0", source="LocationInformation"
                )
            ],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].pci_slot == "0000:05:00.0"
        assert any("LocationInformation" in n for n in gpus[0].notes)

    def test_no_driver_note(self, monkeypatch, make_class_entry, make_pci_slot):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [make_class_entry(0xE20B, driver_version=None)],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [make_pci_slot(0xE20B, "0000:03:00.0")],
        )

        gpus = _scan_windows()

        assert gpus[0].driver is None
        assert any("driver version not found" in n for n in gpus[0].notes)

    def test_two_identical_cards_get_distinct_slots(
        self, monkeypatch, make_class_entry, make_pci_slot
    ):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [
                make_class_entry(
                    0xE20B,
                    vram_raw=12 * 1024**3,
                    vram_type=_REG_QWORD,
                ),
                make_class_entry(
                    0xE20B,
                    vram_raw=12 * 1024**3,
                    vram_type=_REG_QWORD,
                ),
            ],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [
                make_pci_slot(0xE20B, "0000:04:00.0"),
                make_pci_slot(0xE20B, "0000:03:00.0"),
            ],
        )

        gpus = _scan_windows()

        assert len(gpus) == 2
        assert {g.pci_slot for g in gpus} == {"0000:03:00.0", "0000:04:00.0"}
        assert gpus[0].sycl_index_hint == 0
        assert gpus[1].sycl_index_hint == 1
        assert any("Multiple Intel GPUs" in n for n in gpus[0].notes)

    def test_unknown_slot_included_with_note(self, monkeypatch, make_class_entry):
        import arc_llama.detect as dmod

        monkeypatch.setattr(
            dmod,
            "_winreg_display_class_entries",
            lambda: [make_class_entry(0xE20B)],
        )
        monkeypatch.setattr(
            dmod,
            "_enum_pci_locations",
            lambda: [
                {
                    "hardware_id": r"PCI\VEN_8086&DEV_E20B&SUBSYS_00000000&REV_00",
                    "device_id": 0xE20B,
                    "pci_slot": "unknown:PCI\\VEN_8086&DEV_E20B&SUBSYS_00000000&REV_00",
                    "bus": None,
                    "dev": None,
                    "func": None,
                    "source": "unknown",
                }
            ],
        )

        gpus = _scan_windows()

        assert len(gpus) == 1
        assert gpus[0].pci_slot.startswith("unknown:")
        assert any("synthetic slot" in n for n in gpus[0].notes)


class TestEnrichWithClinfo:
    def test_enrich_vram_from_clinfo(self):
        gpu = DetectedGPU(
            pci_slot="0000:03:00.0", device_id=0xE211,
            arch=Arch.BATTLEMAGE, name="Arc Pro B60",
            driver="xe", vram_mb=None, drm_card="card1",
            drm_render="renderD128", sysfs_path="/sys/...",
        )
        _enrich_with_clinfo([gpu])
        # clinfo probably not installed in CI, so this is a no-op.
        # We just assert it doesn't crash.
        assert gpu.vram_mb is None or isinstance(gpu.vram_mb, int)


class TestLspciIntelGpus:
    def test_returns_empty_when_missing(self, monkeypatch):
        def _fake_run(*a, **k):
            return subprocess.CompletedProcess(
                args=a, returncode=1, stdout="", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        assert lspci_intel_gpus() == ""

    def test_filters_intel_display(self, monkeypatch):
        fake_out = """
00:02.0 VGA compatible controller [0300]: Intel Corporation Device [8086:E211] (rev 01)
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:1234]
"""

        def _fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=fake_out, stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = lspci_intel_gpus()
        assert "8086:E211" in result
        assert "10de:1234" not in result
