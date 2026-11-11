from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from arc_llama.arch import Backend
from arc_llama.binary import (
    detect_backends,
    detect_llama_server_backend,
    list_devices,
    list_vulkan_devices,
    pick_intel_by_memory,
    resolve_vulkan_index,
)


def _make_fake_binary(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "llama-server"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sibling_lib(name: str) -> str:
    """Return the platform-appropriate shared-library filename for ggml backends.

    *name* is the bare backend name, e.g. ``vulkan`` or ``sycl``.  Windows
    official builds ship ``ggml-vulkan.dll``; Linux ships ``libggml-vulkan.so``.
    """
    if sys.platform == "win32":
        return f"ggml-{name}.dll"
    return f"libggml-{name}.so"


def _system_lib_dir(tmp_path: Path) -> Path:
    """Return the platform-appropriate library directory under *tmp_path*."""
    lib_dir = tmp_path / "lib"
    if sys.platform == "win32":
        lib_dir.mkdir(parents=True, exist_ok=True)
        return lib_dir
    triplet_dir = lib_dir / "x86_64-linux-gnu"
    triplet_dir.mkdir(parents=True, exist_ok=True)
    return triplet_dir


class TestDetectBackends:
    def test_detects_sycl(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"some_prefix ggml_backend_sycl blah")
        assert detect_backends(binary) == {Backend.SYCL}
        assert detect_llama_server_backend(binary) == Backend.SYCL

    def test_detects_vulkan(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"some_prefix ggml_backend_vulkan blah")
        assert detect_backends(binary) == {Backend.VULKAN}
        assert detect_llama_server_backend(binary) == Backend.VULKAN

    def test_detects_both_prefers_sycl(self, tmp_path: Path):
        binary = _make_fake_binary(
            tmp_path,
            b"ggml_backend_sycl ggml_backend_vulkan",
        )
        assert detect_backends(binary) == {Backend.SYCL, Backend.VULKAN}
        assert detect_llama_server_backend(binary) == Backend.SYCL

    def test_no_match_returns_none(self, tmp_path: Path):
        binary = _make_fake_binary(tmp_path, b"cpu only binary with no gpu backends")
        assert detect_backends(binary) == set()
        assert detect_llama_server_backend(binary) is None

    def test_missing_file_returns_empty(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist"
        assert detect_backends(missing) == set()
        assert detect_llama_server_backend(missing) is None

    @pytest.mark.parametrize(
        "marker,expected",
        [
            (b"libsycl.so.7", Backend.SYCL),
            (b"libze_intel_gpu.so.1", Backend.SYCL),
            (b"libvulkan.so.1", Backend.VULKAN),
            (b"vkGetInstanceProcAddr", Backend.VULKAN),
        ],
    )
    def test_alternative_markers(self, tmp_path: Path, marker: bytes, expected: Backend):
        binary = _make_fake_binary(tmp_path, b"header " + marker + b" footer")
        assert detect_backends(binary) == {expected}


def test_detects_vulkan_from_sibling_lib(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe no gpu markers")
    (tmp_path / _sibling_lib("ggml-vulkan")).write_bytes(b"ggml_backend_vulkan")
    assert Backend.VULKAN in detect_backends(binary)
    assert detect_llama_server_backend(binary) == Backend.VULKAN


def test_detects_sycl_from_sibling_lib(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe no gpu markers")
    (tmp_path / _sibling_lib("ggml-sycl")).write_bytes(b"ggml_backend_sycl")
    assert detect_llama_server_backend(binary) == Backend.SYCL


def test_prefers_sycl_across_sibling_libs(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe no gpu markers")
    (tmp_path / _sibling_lib("ggml-sycl")).write_bytes(b"ggml_backend_sycl")
    (tmp_path / _sibling_lib("ggml-vulkan")).write_bytes(b"ggml_backend_vulkan")
    assert detect_backends(binary) == {Backend.SYCL, Backend.VULKAN}
    assert detect_llama_server_backend(binary) == Backend.SYCL


def test_sibling_scan_does_not_break_bare_binary(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe no gpu markers")
    assert detect_backends(binary) == set()
    assert detect_llama_server_backend(binary) is None


def test_backend_name_enumeration_is_not_detected(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe")
    (tmp_path / "libggml.so").write_bytes(b"available backends: cpu sycl vulkan cuda blas")
    assert detect_backends(binary) == set()
    assert detect_llama_server_backend(binary) is None


def test_strong_vulkan_markers_still_detect_in_sibling(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe")
    (tmp_path / _sibling_lib("ggml-vulkan")).write_bytes(
        b"libvulkan.so.1 and vkGetInstanceProcAddr symbol"
    )
    assert detect_llama_server_backend(binary) == Backend.VULKAN


def test_vulkan_build_not_misdetected_as_sycl(tmp_path):
    binary = _make_fake_binary(tmp_path, b"cpu only exe")
    (tmp_path / "libggml.so").write_bytes(b"cpu sycl vulkan cuda")
    (tmp_path / _sibling_lib("ggml-vulkan")).write_bytes(b"libvulkan.so.1 vkGetInstanceProcAddr")
    assert detect_backends(binary) == {Backend.VULKAN}
    assert detect_llama_server_backend(binary) == Backend.VULKAN


def test_detects_vulkan_from_system_lib_dir(tmp_path):
    binary = _make_fake_binary(tmp_path / "bin", b"cpu only exe")
    lib = _system_lib_dir(tmp_path) / _sibling_lib("ggml-vulkan")
    lib.write_bytes(b"libvulkan.so.1 vkGetInstanceProcAddr")
    assert detect_backends(binary) == {Backend.VULKAN}
    assert detect_llama_server_backend(binary) == Backend.VULKAN


def test_detects_sycl_from_system_lib_triplet_dir(tmp_path):
    binary = _make_fake_binary(tmp_path / "bin", b"cpu only exe")
    lib = _system_lib_dir(tmp_path) / _sibling_lib("ggml-sycl")
    lib.write_bytes(b"ggml_backend_sycl")
    assert detect_backends(binary) == {Backend.SYCL}
    assert detect_llama_server_backend(binary) == Backend.SYCL


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shared-library symlinks")
def test_backend_scan_deduplicates_versioned_symlink_aliases(tmp_path, monkeypatch):
    import arc_llama.binary as binary_mod

    binary = _make_fake_binary(tmp_path, b"cpu only exe")
    real = tmp_path / "libggml-sycl.so.0.19.0"
    real.write_bytes(b"ggml_backend_sycl")
    (tmp_path / "libggml-sycl.so.0").symlink_to(real.name)
    (tmp_path / "libggml-sycl.so").symlink_to(real.name)
    scanned: list[Path] = []
    original = binary_mod._scan_file

    def recording_scan(path: Path):
        scanned.append(path)
        return original(path)

    monkeypatch.setattr(binary_mod, "_scan_file", recording_scan)

    assert detect_backends(binary) == {Backend.SYCL}
    assert len(scanned) == 2  # executable plus one shared-library inode


class TestVulkanDeviceResolution:
    """Vulkan enumerates every vendor, so index 0 is not necessarily the Arc.

    Real output from a machine with an RTX 4060 Ti and an Arc Pro B60:

        Vulkan0: NVIDIA GeForce RTX 4060 Ti (16380 MiB, 13404 MiB free)
        Vulkan1: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, ...)

    arc-llama used to pass sycl_index (0) as the Vulkan index and ran the
    model on the NVIDIA card.
    """

    MIXED = [
        (0, "NVIDIA GeForce RTX 4060 Ti"),
        (1, "Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)"),
    ]

    def test_picks_intel_not_index_zero(self):
        assert resolve_vulkan_index(self.MIXED) == 1

    def test_single_intel_device(self):
        assert resolve_vulkan_index([(0, "Intel(R) Arc(tm) A770 Graphics")]) == 0

    def test_no_intel_device_returns_none(self):
        assert resolve_vulkan_index([(0, "NVIDIA GeForce RTX 4060 Ti")]) is None

    def test_empty_returns_none(self):
        assert resolve_vulkan_index([]) is None

    def test_two_intel_cards_disambiguated_by_name(self):
        devices = [
            (0, "Intel(R) Arc(tm) A770 Graphics"),
            (1, "Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)"),
        ]
        assert resolve_vulkan_index(devices, gpu_name="Arc Pro B60 Graphics") == 1

    def test_two_intel_cards_without_a_name_is_ambiguous(self):
        """Ambiguous must be None, not a guess: guessing is the original bug."""
        devices = [
            (0, "Intel(R) Arc(tm) A770 Graphics"),
            (1, "Intel(R) Arc(tm) A770 Graphics"),
        ]
        assert resolve_vulkan_index(devices) is None

    def test_parses_real_list_devices_output(self, monkeypatch, tmp_path):
        """Verbatim output from the b10192 Vulkan build on the mixed-GPU box.

        subprocess.run is stubbed rather than executing a shell script, so this
        runs on Windows too. The parser is what is under test, not exec.
        """
        stdout = (
            "Available devices:\n"
            "  Vulkan0: NVIDIA GeForce RTX 4060 Ti (16380 MiB, 13404 MiB free)\n"
            "  Vulkan1: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)"
            " (24480 MiB, 21994 MiB free)\n"
        )

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        devices = list_vulkan_devices(tmp_path / "llama-server")
        assert devices == [
            (0, "NVIDIA GeForce RTX 4060 Ti"),
            (1, "Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)"),
        ]
        assert resolve_vulkan_index(devices) == 1

    def test_unrunnable_binary_returns_empty(self, tmp_path):
        assert list_vulkan_devices(tmp_path / "does-not-exist") == []


# Captured-style ``--list-devices`` outputs for the device-pick fixtures.
LIST_DEVICES = {
    "igpu_arc": (
        "Available devices:\n"
        "  Vulkan0: Intel(R) UHD Graphics 770 (ADL-S GT1) (23950 MiB, 23950 MiB free)\n"
        "  Vulkan1: Intel(R) Arc(tm) B580 Graphics (BMG G21) (12216 MiB, 11800 MiB free)\n"
    ),
    "arc_nvidia": (
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 4060 Ti (16380 MiB, 13404 MiB free)\n"
        "  Vulkan1: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, 21994 MiB free)\n"
    ),
    "two_arcs_nvidia": (
        "Available devices:\n"
        "  Vulkan0: NVIDIA GeForce RTX 4060 Ti (16380 MiB, 13404 MiB free)\n"
        "  Vulkan1: Intel(R) Arc(tm) B580 Graphics (BMG G21) (12216 MiB, 11800 MiB free)\n"
        "  Vulkan2: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, 21994 MiB free)\n"
    ),
    "two_same_arcs": (
        "  Vulkan0: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, 24000 MiB free)\n"
        "  Vulkan1: Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (24480 MiB, 21994 MiB free)\n"
    ),
    "sycl": (
        "ggml_sycl_init: found 2 SYCL devices:\n"
        "Available devices:\n"
        "  SYCL0: Intel(R) Arc(TM) B580 Graphics (12216 MiB, 12216 MiB free)\n"
        "  SYCL1: Intel(R) UHD Graphics 770 (23950 MiB, 23950 MiB free)\n"
    ),
    "empty": "",
}


def _listed(monkeypatch, tmp_path, key: str, backend: str = "Vulkan"):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=LIST_DEVICES[key], stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return list_devices(tmp_path / "llama-server", backend)


class TestDevicePickFixtures:
    """Name first; when names can't decide among Intel devices, memory size."""

    def test_keeps_total_memory(self, monkeypatch, tmp_path):
        assert _listed(monkeypatch, tmp_path, "arc_nvidia") == [
            (0, "NVIDIA GeForce RTX 4060 Ti", 16380),
            (1, "Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)", 24480),
        ]

    def test_igpu_and_arc(self, monkeypatch, tmp_path):
        devices = _listed(monkeypatch, tmp_path, "igpu_arc")
        names = [(i, n) for i, n, _ in devices]
        assert resolve_vulkan_index(names, gpu_name="Arc B580 Graphics") == 1
        # A fallback name matches nothing; the iGPU reports more memory than
        # the card, so the pick goes by the card's detected VRAM, not size.
        assert resolve_vulkan_index(names, gpu_name="Arc B-series (variant)") is None
        assert pick_intel_by_memory(devices, 12288) == 1

    def test_arc_with_nvidia_listed_first(self, monkeypatch, tmp_path):
        devices = _listed(monkeypatch, tmp_path, "arc_nvidia")
        names = [(i, n) for i, n, _ in devices]
        assert resolve_vulkan_index(names, gpu_name="Arc Pro B-series") == 1

    def test_two_arcs_with_unknown_name(self, monkeypatch, tmp_path):
        devices = _listed(monkeypatch, tmp_path, "two_arcs_nvidia")
        names = [(i, n) for i, n, _ in devices]
        assert resolve_vulkan_index(names, gpu_name="Arc Pro B60 Graphics") == 2
        assert resolve_vulkan_index(names, gpu_name="Arc B580 Graphics") == 1
        assert resolve_vulkan_index(names, gpu_name="Arc Pro B-series") is None
        assert pick_intel_by_memory(devices, 24576) == 2
        assert pick_intel_by_memory(devices, 12288) == 1
        assert pick_intel_by_memory(devices) == 2  # no VRAM known: most memory, never NVIDIA

    def test_identical_cards_stay_unpicked(self, monkeypatch, tmp_path):
        devices = _listed(monkeypatch, tmp_path, "two_same_arcs")
        assert pick_intel_by_memory(devices, 24576) is None
        assert pick_intel_by_memory(devices) is None

    def test_empty_output(self, monkeypatch, tmp_path):
        devices = _listed(monkeypatch, tmp_path, "empty")
        assert devices == []
        assert resolve_vulkan_index([]) is None
        assert pick_intel_by_memory(devices) is None

    def test_parses_sycl_lines_by_prefix(self, monkeypatch, tmp_path):
        assert _listed(monkeypatch, tmp_path, "sycl", "SYCL") == [
            (0, "Intel(R) Arc(TM) B580 Graphics", 12216),
            (1, "Intel(R) UHD Graphics 770", 23950),
        ]
        assert _listed(monkeypatch, tmp_path, "sycl") == []

