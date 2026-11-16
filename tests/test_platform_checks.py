"""Tests for host/driver competitive-inference checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arc_llama.platform_checks import (
    format_bytes,
    max_memory_bar_bytes,
    mesa_upgrade_steps,
    mesa_version,
    old_mesa_vulkan_warning,
    old_windows_arc_driver_warning,
    oneapi_runtime_env_needed,
    oneapi_setvars_path,
    parse_kernel_version,
    parse_mesa_version,
    pcie_link_warning,
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
        # Isolate every loader search path from the developer's oneAPI install.
        monkeypatch.setenv("PATH", "")
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "program-files"))
        # Ensure ldconfig doesn't accidentally find libs on this test runner.
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: None)
        assert oneapi_runtime_env_needed() is True

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows resolves DLLs through PATH")
    def test_runtime_on_the_loader_path_is_enough(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libsvml.so").write_text("")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(lib_dir))
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: None)
        assert oneapi_runtime_env_needed() is False

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows resolves DLLs through PATH")
    def test_runtime_only_on_disk_still_needs_setvars(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        lib_dir = tmp_path / "lib" / "intel64"
        lib_dir.mkdir(parents=True)
        (lib_dir / "libsvml.so").write_text("")
        monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: None)
        assert oneapi_runtime_env_needed() is True

    @pytest.mark.skipif(sys.platform == "win32", reason="ldconfig is Linux-only")
    def test_level_zero_alone_does_not_count_but_svml_does(self, monkeypatch: pytest.MonkeyPatch):
        import subprocess as sp

        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setattr("arc_llama.platform_checks.shutil.which", lambda _x: "/sbin/ldconfig")
        cache = "\tlibze_loader.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libze_loader.so.1\n"
        monkeypatch.setattr("arc_llama.platform_checks.subprocess.run",
                            lambda *a, **k: sp.CompletedProcess(a, 0, stdout=cache, stderr=""))
        assert oneapi_runtime_env_needed() is True
        cache += "\tlibsvml.so (libc6,x86-64) => /opt/intel/oneapi/compiler/2026.1/lib/libsvml.so\n"
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


# Shaped like `vulkaninfo --summary` on an Arc Pro B60 + NVIDIA box (trimmed).
VULKANINFO_SUMMARY = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.318
\tdriverVersion      = 25.2.8
\tvendorID           = 0x8086
\tdeviceID           = 0xe211
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = Intel(R) Graphics (BMG G21)
\tdriverID           = DRIVER_ID_INTEL_OPEN_SOURCE_MESA
\tdriverName         = Intel open-source Mesa driver
\tdriverInfo         = Mesa 25.2.8-0ubuntu0.24.04.1
GPU1:
\tvendorID           = 0x10de
\tdriverName         = NVIDIA
\tdriverInfo         = 580.95.05
"""
GLXINFO_B = "OpenGL vendor string: Intel\nOpenGL version string: 4.6 (Compatibility Profile) Mesa 26.1.2\n"


class TestMesaVersion:
    def test_parse_vulkaninfo_and_glxinfo(self):
        assert parse_mesa_version(VULKANINFO_SUMMARY) == (25, 2)
        assert parse_mesa_version(GLXINFO_B) == (26, 1)
        assert parse_mesa_version("driverInfo = 580.95.05") is None
        assert parse_mesa_version("") is None

    def test_warning_only_below_26_1(self):
        assert "Mesa 25.2" in (old_mesa_vulkan_warning((25, 2)) or "")
        assert old_mesa_vulkan_warning((26, 0)) is not None
        assert old_mesa_vulkan_warning((26, 1)) is None
        assert old_mesa_vulkan_warning((27, 0)) is None
        assert old_mesa_vulkan_warning(None) is None

    def test_reads_vulkaninfo_then_glxinfo(self, monkeypatch):
        import subprocess

        import arc_llama.platform_checks as pc

        monkeypatch.setattr(pc.sys, "platform", "linux")
        monkeypatch.setattr(pc.shutil, "which", lambda name: f"/usr/bin/{name}")
        outputs = {"vulkaninfo": "no mesa here", "glxinfo": GLXINFO_B}
        monkeypatch.setattr(
            pc.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, outputs[cmd[0]], ""),
        )
        assert mesa_version() == (26, 1)

    def test_no_op_without_tools_or_on_windows(self, monkeypatch):
        import arc_llama.platform_checks as pc

        def boom(*a, **k):
            raise AssertionError("must not run a tool")

        monkeypatch.setattr(pc.subprocess, "run", boom)
        monkeypatch.setattr(pc.sys, "platform", "linux")
        monkeypatch.setattr(pc.shutil, "which", lambda name: None)
        assert mesa_version() is None
        monkeypatch.setattr(pc.sys, "platform", "win32")
        monkeypatch.setattr(pc.shutil, "which", lambda name: f"C:/bin/{name}.exe")
        assert mesa_version() is None


@pytest.mark.parametrize("backend,shown", [("vulkan", True), ("sycl", False)])
def test_doctor_warns_on_old_mesa_for_vulkan(tmp_path: Path, monkeypatch, backend, shown):
    from click.testing import CliRunner

    import arc_llama.cli_diagnostics as diagnostics
    from arc_llama.cli import cli
    from arc_llama.config import Config, GPUConfig

    monkeypatch.setattr(diagnostics, "mesa_version", lambda: (25, 2))
    cfg = Config(gpus=[GPUConfig("0000:03:00.0", 0, "battlemage", vram_mb=24576, backend=backend)])
    cfg.save(tmp_path / "config.toml")
    result = CliRunner().invoke(cli, ["-c", str(tmp_path / "config.toml"), "doctor"])
    assert ("mesa_vulkan" in result.output) is shown


@pytest.mark.parametrize(
    "release,base",
    [
        (
            {
                "ID": "ubuntu",
                "ID_LIKE": "debian",
                "VERSION_CODENAME": "noble",
                "UBUNTU_CODENAME": "noble",
            },
            "noble",
        ),
        (
            {
                "ID": "linuxmint",
                "ID_LIKE": "ubuntu debian",
                "VERSION_CODENAME": "zena",
                "UBUNTU_CODENAME": "noble",
            },
            "noble",
        ),
        ({"ID": "fedora"}, None),
        ({}, None),
    ],
)
def test_mesa_upgrade_steps_print_the_ppa_and_undo_for_ubuntu_bases(release, base):
    text = "\n".join(mesa_upgrade_steps(release))
    if base is None:
        assert "package manager" in text and "ppa" not in text
    else:
        assert "sudo add-apt-repository ppa:kisak/kisak-mesa" in text
        assert f"sudo ppa-purge -d {base} ppa:kisak/kisak-mesa" in text
        assert "zena" not in text


def test_os_release_parses_quoted_values(tmp_path):
    from arc_llama.platform_checks import _os_release

    f = tmp_path / "os-release"
    f.write_text('ID=linuxmint\nID_LIKE="ubuntu debian"\nUBUNTU_CODENAME=noble\n# comment\n')
    assert _os_release(f) == {
        "ID": "linuxmint",
        "ID_LIKE": "ubuntu debian",
        "UBUNTU_CODENAME": "noble",
    }
    assert _os_release(tmp_path / "missing") == {}


@pytest.mark.parametrize('speed,width,max_speed,max_width,expected', [
    ('16.0 GT/s PCIe', '8', '16.0 GT/s PCIe', '8', None),
    ('2.5 GT/s PCIe', '1', '16.0 GT/s PCIe', '8', 'Gen1 x1, card supports Gen4 x8'),
    ('16.0 GT/s PCIe', '4', '16.0 GT/s PCIe', '8', 'Gen4 x4, card supports Gen4 x8'),
    ('8.0 GT/s PCIe', '8', '16.0 GT/s PCIe', '8', None),
    ('Unknown speed', '8', '16.0 GT/s PCIe', '8', None),
    ('nan', '8', '16.0 GT/s PCIe', '8', None),
    ('16.0 GT/s PCIe', '0', '16.0 GT/s PCIe', '8', None),
])
def test_pcie_link_from_fake_sysfs(tmp_path, speed, width, max_speed, max_width, expected):
    # A portable component name lets the same parser tests run on Windows.
    device = tmp_path / 'gpu0'
    device.mkdir()
    for name, value in zip(('current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'),
                           (speed, width, max_speed, max_width), strict=True):
        (device / name).write_text(value + '\n')
    warning = pcie_link_warning('gpu0', tmp_path)
    if expected:
        assert expected in warning
    else:
        assert warning is None
    (device / 'max_link_width').unlink()
    assert pcie_link_warning('gpu0', tmp_path) is None
    assert pcie_link_warning('absent', tmp_path) is None


@pytest.mark.parametrize('version,warning', [
    ('31.0.101.9999', True), ('32.0.101.8734', True),
    ('32.0.101.8735', False), ('32.0.101.9000', False),
    ('33.0.0.0', False), ('unknown', False), ('32.0.101', False), (None, False),
])
def test_windows_arc_driver_version_warning(version, warning):
    text = old_windows_arc_driver_warning(version)
    assert (text is not None) is warning
    if warning:
        assert '32.0.101.8735' in text


@pytest.mark.parametrize('windows', [False, True])
def test_doctor_link_and_driver_checks_are_platform_specific_warnings(tmp_path, monkeypatch, windows):
    from io import StringIO
    from types import SimpleNamespace

    import click
    from click.testing import CliRunner
    from rich.console import Console

    import arc_llama.cli_diagnostics as diagnostics
    import arc_llama.plugins as plugins
    from arc_llama.agent import landlock
    from arc_llama.arch import Arch

    gpu = SimpleNamespace(name='Test Arc', pci_slot='gpu0', device_id=0x56A0,
                          arch=Arch.ALCHEMIST, driver='32.0.101.8734',
                          vram_gb=16, notes=[], sysfs_path='')
    calls = []
    def link(slot):
        calls.append(slot)
        return 'gpu0 running at Gen1 x1, card supports Gen4 x8'
    monkeypatch.setattr(diagnostics, 'pcie_link_warning', link)
    monkeypatch.setattr(diagnostics.sys, 'platform', 'win32' if windows else 'linux')
    monkeypatch.setattr(diagnostics, 'kernel_module_loaded', lambda name: True)
    monkeypatch.setattr(diagnostics, 'level_zero_loader_present', lambda: (True, 'fake loader'))
    monkeypatch.setattr(diagnostics, 'user_in_groups', lambda *names: dict.fromkeys(names, True))
    monkeypatch.setattr(diagnostics, 'oneapi_setvars_path', lambda: None)
    monkeypatch.setattr(diagnostics.shutil, 'which', lambda name: None)
    monkeypatch.setattr(diagnostics.platform, 'uname', lambda: SimpleNamespace(release='6.14.0'))
    monkeypatch.setattr(diagnostics.platform, 'release', lambda: 'test')
    monkeypatch.setattr(landlock, 'status', lambda: 'available')
    monkeypatch.setattr(landlock, 'abi_version', lambda: 6)
    monkeypatch.setattr(plugins, 'plugin_health', lambda: [])
    output = StringIO()
    @click.group()
    @click.pass_context
    def group(ctx):
        ctx.obj = {'config_path': tmp_path / 'absent.toml'}
    diagnostics.register_diagnostic_commands(
        group, console=Console(file=output, width=180), load_config=lambda _: None,
        detect_gpus=lambda **kw: [gpu], is_windows=lambda: windows,
    )
    result = CliRunner().invoke(group, ['doctor'])
    assert result.exit_code == 0, result.exception
    text = output.getvalue()
    assert ('windows_arc_driver:' in text) is windows
    assert ('pcie_link:' in text) is not windows
    assert calls == ([] if windows else ['gpu0'])
    assert 'FAIL' not in text
    assert ('Update the Arc graphics driver from Intel' in text) is windows


def _write_pcie_link(path, speed, width, max_speed, max_width):
    path.mkdir(parents=True, exist_ok=True)
    for name, value in zip(('current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'),
                           (speed, width, max_speed, max_width), strict=True):
        (path / name).write_text(str(value))


def test_pcie_checks_card_upstream_port_instead_of_internal_switch_link(tmp_path):
    root = tmp_path / 'root-port'
    upstream = root / 'card-upstream'
    downstream = upstream / 'card-downstream'
    gpu = downstream / 'gpu'
    _write_pcie_link(root, '8.0 GT/s PCIe', 4, '8.0 GT/s PCIe', 4)
    _write_pcie_link(upstream, '8.0 GT/s PCIe', 4, '16.0 GT/s PCIe', 8)
    _write_pcie_link(downstream, '16.0 GT/s PCIe', 16, '16.0 GT/s PCIe', 16)
    _write_pcie_link(gpu, '16.0 GT/s PCIe', 16, '16.0 GT/s PCIe', 16)
    for bridge in (root, upstream, downstream):
        (bridge / 'class').write_text('0x060400')
    for bridge in (upstream, downstream):
        (bridge / 'vendor').write_text('0x8086')
    (gpu / 'class').write_text('0x030000')
    slot = str(gpu.relative_to(tmp_path))
    assert 'Gen3 x4, card supports Gen4 x8' in pcie_link_warning(slot, tmp_path)
    # The internal endpoint and root port are both full width. Only the card's
    # upstream port reveals the slot-width restriction, and idle speed is benign.
    (upstream / 'current_link_width').write_text('8')
    (upstream / 'current_link_speed').write_text('2.5 GT/s PCIe')
    assert pcie_link_warning(slot, tmp_path) is None
    (upstream / 'current_link_width').unlink()
    assert pcie_link_warning(slot, tmp_path) is None


def test_direct_gpu_uses_its_own_slot_facing_link(tmp_path):
    root = tmp_path / 'root-port'
    gpu = root / 'gpu'
    _write_pcie_link(root, '16.0 GT/s PCIe', 8, '16.0 GT/s PCIe', 8)
    (root / 'class').write_text('0x060400')
    _write_pcie_link(gpu, '16.0 GT/s PCIe', 4, '16.0 GT/s PCIe', 8)
    assert 'Gen4 x4, card supports Gen4 x8' in pcie_link_warning(str(gpu.relative_to(tmp_path)), tmp_path)


def test_pcie_stops_at_card_above_non_intel_chipset(tmp_path):
    root = tmp_path / 'cpu-root'
    chipset = root / 'chipset-switch'
    upstream = chipset / 'card-upstream'
    downstream = upstream / 'card-downstream'
    gpu = downstream / 'gpu'
    for bridge in (root, chipset, upstream, downstream):
        _write_pcie_link(bridge, '8.0 GT/s PCIe', 4, '8.0 GT/s PCIe', 4)
        (bridge / 'class').write_text('0x060400')
        (bridge / 'vendor').write_text('0x1022' if bridge == chipset else '0x8086')
    _write_pcie_link(upstream, '2.5 GT/s PCIe', 1, '16.0 GT/s PCIe', 8)
    _write_pcie_link(gpu, '16.0 GT/s PCIe', 16, '16.0 GT/s PCIe', 16)
    assert 'Gen1 x1, card supports Gen4 x8' in pcie_link_warning(str(gpu.relative_to(tmp_path)), tmp_path)

    # Even an all-Intel motherboard must not extend the walk past two levels.
    (chipset / 'vendor').write_text('0x8086')
    assert 'Gen1 x1, card supports Gen4 x8' in pcie_link_warning(str(gpu.relative_to(tmp_path)), tmp_path)
