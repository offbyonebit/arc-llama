from __future__ import annotations

from unittest.mock import Mock

import pytest

from arc_llama import gpu_ownership, launcher
from arc_llama.failures import StartupFailureError
from arc_llama.launcher import LaunchPlan, LlamaServer


def _owner(tmp_path, monkeypatch, *, gpu="0000:06:00.0", name="llama-server",
           allocation="21987752 KiB", region="system", resident="0 KiB"):
    monkeypatch.setattr(gpu_ownership.sys, "platform", "linux")
    monkeypatch.setattr(gpu_ownership, "_PROC_ROOT", tmp_path)
    proc = tmp_path / "1504"
    (proc / "fdinfo").mkdir(parents=True)
    (proc / "comm").write_text(name + "\n")
    (proc / "cgroup").write_text("0::/user.slice/app.slice/qwen3.8-27b-server.service\n")
    (proc / "fdinfo" / "9").write_text(
        f"drm-driver: xe\ndrm-pdev: {gpu}\ndrm-total-{region}: {allocation}\n"
        f"drm-resident-{region}: {resident}\ndrm-total-cycles-rcs: 123456789\n"
    )
    return proc


@pytest.mark.parametrize("region", ["system", "vram0", "gtt"])
def test_allocated_buffers_block_even_when_nonresident(tmp_path, monkeypatch, region):
    _owner(tmp_path, monkeypatch, region=region)
    with pytest.raises(StartupFailureError) as caught:
        gpu_ownership.check_gpu_ownership("06:00.0")
    assert caught.value.category == "gpu_unavailable"
    assert caught.value.details == {
        "gpu": "06:00.0", "pid": 1504, "service": "qwen3.8-27b-server.service",
        "reason": "external_model_owner",
    }
    assert "qwen3.8-27b-server.service" in caught.value.message


def test_bare_byte_allocation_blocks(tmp_path, monkeypatch):
    _owner(tmp_path, monkeypatch, allocation="22515458048")
    with pytest.raises(StartupFailureError):
        gpu_ownership.check_gpu_ownership("0000:06:00.0")


@pytest.mark.parametrize("changes", [
    {"gpu": "0000:07:00.0"}, {"name": "Xorg"},
    {"allocation": "0 KiB"}, {"allocation": "0"}, {"allocation": "malformed KiB"},
])
def test_unrelated_or_empty_clients_do_not_block(tmp_path, monkeypatch, changes):
    _owner(tmp_path, monkeypatch, **changes)
    gpu_ownership.check_gpu_ownership("0000:06:00.0")


def test_managed_pid_is_excluded_before_eviction(tmp_path, monkeypatch):
    _owner(tmp_path, monkeypatch)
    gpu_ownership.check_gpu_ownership("0000:06:00.0", {1504})


def test_exited_process_and_unreadable_fdinfo_are_tolerated(tmp_path, monkeypatch):
    proc = _owner(tmp_path, monkeypatch)
    (proc / "fdinfo" / "9").unlink()
    (proc / "fdinfo" / "9").mkdir()  # unreadable as a file
    (tmp_path / "999").mkdir()  # exited during enumeration
    gpu_ownership.check_gpu_ownership("0000:06:00.0")


def test_windows_does_not_scan_proc(tmp_path, monkeypatch):
    _owner(tmp_path, monkeypatch)
    monkeypatch.setattr(gpu_ownership.sys, "platform", "win32")
    gpu_ownership.check_gpu_ownership("0000:06:00.0")


def test_launch_conflict_releases_unspawned_resources(tmp_path, monkeypatch):
    _owner(tmp_path / "proc", monkeypatch)
    popen = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    plan = LaunchPlan(argv=["unused"], env={}, gpu_pci_slot="0000:06:00.0",
                      resident_lock_path=tmp_path / "state" / "resident.lock")
    srv = LlamaServer(plan, "target")
    with pytest.raises(StartupFailureError, match="Another llama-server"):
        srv.start(log_dir=tmp_path / "logs")
    popen.assert_not_called()
    assert srv.process is None and srv._resident_lock is None
    assert srv._log_file is None and srv.log_path is None


async def test_conflict_preserves_healthy_resident(tmp_path, monkeypatch):
    from helpers import FakeServer, fake_router

    rt = fake_router(tmp_path, monkeypatch)
    await rt.ensure_active("qwen")
    _owner(tmp_path / "proc", monkeypatch, gpu="0000:04:00.0")

    def preflight(model, gpu, plan, managed_pids=None):
        gpu_ownership.check_gpu_ownership(gpu.pci_slot, managed_pids)

    monkeypatch.setattr("arc_llama.router.preflight_launch", preflight)
    with pytest.raises(StartupFailureError, match="Another llama-server"):
        await rt.ensure_active("gemma")
    assert rt._servers["qwen"].ready
    assert FakeServer.starts == ["qwen"] and not FakeServer.stops


@pytest.mark.parametrize("total,available,reserve", [
    (32768, 4000, 4096), (8192, 1000, 1024), (2048, 500, 512),
])
def test_host_memory_reserve_scales_with_ram(tmp_path, monkeypatch, total, available, reserve):
    monkeypatch.setattr(gpu_ownership.sys, "platform", "linux")
    monkeypatch.setattr(gpu_ownership, "_PROC_ROOT", tmp_path)
    (tmp_path / "meminfo").write_text(
        f"MemTotal: {total * 1024} kB\nMemAvailable: {available * 1024} kB\n"
    )
    assert gpu_ownership.host_memory_pressure() == {
        "available_host_mb": available, "reserved_host_mb": reserve,
    }
    (tmp_path / "meminfo").write_text(
        f"MemTotal: {total * 1024} kB\nMemAvailable: {reserve * 1024} kB\n"
    )
    assert gpu_ownership.host_memory_pressure() is None


@pytest.mark.parametrize("allocation", ["371 MiB", "1 GiB"])
def test_other_drm_memory_units_block(tmp_path, monkeypatch, allocation):
    _owner(tmp_path, monkeypatch, allocation=allocation)
    with pytest.raises(StartupFailureError):
        gpu_ownership.check_gpu_ownership("0000:06:00.0")
