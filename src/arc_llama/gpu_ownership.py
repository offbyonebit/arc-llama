"""Detect external llama-server allocations on a Linux DRM GPU.

The resident lock coordinates participating launchers. DRM fdinfo also finds
legacy services that bypass that lock, including buffers evicted to system
memory. This is a best-effort admission check, not an OS-wide GPU reservation.
"""
from __future__ import annotations

import sys
from pathlib import Path

from arc_llama.failures import StartupFailureError

_PROC_ROOT = Path("/proc")


def _pci_slot(value: str) -> str:
    value = value.strip().lower()
    return f"0000:{value}" if value.count(":") == 1 else value


def check_gpu_ownership(pci_slot: str, ignored_pids: set[int] | None = None) -> None:
    """Refuse a load when another readable llama-server owns this GPU.

    Only model servers with positive allocation totals on the exact PCI device
    count. Other GPUs, desktop DRM clients, and empty handles are unaffected.
    Windows does not expose Linux DRM fdinfo; its shutdown checks still apply.
    """
    if sys.platform != "linux":
        return
    ignored = ignored_pids or set()
    try:
        processes = list(_PROC_ROOT.iterdir())
    except OSError:
        return
    for process in processes:
        if not process.name.isdecimal() or int(process.name) in ignored:
            continue
        try:
            if not (process / "comm").read_text().strip().startswith("llama-server"):
                continue
            handles = list((process / "fdinfo").iterdir())
        except OSError:
            # A process may exit during inspection, or belong to another user.
            continue
        for handle in handles:
            try:
                with handle.open() as stream:
                    text = stream.read(16_384)
            except OSError:
                continue
            fields = dict(
                line.split(":", 1) for line in text.splitlines() if ":" in line
            )
            if _pci_slot(fields.get("drm-pdev", "")) != _pci_slot(pci_slot):
                continue
            allocated = False
            for key, value in fields.items():
                if key.startswith("drm-total-") and key != "drm-total-cycles":
                    # Memory accounting values have a byte-unit suffix. Cycle
                    # counters and resident bytes are not allocation totals.
                    parts = value.split()
                    if len(parts) == 2 and parts[1] in {"B", "KiB", "MiB", "GiB"}:
                        try:
                            allocated |= int(parts[0]) > 0
                        except ValueError:
                            continue
            if not allocated:
                continue
            service = None
            try:
                service = next(
                    (part for part in (process / "cgroup").read_text().split("/")
                     if part.strip().endswith(".service")), None
                )
                if service:
                    service = service.strip()
            except OSError:
                pass
            pid = int(process.name)
            owner = f"{service} (PID {pid})" if service else f"PID {pid}"
            raise StartupFailureError(
                "gpu_unavailable",
                f"Another llama-server owns allocations on GPU {pci_slot}: {owner}.",
                "Stop the conflicting model server before loading this model. "
                "Disable its automatic startup if Arc Llama should own this GPU.",
                details={"gpu": pci_slot, "pid": pid, "service": service,
                         "reason": "external_model_owner"},
            )


def host_memory_pressure() -> dict[str, int] | None:
    """Detect critically low available Linux RAM without querying the GPU.

    Keep an eighth of host RAM available during loading, bounded to
    512 MiB–4 GiB. Missing OS counters disable this best-effort check.
    """
    if sys.platform != "linux":
        return None
    try:
        fields = {}
        for line in (_PROC_ROOT / "meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemAvailable", "MemTotal"}:
                fields[key] = int(value.split()[0]) // 1024
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    reserve = max(512, min(4096, total // 8))
    if available >= reserve:
        return None
    return {"available_host_mb": available, "reserved_host_mb": reserve}
