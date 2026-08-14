"""Discover Intel GPUs on the local machine.

Detection is layered so we work even on minimal systems:

  Linux:
    1. Read `/sys/bus/pci/devices/` directly — fast, root-free, always available.
    2. Cross-reference with `/sys/class/drm/cardN` to identify the kernel driver
       (`i915`, `xe`) and pull VRAM size from `device/mem_info_vram_total`.
    3. Optionally call `clinfo` if installed, to enrich with OpenCL platform info.

  Windows (no pywin32 / WMI / new pip dependencies):
    1. `winreg` reads the display-adapter class key for PCI IDs, the marketing
       name, the Intel driver version string, and the
       `HardwareInformation.qwMemorySize` VRAM value.
    2. `cfgmgr32.dll` reads the numeric device-node registry properties for the
       PCI bus number and address; the address packs device/function.  This is
       locale-independent, so a German or Japanese `LocationInformation` string
       cannot silently drop GPUs.  The string is kept only as a last-resort
       fallback, with a note.

We never assume `clinfo`, `sycl-ls`, or oneAPI are available — `arc-llama doctor`
checks for those separately and tells the user how to install them.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from arc_llama.arch import Arch, arch_for_device_id, known_vram_mib

INTEL_VENDOR_ID = 0x8086
PCI_CLASS_VGA = 0x030000      # 0x03_00_00 — VGA-compatible controller
PCI_CLASS_DISPLAY = 0x038000  # 0x03_80_00 — other display controller

# Mirrors of winreg type constants; defined here so VRAM decoding can run
# without importing winreg on non-Windows paths.
_REG_QWORD = 11
_REG_BINARY = 3


@dataclass
class DetectedGPU:
    """One Intel GPU discovered on this host."""
    pci_slot: str          # e.g. "0000:03:00.0"
    device_id: int         # PCI device ID (e.g. 0xE212 for Arc Pro B60)
    arch: Arch
    name: str              # marketing name from arch table or fallback
    driver: str | None     # kernel driver bound: "xe", "i915", or None
    vram_mb: int | None    # total VRAM in MiB if discoverable, else None
    drm_card: str | None   # e.g. "card1"
    drm_render: str | None # e.g. "renderD128"
    sysfs_path: str        # /sys/bus/pci/devices/<slot>
    notes: list[str] = field(default_factory=list)

    @property
    def vram_gb(self) -> float | None:
        return None if self.vram_mb is None else round(self.vram_mb / 1024, 1)

    @property
    def sycl_index_hint(self) -> int:
        """Best-guess index to feed into ONEAPI_DEVICE_SELECTOR=level_zero:N.

        On Linux this is discovery order. On Windows it is PCI slot order by
        default, but if `sycl-ls` is available and ``enrich=True`` we replace it
        with the actual Level Zero ordinal parsed from ``sycl-ls`` output.
        Override at runtime if you have multiple Intel GPUs and the wrong one
        gets picked.
        """
        return getattr(self, "_index", 0)


# ---------------------------------------------------------------------------
# cfgmgr32 helpers (Windows only; never imported at module top level).
# ---------------------------------------------------------------------------

def _cfgmgr32_bdf(instance_id: str) -> tuple[int, int, int] | None:
    """Return (bus, device, function) for a PNP device instance via cfgmgr32.

    Never raises; returns None on failure or non-Windows.
    """
    if sys.platform != "win32":
        return None
    try:
        cfg = ctypes.windll.cfgmgr32
    except OSError:
        return None

    # The CM_DRP_* ordinals are 1-based: cfgmgr32.h defines them as the
    # matching SPDRP_* value plus one. Passing the SPDRP numbers (0x15/0x1C)
    # here silently reads the wrong properties, the DWORD reads fail, and we
    # fall back to the localized LocationInformation string this function
    # exists to avoid.
    cm_locate_devnode_normal = 0x00000000
    cm_drp_busnumber = 0x00000016  # SPDRP_BUSNUMBER (0x15) + 1
    cm_drp_address = 0x0000001D    # SPDRP_ADDRESS (0x1C) + 1
    cr_success = 0x00000000

    try:
        cfg.CM_Locate_DevNodeW.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        cfg.CM_Locate_DevNodeW.restype = ctypes.c_uint32
        cfg.CM_Get_DevNode_Registry_PropertyW.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
        ]
        cfg.CM_Get_DevNode_Registry_PropertyW.restype = ctypes.c_uint32
    except AttributeError:
        return None

    devinst = ctypes.c_uint32()
    try:
        ret = cfg.CM_Locate_DevNodeW(
            ctypes.byref(devinst),
            instance_id,
            cm_locate_devnode_normal,
        )
    except OSError:
        return None
    if ret != cr_success:
        return None

    def _read_dword(prop: int) -> int | None:
        val = ctypes.c_uint32()
        size = ctypes.c_uint32(ctypes.sizeof(val))
        try:
            ret2 = cfg.CM_Get_DevNode_Registry_PropertyW(
                devinst.value,
                prop,
                None,
                ctypes.byref(val),
                ctypes.byref(size),
                0,
            )
        except OSError:
            return None
        if ret2 != cr_success:
            return None
        return val.value

    bus = _read_dword(cm_drp_busnumber)
    address = _read_dword(cm_drp_address)
    if bus is None or address is None:
        return None

    # VERIFY ON HARDWARE: for PCI, CM_DRP_ADDRESS is documented to pack the
    # device number in the high 16 bits and the function number in the low
    # 16 bits. Confirm on a real Arc box that the synthesized slot matches the
    # physical PCIe slot shown in Device Manager / GPU-Z.
    dev = (address >> 16) & 0xFFFF
    func = address & 0xFFFF
    return bus, dev, func


# ---------------------------------------------------------------------------
# Windows registry helpers (winreg is imported inside each function).
# ---------------------------------------------------------------------------

def _decode_qw_memory_size(raw, typ: int | None) -> int | None:
    """Interpret HardwareInformation.qwMemorySize as a byte count.

    The value is written as either REG_QWORD or REG_BINARY. Be defensive about
    both and about short buffers.
    """
    if raw is None or typ is None:
        return None
    if typ == _REG_QWORD:
        if isinstance(raw, int):
            return raw
        if isinstance(raw, bytes):
            return int.from_bytes(raw[:8].ljust(8, b"\x00"), "little")
    if typ == _REG_BINARY and isinstance(raw, bytes):
        return int.from_bytes(raw[:8].ljust(8, b"\x00"), "little")
    return None


def _winreg_display_class_entries() -> list[dict]:
    """Read Intel display-adapter class entries for driver and VRAM info.

    Returns a list of dicts with keys like MatchingDeviceId, HardwareID,
    DriverDesc, DriverVersion, ProviderName, and
    HardwareInformation.qwMemorySize. Never raises.
    """
    if sys.platform != "win32":
        return []
    import winreg

    class_guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
    key_path = rf"SYSTEM\CurrentControlSet\Control\Class\{class_guid}"
    entries: list[dict] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            idx = 0
            while True:
                try:
                    subname = winreg.EnumKey(root, idx)
                except OSError:
                    break
                idx += 1
                # Device instances are numeric subkeys; skip "Properties" etc.
                if not subname.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, subname) as sk:
                        entry: dict = {}
                        for val_name in (
                            "MatchingDeviceId",
                            "HardwareID",
                            "DriverDesc",
                            "DriverVersion",
                            "ProviderName",
                            "DeviceDesc",
                            "HardwareInformation.qwMemorySize",
                        ):
                            try:
                                val, typ = winreg.QueryValueEx(sk, val_name)
                                entry[val_name] = val
                                if val_name == "HardwareInformation.qwMemorySize":
                                    entry["HardwareInformation.qwMemorySize_type"] = typ
                            except OSError:
                                pass
                        entries.append(entry)
                except OSError:
                    continue
    except OSError:
        pass
    return entries


def _enum_pci_locations() -> list[dict]:
    """Read Intel display PCI instances and determine their BDF location.

    The primary source is cfgmgr32 numeric properties (locale-independent).
    If those are unavailable we fall back to parsing the registry
    LocationInformation string. Returns dicts with hardware_id, device_id,
    pci_slot, bus, dev, func, source. Never raises.
    """
    if sys.platform != "win32":
        return []
    import winreg

    loc_re = re.compile(
        r"PCI\s+bus\s+(\d+),\s*device\s+(\d+),\s*function\s+(\d+)", re.IGNORECASE
    )
    slots: list[dict] = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\PCI"
        ) as pci_root:
            i = 0
            while True:
                try:
                    hw_id = winreg.EnumKey(pci_root, i)
                except OSError:
                    break
                i += 1
                if not re.search(r"VEN_8086", hw_id, re.IGNORECASE):
                    continue
                dev_id = _parse_pnp_device_id(hw_id)
                if dev_id is None:
                    continue
                try:
                    with winreg.OpenKey(pci_root, hw_id) as hw_key:
                        j = 0
                        while True:
                            try:
                                inst_id = winreg.EnumKey(hw_key, j)
                            except OSError:
                                break
                            j += 1
                            # VERIFY ON HARDWARE: confirm this is the full
                            # device-instance ID format CM_Locate_DevNodeW expects.
                            full_id = f"{hw_id}\\{inst_id}"
                            bdf = _cfgmgr32_bdf(full_id)
                            source = "cfgmgr32"
                            if bdf is None:
                                try:
                                    with winreg.OpenKey(hw_key, inst_id) as inst_key:
                                        loc, _ = winreg.QueryValueEx(
                                            inst_key, "LocationInformation"
                                        )
                                except OSError:
                                    loc = None
                                m = loc_re.search(loc or "")
                                if m:
                                    bdf = tuple(map(int, m.groups()))
                                    source = "LocationInformation"
                                else:
                                    bdf = None
                                    source = "unknown"
                            if bdf is not None:
                                bus, dev, func = bdf
                                slot = f"0000:{bus:02x}:{dev:02x}.{func}"
                            else:
                                slot = f"unknown:{hw_id}"
                            slots.append(
                                {
                                    "hardware_id": hw_id,
                                    "device_id": dev_id,
                                    "pci_slot": slot,
                                    "bus": bdf[0] if bdf else None,
                                    "dev": bdf[1] if bdf else None,
                                    "func": bdf[2] if bdf else None,
                                    "source": source,
                                }
                            )
                except OSError:
                    continue
    except OSError:
        pass
    return slots


def _parse_pnp_device_id(pnp: str) -> int | None:
    """Extract the PCI device ID from a PNPDeviceID / MatchingDeviceId string.

    Input looks like ``PCI\\VEN_8086&DEV_E20B&SUBSYS_...&REV_..``.
    """
    m = re.search(r"DEV_([0-9A-Fa-f]{4})", pnp, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1), 16)


# ---------------------------------------------------------------------------
# Windows scan orchestration.
# ---------------------------------------------------------------------------

def _scan_windows() -> list[DetectedGPU]:
    """Enumerate Intel GPUs on Windows.

    PCI slots come from cfgmgr32 numeric properties (with a registry string
    fallback). VRAM comes from the display-class registry value
    ``HardwareInformation.qwMemorySize`` if present, then the device-ID table,
    then None with a note. iGPUs and the Microsoft Basic Display Adapter are
    filtered out. A missing driver is reported as driver=None with a note.
    """
    class_entries = _winreg_display_class_entries()
    pci_slots = _enum_pci_locations()

    # Driver version and VRAM by device ID. Multiple identical cards share an ID.
    driver_map: dict[int, list[str]] = {}
    vram_map: dict[int, list[tuple[int, str]]] = {}
    for e in class_entries:
        desc = e.get("DriverDesc", "")
        if "Microsoft Basic" in desc:
            continue
        pnp = e.get("MatchingDeviceId") or ""
        if not pnp and isinstance(e.get("HardwareID"), list):
            pnp = e["HardwareID"][0]
        dev_id = _parse_pnp_device_id(pnp)
        if dev_id is None:
            continue
        ver = e.get("DriverVersion")
        if ver:
            driver_map.setdefault(dev_id, []).append(ver)
        vram_bytes = _decode_qw_memory_size(
            e.get("HardwareInformation.qwMemorySize"),
            e.get("HardwareInformation.qwMemorySize_type"),
        )
        if vram_bytes is not None:
            vram_map.setdefault(dev_id, []).append((vram_bytes, "registry"))

    # Stable PCI slots by device ID, sorted so assignment is deterministic.
    slots_by_dev: dict[int, list[dict]] = {}
    for s in pci_slots:
        slots_by_dev.setdefault(s["device_id"], []).append(s)
    for lst in slots_by_dev.values():
        lst.sort(
            key=lambda x: (
                x.get("bus") if x.get("bus") is not None else -1,
                x.get("dev") if x.get("dev") is not None else -1,
                x.get("func") if x.get("func") is not None else -1,
                x["pci_slot"],
            )
        )

    found: list[DetectedGPU] = []
    for dev_id, slot_list in slots_by_dev.items():
        for slot_entry in slot_list:
            arch, name = arch_for_device_id(dev_id)

            # iGPU discrimination: never let an integrated GPU masquerade as Arc.
            if arch == Arch.LUNAR_LAKE:
                continue

            versions = driver_map.get(dev_id, [])
            driver = versions[0] if versions else None

            vram_mb: int | None = None
            vram_source: str | None = None
            vram_entries = vram_map.get(dev_id, [])
            if vram_entries:
                vram_bytes, vram_source = vram_entries.pop(0)
                vram_mb = vram_bytes // (1024 * 1024)

            if vram_mb is None:
                known = known_vram_mib(dev_id)
                if known is not None:
                    vram_mb = known
                    vram_source = "device-id table"

            is_dgpu = arch in (Arch.ALCHEMIST, Arch.BATTLEMAGE)
            if vram_mb is not None and vram_mb < 512 and not is_dgpu:
                continue

            gpu = DetectedGPU(
                pci_slot=slot_entry["pci_slot"],
                device_id=dev_id,
                arch=arch,
                name=name,
                driver=driver,
                vram_mb=vram_mb,
                drm_card=None,
                drm_render=None,
                sysfs_path="",
            )

            if vram_mb is not None and vram_source == "registry":
                gpu.notes.append(
                    f"VRAM {vram_mb // 1024} GB from HardwareInformation.qwMemorySize."
                )
            elif vram_mb is not None and vram_source == "device-id table":
                gpu.notes.append(
                    f"VRAM {vram_mb // 1024} GB from device-ID fallback table "
                    "(HardwareInformation.qwMemorySize was missing)."
                )
            else:
                gpu.notes.append(
                    "VRAM size not discoverable from registry or device-ID table."
                )

            if driver is None:
                gpu.notes.append(
                    "Intel GPU driver version not found in registry — the driver "
                    "may not be installed."
                )

            source = slot_entry.get("source")
            if source == "LocationInformation":
                gpu.notes.append(
                    "PCI location parsed from localized LocationInformation string; "
                    "slot may be less reliable than numeric cfgmgr32 data."
                )
            elif source == "unknown":
                gpu.notes.append(
                    "PCI location could not be determined; using a synthetic slot."
                )

            found.append(gpu)

    def _slot_sort_key(g: DetectedGPU):
        m = re.match(
            r"0000:([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9a-fA-F])", g.pci_slot
        )
        if m:
            return tuple(int(x, 16) for x in m.groups())
        return (0xFFFF, 0xFFFF, 0xFFFF)

    found.sort(key=_slot_sort_key)

    for i, g in enumerate(found):
        g._index = i  # type: ignore[attr-defined]

    if len(found) > 1:
        hint_note = (
            "Multiple Intel GPUs detected. sycl_index_hint follows PCI slot "
            "order, but the real Level Zero ordinal is the order shown by "
            "`sycl-ls`. Override `sycl_index` per GPU in the config if the "
            "wrong device is selected."
        )
        for g in found:
            g.notes.append(hint_note)

    return found


# ---------------------------------------------------------------------------
# SYCL index enrichment (Windows only, best-effort).
# ---------------------------------------------------------------------------

_SYCL_LS_LEVEL_ZERO_RE = re.compile(
    r"^\[\s*level_zero\s*:\s*gpu\s*(?::\s*(\d+))?\s*\]\s*(.+)$",
    re.IGNORECASE,
)


def _parse_sycl_ls_level_zero(text: str) -> list[tuple[int, str]]:
    """Parse level_zero GPU entries from ``sycl-ls`` output.

    # VERIFY ON HARDWARE: the exact ``sycl-ls`` bracket format varies by
    oneAPI version. Expected forms are ``[level_zero:gpu:N] <name>`` or
    ``[level_zero:gpu] <name>``.
    """
    entries: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = _SYCL_LS_LEVEL_ZERO_RE.match(line)
        if not m:
            continue
        idx_str, name = m.groups()
        idx = int(idx_str) if idx_str is not None else len(entries)
        entries.append((idx, name.strip()))
    return entries


def _enrich_with_sycl_ls(gpus: list[DetectedGPU]) -> None:
    """Best-effort: use ``sycl-ls`` to set the real Level Zero ordinal.

    Never raises. If ``sycl-ls`` is missing or unparseable the PCI-slot-based
    hint is left untouched.
    """
    if sys.platform != "win32":
        return
    try:
        out = subprocess.run(
            ["sycl-ls"], capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if out.returncode != 0:
        return

    entries = _parse_sycl_ls_level_zero(out.stdout)
    intel_entries = [(i, n) for (i, n) in entries if "Intel" in n]
    if not intel_entries:
        return

    for gpu in gpus:
        keys = [
            gpu.name,
            f"0x{gpu.device_id:04X}",
            "Arc",
        ]
        for k in keys:
            for pos, (idx, n) in enumerate(intel_entries):
                if k.lower() in n.lower():
                    gpu._index = idx  # type: ignore[attr-defined]
                    gpu.notes.append(f"sycl-ls level_zero:{idx} = {n}")
                    intel_entries.pop(pos)
                    break
            else:
                continue
            break


# ---------------------------------------------------------------------------
# Linux scan (unchanged behaviour).
# ---------------------------------------------------------------------------

def _read_hex(path: Path) -> int | None:
    try:
        return int(path.read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _driver_name(sysfs: Path) -> str | None:
    """Resolve the bound kernel driver from /sys/bus/pci/devices/<slot>/driver."""
    driver_link = sysfs / "driver"
    if not driver_link.exists():
        return None
    try:
        return os.path.basename(os.readlink(driver_link))
    except OSError:
        return None


def _drm_nodes(sysfs: Path) -> tuple[str | None, str | None]:
    """Find (cardN, renderDN) under /sys/bus/pci/devices/<slot>/drm/."""
    drm_dir = sysfs / "drm"
    if not drm_dir.exists():
        return None, None
    card = render = None
    for entry in drm_dir.iterdir():
        n = entry.name
        if n.startswith("card") and card is None:
            card = n
        elif n.startswith("renderD") and render is None:
            render = n
    return card, render


def _vram_mib(sysfs: Path) -> int | None:
    """Pull VRAM size from xe / i915 / amdgpu sysfs layouts."""
    candidates = [
        sysfs / "mem_info_vram_total",                       # amdgpu / some xe
        sysfs / "device" / "mem_info_vram_total",
        sysfs / "tile0" / "physical_vram_size_bytes",        # xe (Battlemage)
        sysfs / "device" / "tile0" / "physical_vram_size_bytes",
        sysfs / "lmem_total_bytes",                          # i915 dGPU
        sysfs / "device" / "lmem_total_bytes",
    ]
    for c in candidates:
        v = _read_int(c)
        if v:
            return v // (1024 * 1024)
    return None


def _scan_pci() -> list[DetectedGPU]:
    """Walk /sys/bus/pci/devices and collect Intel GPUs."""
    pci_root = Path("/sys/bus/pci/devices")
    if not pci_root.exists():
        return []
    found: list[DetectedGPU] = []
    for slot_dir in sorted(pci_root.iterdir()):
        vendor = _read_hex(slot_dir / "vendor")
        if vendor != INTEL_VENDOR_ID:
            continue
        klass = _read_hex(slot_dir / "class")
        if klass not in (PCI_CLASS_VGA, PCI_CLASS_DISPLAY):
            continue
        device_id = _read_hex(slot_dir / "device")
        if device_id is None:
            continue
        arch, name = arch_for_device_id(device_id)
        driver = _driver_name(slot_dir)
        card, render = _drm_nodes(slot_dir)
        vram = _vram_mib(slot_dir)
        vram_from_table = False
        if vram is None:
            # Driver didn't expose VRAM via sysfs (common on older i915 and some
            # early xe releases). Fall back to the known-size table keyed by
            # PCI device ID so context sizing still works without clinfo.
            known = known_vram_mib(device_id)
            if known is not None:
                vram = known
                vram_from_table = True
        gpu = DetectedGPU(
            pci_slot=slot_dir.name,
            device_id=device_id,
            arch=arch,
            name=name,
            driver=driver,
            vram_mb=vram,
            drm_card=card,
            drm_render=render,
            sysfs_path=str(slot_dir),
        )
        if vram_from_table:
            gpu.notes.append(
                f"VRAM {vram // 1024} GB from device-ID fallback table "
                f"(driver exposed no mem_info_vram_total)."
            )
        if driver is None:
            gpu.notes.append("No kernel driver bound — install `xe` or `i915` modules.")
        found.append(gpu)
    for i, g in enumerate(found):
        g._index = i  # type: ignore[attr-defined]
    return found


_CLINFO_DEVICE_RE = re.compile(r"^\s*Device Name\s+(.+?)\s*$", re.MULTILINE)
_CLINFO_GMEM_RE = re.compile(r"^\s*Global memory size\s+(\d+)", re.MULTILINE)


def _parse_clinfo_devices(text: str) -> list[tuple[str, int | None]]:
    """Split clinfo full output into per-device blocks and extract (name, gmem_bytes).

    clinfo prints each device's properties in a contiguous block. We split on
    `Device Name` boundaries and look for `Global memory size` within each block.
    """
    matches = list(_CLINFO_DEVICE_RE.finditer(text))
    out: list[tuple[str, int | None]] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        gmem_match = _CLINFO_GMEM_RE.search(block)
        gmem = int(gmem_match.group(1)) if gmem_match else None
        out.append((name, gmem))
    return out


def _enrich_with_clinfo(gpus: list[DetectedGPU]) -> None:
    """Best-effort: pull OpenCL device names + global memory size from clinfo.

    Runs the full `clinfo` (read-only, but heavier than `-l`). Skips silently
    if clinfo isn't installed.
    """
    try:
        out = subprocess.run(
            ["clinfo"], capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if out.returncode != 0:
        return
    devices = _parse_clinfo_devices(out.stdout)
    intel_arc_devices = [
        (n, g) for (n, g) in devices
        if "Intel" in n and ("Arc" in n or "Graphics" in n)
    ]
    if not intel_arc_devices:
        return
    for gpu in gpus:
        # Match on a meaningful substring of the name. We deliberately keep this
        # forgiving because Intel marketing names drift across driver versions.
        keys = [
            gpu.name,
            f"0x{gpu.device_id:04X}",
            "Arc",  # last-resort: assume the first remaining Arc entry is ours
        ]
        for k in keys:
            for i, (n, gmem) in enumerate(intel_arc_devices):
                if k.lower() in n.lower():
                    gpu.notes.append(f"OpenCL: {n}")
                    if gpu.vram_mb is None and gmem:
                        gpu.vram_mb = gmem // (1024 * 1024)
                    intel_arc_devices.pop(i)
                    break
            else:
                continue
            break


def detect_gpus(enrich: bool = True) -> list[DetectedGPU]:
    """Return every Intel GPU on this host. Sorted by PCI slot.

    On Windows this uses the display-adapter class registry for PCI IDs,
    driver version and VRAM, and cfgmgr32 for stable PCI slot location. On
    Linux it walks /sys/bus/pci/devices.

    Args:
        enrich: if True, also call `clinfo` (and on Windows `sycl-ls`) to enrich
            notes and improve the SYCL index hint. Set False in tests or when
            GPUs are in active use and you want zero subprocess noise.
    """
    if sys.platform == "win32":
        gpus = _scan_windows()
        if enrich and gpus:
            _enrich_with_sycl_ls(gpus)
            _enrich_with_clinfo(gpus)
        return gpus

    gpus = _scan_pci()
    if enrich and gpus:
        _enrich_with_clinfo(gpus)
    return gpus


def lspci_intel_gpus() -> str:
    """Return raw `lspci -nn` output filtered to Intel display devices.

    Useful for `arc-llama doctor` and for issue reports when arc-llama doesn't
    recognise a card.
    """
    try:
        out = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if out.returncode != 0:
        return ""
    return "\n".join(
        line for line in out.stdout.splitlines()
        if re.search(r"\[8086:[0-9A-Fa-f]+\]", line) and (
            "VGA" in line or "Display" in line or "3D" in line
        )
    )
