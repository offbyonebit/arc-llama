"""Intel GPU architecture profiles.

Each Arc generation has its own set of SYCL/OneAPI env-var quirks and known bugs.
This module is the single source of truth for that knowledge — when llama.cpp's
SYCL backend changes behaviour, update the profile here, not in launcher code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Arch(str, Enum):
    ALCHEMIST = "alchemist"   # Xe-HPG, A-series (A310/A380/A580/A750/A770)
    BATTLEMAGE = "battlemage" # Xe2, B-series (B570/B580, Pro B60)
    LUNAR_LAKE = "lunar_lake" # Xe2-LPG iGPU
    UNKNOWN = "unknown"


@dataclass
class ArchProfile:
    """SYCL recipe for a specific Intel GPU generation."""
    arch: Arch
    display_name: str
    sycl_env: dict[str, str]
    """Env vars to export before invoking llama-server."""
    sycl_env_remove: list[str] = field(default_factory=list)
    """Env vars to *unset* — if set in the user's shell they break this arch."""
    notes: list[str] = field(default_factory=list)
    """Human-readable notes shown in `arc-llama doctor`."""
    safe_kv_q8: bool = True
    """Whether q8_0 K/V cache produces correct generation on this arch."""
    prefer_uniform_quants: bool = True
    """If true, recommend Q4_K_M over Unsloth Dynamic XL/UD variants."""


# ---------------------------------------------------------------------------
# Known PCI device IDs. Vendor is always 0x8086.
# Sources: Intel ark, mesa drm_pciids, Linux i915/xe driver tables.
# Extend liberally — unknown IDs fall through to OpenCL device-name parsing.
# ---------------------------------------------------------------------------

# Alchemist (Xe-HPG, DG2)
ALCHEMIST_IDS: dict[int, str] = {
    0x4F80: "Arc A-series (DG2)",
    0x4F81: "Arc A-series (DG2)",
    0x4F82: "Arc A-series (DG2)",
    0x4F83: "Arc A-series (DG2)",
    0x4F84: "Arc A-series (DG2)",
    0x4F85: "Arc A-series (DG2)",
    0x4F86: "Arc A-series (DG2)",
    0x4F87: "Arc A-series (DG2)",
    0x4F88: "Arc A-series (DG2)",
    0x5690: "Arc A770M",
    0x5691: "Arc A730M",
    0x5692: "Arc A550M",
    0x5693: "Arc A370M",
    0x5694: "Arc A350M",
    0x5695: "Arc A200M",
    0x56A0: "Arc A770",
    0x56A1: "Arc A750",
    0x56A2: "Arc A580",
    0x56A3: "Arc A380 (variant)",
    0x56A4: "Arc A310",
    0x56A5: "Arc A380",
    0x56A6: "Arc A380",
    0x56A8: "Arc Pro A60",
    0x56A9: "Arc Pro A60M",
    0x56B0: "Arc Pro A30M",
    0x56B1: "Arc Pro A40 / A50",
    0x56B2: "Arc Pro A60M",
    0x56B3: "Arc Pro A60",
    0x56BA: "Arc A380E",
    0x56BB: "Arc A310E",
    0x56BC: "Arc A370E",
    0x56BD: "Arc A350E",
    0x56C0: "Data Center GPU Flex 170",
    0x56C1: "Data Center GPU Flex 140",
    0x56C2: "Data Center GPU Flex 170V",
}

# Battlemage (Xe2, BMG)
BATTLEMAGE_IDS: dict[int, str] = {
    0xE202: "Arc B-series (Battlemage)",
    0xE20B: "Arc B580",
    0xE20C: "Arc B570",
    0xE20D: "Arc B-series (variant)",
    0xE210: "Arc B-series (variant)",
    0xE211: "Arc Pro B60",      # confirmed on real hardware 2026-05-02
    0xE212: "Arc Pro B-series", # tentative; reserved
    0xE215: "Arc Pro B-series",
    0xE216: "Arc Pro B-series",
}

# Lunar Lake iGPU (Xe2-LPG)
LUNAR_LAKE_IDS: dict[int, str] = {
    0x6420: "Lunar Lake iGPU",
    0x64A0: "Lunar Lake iGPU",
    0x64B0: "Lunar Lake iGPU",
}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

ALCHEMIST_PROFILE = ArchProfile(
    arch=Arch.ALCHEMIST,
    display_name="Alchemist (Xe-HPG)",
    sycl_env={
        "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        "ZES_ENABLE_SYSMAN": "1",
    },
    sycl_env_remove=[
        # Doesn't appear required on Alchemist, but if a user has copied
        # a Battlemage workaround into their shell we don't want it lingering.
        "GGML_SYCL_DISABLE_OPT",
    ],
    notes=[
        "Use the `i915` driver on kernels <6.8 or `xe` on 6.8+.",
        "ReBAR strongly recommended — without it, perf drops sharply.",
        "Enable `intel-compute-runtime` and `intel-level-zero-gpu` packages.",
    ],
    safe_kv_q8=True,
    prefer_uniform_quants=True,
)

BATTLEMAGE_PROFILE = ArchProfile(
    arch=Arch.BATTLEMAGE,
    display_name="Battlemage (Xe2)",
    sycl_env={
        "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        "ZES_ENABLE_SYSMAN": "1",
        # =1 reproducibly SIGSEGVs in PersistentDeviceCodeCache::getItemFromDisc
        # on Battlemage with libsycl.so.9 from oneAPI 2026.0. Cost of =0 is a
        # ~20s JIT recompile per cold start.
        "SYCL_CACHE_PERSISTENT": "0",
    },
    sycl_env_remove=[
        # Killed MMVQ + reorder kernels on Battlemage — ~50% gen-speed hit on
        # dense models. Originally added defensively across launch scripts;
        # don't reintroduce it for plain llama.cpp.
        "GGML_SYCL_DISABLE_OPT",
        # IPEX-LLM Ollama bundle ships this; causes degenerate logits (gibberish
        # like `性价 SetLastError`) on every inference *after the first* on
        # Qwen2.5-class models. Plain llama.cpp doesn't need it; if it's set in
        # the inherited env, strip it.
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    ],
    notes=[
        "Requires kernel 6.14+ and Mesa 24.x+ for stable `xe` driver.",
        "ReBAR REQUIRED — without it llama.cpp will fall back to slow paths.",
        "First inference per cold start pays ~20s of SYCL JIT compile. An AOT "
        "build (-DGGML_SYCL_DEVICE_ARCH=bmg-g21) eliminates it entirely.",
        "q8_0 K/V cache works correctly but on some builds underutilises memory "
        "bandwidth on dense models. Run `arc-llama tune MODEL` to measure.",
    ],
    safe_kv_q8=True,
    prefer_uniform_quants=True,
)

LUNAR_LAKE_PROFILE = ArchProfile(
    arch=Arch.LUNAR_LAKE,
    display_name="Lunar Lake iGPU (Xe2-LPG)",
    sycl_env={
        "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        "ZES_ENABLE_SYSMAN": "1",
        "SYCL_CACHE_PERSISTENT": "0",
    },
    sycl_env_remove=[
        "GGML_SYCL_DISABLE_OPT",
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    ],
    notes=[
        "iGPU shares system RAM as VRAM — total budget is system memory minus "
        "what the OS and apps already hold.",
        "Prefer smaller models (≤7B Q4_K_M) for usable speeds.",
    ],
    safe_kv_q8=True,
    prefer_uniform_quants=True,
)

UNKNOWN_PROFILE = ArchProfile(
    arch=Arch.UNKNOWN,
    display_name="Unknown Intel GPU",
    sycl_env={
        "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        "ZES_ENABLE_SYSMAN": "1",
        "SYCL_CACHE_PERSISTENT": "0",
    },
    sycl_env_remove=[],
    notes=[
        "Device ID didn't match a known profile — applying conservative defaults.",
        "If this is a newer Intel GPU, please file an issue with `lspci -nn` output.",
    ],
    safe_kv_q8=True,
    prefer_uniform_quants=True,
)


PROFILES: dict[Arch, ArchProfile] = {
    Arch.ALCHEMIST: ALCHEMIST_PROFILE,
    Arch.BATTLEMAGE: BATTLEMAGE_PROFILE,
    Arch.LUNAR_LAKE: LUNAR_LAKE_PROFILE,
    Arch.UNKNOWN: UNKNOWN_PROFILE,
}


def arch_for_device_id(device_id: int) -> tuple[Arch, str]:
    """Resolve a PCI device ID (host byte order) to (arch, marketing-name)."""
    if device_id in BATTLEMAGE_IDS:
        return Arch.BATTLEMAGE, BATTLEMAGE_IDS[device_id]
    if device_id in ALCHEMIST_IDS:
        return Arch.ALCHEMIST, ALCHEMIST_IDS[device_id]
    if device_id in LUNAR_LAKE_IDS:
        return Arch.LUNAR_LAKE, LUNAR_LAKE_IDS[device_id]
    # Heuristic ranges for IDs we don't list explicitly.
    if 0xE200 <= device_id <= 0xE2FF:
        return Arch.BATTLEMAGE, f"Arc B-series (unrecognised ID 0x{device_id:04X})"
    if 0x5600 <= device_id <= 0x56FF or 0x4F80 <= device_id <= 0x4F8F:
        return Arch.ALCHEMIST, f"Arc A-series (unrecognised ID 0x{device_id:04X})"
    return Arch.UNKNOWN, f"Intel GPU 0x{device_id:04X}"


def profile_for(arch: Arch) -> ArchProfile:
    return PROFILES.get(arch, UNKNOWN_PROFILE)
