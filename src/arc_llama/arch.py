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


class Backend(str, Enum):
    """Which llama.cpp compute backend to use for a GPU."""
    SYCL = "sycl"     # Intel oneAPI/SYCL path; best prompt-eval on Arc
    VULKAN = "vulkan" # Cross-vendor Vulkan path; often better token-gen on Arc


@dataclass(frozen=True)
class HardwareSupport:
    """Evidence level for a detected GPU/backend combination."""

    status: str
    summary: str


def hardware_support(device_id: int, arch: Arch, backend: Backend) -> HardwareSupport:
    """Return conservative support status without equating detection to validation."""
    if device_id == 0xE211:
        return HardwareSupport("validated", f"Arc Pro B60 {backend.value} path tested")
    if arch in (Arch.ALCHEMIST, Arch.BATTLEMAGE, Arch.LUNAR_LAKE):
        return HardwareSupport("detected", f"{arch.value} detected; this exact SKU/backend needs confirmation")
    return HardwareSupport("unknown", f"unknown Intel GPU/backend combination ({backend.value})")


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
    safe_kv_q8_vulkan: bool = False
    """Whether q8_0 K/V cache is safe on the Vulkan backend for this arch.

    Vulkan requires --flash-attn for quantized V-cache. Profiles that set
    this True must also emit ``--flash-attn`` (default_recipe / launch policy
    do that automatically).
    """
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
# Known VRAM by PCI device ID (MiB). Fallback when the kernel driver doesn't
# expose mem_info_vram_total via sysfs (common on older i915 and some early
# xe releases). Source: Intel ARK product pages.
#
# Caveat: a few IDs map to multiple SKUs with different memory sizes
# (notably 0x56A0 = Arc A770, sold in both 8 GB and 16 GB). Where that's the
# case we list the larger common SKU and rely on sysfs/clinfo to override when
# available; sizing from this table is conservative-by-default territory.
# ---------------------------------------------------------------------------
VRAM_BY_DEVICE_ID: dict[int, int] = {
    # Alchemist (DG2)
    0x56A0: 16384,  # Arc A770 — also an 8 GB SKU shares this ID
    0x56A1: 8192,   # Arc A750
    0x56A2: 8192,   # Arc A580
    0x56A3: 6144,   # Arc A380 (variant)
    0x56A4: 4096,   # Arc A310
    0x56A5: 6144,   # Arc A380
    0x56A6: 6144,   # Arc A380
    0x56A8: 12288,  # Arc Pro A60
    0x56A9: 12288,  # Arc Pro A60M
    0x56B2: 12288,  # Arc Pro A60M
    0x56B3: 12288,  # Arc Pro A60
    0x56BA: 6144,   # Arc A380E
    0x56BB: 4096,   # Arc A310E
    0x5690: 16384,  # Arc A770M
    0x5691: 12288,  # Arc A730M
    0x5692: 8192,   # Arc A550M
    # Battlemage (BMG)
    0xE20B: 12288,  # Arc B580
    0xE20C: 10240,  # Arc B570
    0xE211: 24576,  # Arc Pro B60
    0xE212: 24576,  # Arc Pro B-series (assumed B60-class)
}

# ocloc -device strings for AOT (ahead-of-time) SYCL compilation, keyed by PCI
# device ID. Building llama-server with -DGGML_SYCL_DEVICE_ARCH=<string> bakes
# device code for your GPU generation and eliminates the ~20s SYCL JIT
# recompile paid on every cold start — doubly important on Battlemage, where
# the JIT cache is disabled (SYCL_CACHE_PERSISTENT=0) to dodge a SIGSEGV.
# Source: Intel oneAPI ocloc device naming.
AOT_ARCH_BY_DEVICE_ID: dict[int, str] = {
    # Alchemist ACM-G10 (A770 / A750 / A580)
    0x56A0: "acm-g10", 0x56A1: "acm-g10", 0x56A2: "acm-g10",
    0x56A8: "acm-g10", 0x56B3: "acm-g10",  # Arc Pro A60 (ACM-G10)
    # Alchemist ACM-G11 (A380 / A310)
    0x56A3: "acm-g11", 0x56A4: "acm-g11", 0x56A5: "acm-g11", 0x56A6: "acm-g11",
    0x56BA: "acm-g11", 0x56BB: "acm-g11",
    # Battlemage BMG-G21 (B570 / B580 / Pro B60)
    0xE20B: "bmg-g21", 0xE20C: "bmg-g21", 0xE211: "bmg-g21", 0xE212: "bmg-g21",
}


def known_vram_mib(device_id: int) -> int | None:
    """Return a known VRAM size (MiB) for a device ID, or None if unknown."""
    return VRAM_BY_DEVICE_ID.get(device_id)


def aot_arch_for(device_id: int) -> str | None:
    """Return the ocloc -device string for AOT SYCL compilation, or None.

    Falls back to a generation-level default when the exact device ID isn't
    listed (all known Battlemage dies are BMG-G21; Alchemist splits across
    ACM-G10 and ACM-G11, so the fallback uses the device-ID range heuristic).
    """
    exact = AOT_ARCH_BY_DEVICE_ID.get(device_id)
    if exact:
        return exact
    arch, _ = arch_for_device_id(device_id)
    if arch == Arch.BATTLEMAGE:
        return "bmg-g21"
    if arch == Arch.ALCHEMIST:
        # ACM-G10 = A770/A750/A580/Pro A60; ACM-G11 = A380/A310.
        if 0x56A0 <= device_id <= 0x56A2 or device_id in (0x56A8, 0x56B3, 0x56A9, 0x56B2):
            return "acm-g10"
        return "acm-g11"
    return None


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
    # With auto --flash-attn (recipes + policy), q8 KV is the competitive default.
    safe_kv_q8_vulkan=True,
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
        "Compare SYCL vs Vulkan and draft-mtp with `arc-llama benchmark` — "
        "relative wins depend on model class and llama-server build.",
    ],
    safe_kv_q8=True,
    safe_kv_q8_vulkan=True,
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
    safe_kv_q8_vulkan=True,
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
