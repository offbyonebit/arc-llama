"""Default llama.cpp launch recipes.

A *recipe* is the set of `llama-server` flags we'll feed for a given
(GPU arch, model size, model file size, target context length). Defaults are
chosen to be safe rather than maximal — we'd rather start small and let the user
crank context up than have a first-run experience that OOMs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from arc_llama.arch import Arch, ArchProfile, profile_for


class KVCacheType(str, Enum):
    F16 = "f16"
    F32 = "f32"
    Q8_0 = "q8_0"
    Q5_1 = "q5_1"
    Q5_0 = "q5_0"
    Q4_1 = "q4_1"
    Q4_0 = "q4_0"


# Approx KV bytes per token at f16 for a few well-known architectures.
# Numbers are tuned against `memory_breakdown_print` measurements on a real
# Battlemage B60 stack. They're upper bounds for sizing; actual usage with
# sliding-window attention (Gemma) is several × smaller again.
KV_PER_TOKEN_F16_BYTES: dict[str, int] = {
    "default": 70 * 1024,   # 70 KiB/token f16 — covers ~30B dense models
    "moe_a3b": 20 * 1024,   # ~20 KiB — Qwen3 30B/35B-A3B-class MoE
    "qwen3_27b_dense": 67 * 1024,
    "gemma_swa": 16 * 1024, # interleaved sliding-window attention compresses heavily
}


@dataclass
class LaunchRecipe:
    """A complete llama-server invocation, minus the model path and port."""
    n_gpu_layers: int = 999
    ctx: int = 8192
    parallel: int = 1
    cache_type_k: KVCacheType = KVCacheType.F16
    cache_type_v: KVCacheType = KVCacheType.F16
    threads: int | None = None
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    spec_type: str | None = None
    """Speculative decoding type, e.g. 'draft-mtp'."""
    ubatch_size: int | None = None
    """Ubatch size (-ub). Auto-set to 8 for MTP models to avoid SSM compute-buffer OOM."""
    batch_size: int | None = None
    """Logical batch size (-b). Raised together with ubatch for prefill throughput."""
    flash_attn: str | None = None
    """Flash attention (-fa): 'on', 'off', or 'auto'. None → omit the flag and
    let llama.cpp decide. Must be 'on' + f16 KV cache to hit the oneDNN XMX
    prefill path on Xe2."""
    cache_reuse: int | None = None
    """--cache-reuse N: reuse cached prompt prefixes with a KV shift when at
    least N tokens match. Slashes time-to-first-token in multi-turn chat.
    Incompatible with sliding-window-attention models (gemma_swa)."""
    no_mmap: bool = False
    """--no-mmap: read the model up front instead of paging it in lazily.
    Avoids first-inference page-fault stalls when weights live on slow disks."""
    extra_flags: list[str] = field(default_factory=list)
    """Anything else the user wants appended to the command line verbatim."""

    def to_argv(self) -> list[str]:
        argv = [
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx),
            "--parallel", str(self.parallel),
            "--cache-type-k", self.cache_type_k.value,
            "--cache-type-v", self.cache_type_v.value,
        ]
        if self.threads is not None:
            argv += ["-t", str(self.threads)]
        if self.temp is not None:
            argv += ["--temp", str(self.temp)]
        if self.top_p is not None:
            argv += ["--top-p", str(self.top_p)]
        if self.top_k is not None:
            argv += ["--top-k", str(self.top_k)]
        if self.spec_type:
            argv += ["--spec-type", self.spec_type]
        if self.ubatch_size is not None:
            argv += ["-ub", str(self.ubatch_size)]
        if self.batch_size is not None:
            argv += ["-b", str(self.batch_size)]
        if self.flash_attn is not None:
            argv += ["-fa", self.flash_attn]
        if self.cache_reuse is not None:
            argv += ["--cache-reuse", str(self.cache_reuse)]
        if self.no_mmap:
            argv += ["--no-mmap"]
        argv += list(self.extra_flags)
        return argv


def estimate_kv_bytes(ctx: int, kv_type: KVCacheType, kv_class: str = "default") -> int:
    """Rough estimate of KV-cache bytes at runtime."""
    f16_per_token = KV_PER_TOKEN_F16_BYTES.get(kv_class, KV_PER_TOKEN_F16_BYTES["default"])
    scale = {
        KVCacheType.F32: 2.0,
        KVCacheType.F16: 1.0,
        KVCacheType.Q8_0: 0.5,
        KVCacheType.Q5_1: 0.375,
        KVCacheType.Q5_0: 0.375,
        KVCacheType.Q4_1: 0.3125,
        KVCacheType.Q4_0: 0.3125,
    }.get(kv_type, 1.0)
    return int(ctx * f16_per_token * scale)


DEFAULT_CTX_CAP = 131072
"""Hard ceiling on auto-suggested context length. VRAM math will sometimes
say a 500k+ ctx fits, but real models top out around 128k–256k and our
KV-per-token estimates are inherently approximate. Cap the auto-suggestion
at 131072 — users who want more can override the recipe per-model."""


def suggest_ctx(
    vram_mb: int,
    model_file_mb: int,
    kv_type: KVCacheType,
    kv_class: str = "default",
    compute_buffer_mb: int = 768,
    safety_margin_mb: int = 256,
    ctx_cap: int = DEFAULT_CTX_CAP,
) -> int:
    """Pick the largest power-of-2-ish context that fits comfortably in VRAM.

    Rounds *down* to the nearest multiple of 4096 and clamps to `ctx_cap`.
    """
    free_for_kv = vram_mb - model_file_mb - compute_buffer_mb - safety_margin_mb
    if free_for_kv <= 0:
        return 4096  # last-resort minimum; user should pick a smaller quant
    f16_per_token = KV_PER_TOKEN_F16_BYTES.get(kv_class, KV_PER_TOKEN_F16_BYTES["default"])
    scale = {
        KVCacheType.F32: 2.0,
        KVCacheType.F16: 1.0,
        KVCacheType.Q8_0: 0.5,
        KVCacheType.Q5_1: 0.375,
        KVCacheType.Q5_0: 0.375,
        KVCacheType.Q4_1: 0.3125,
        KVCacheType.Q4_0: 0.3125,
    }.get(kv_type, 1.0)
    bytes_per_token = int(f16_per_token * scale)
    if bytes_per_token <= 0:
        return 4096
    max_tokens = (free_for_kv * 1024 * 1024) // bytes_per_token
    rounded = (max_tokens // 4096) * 4096
    return max(4096, min(rounded, ctx_cap))


XMX_FA_MIN_F16_CTX = 16384
"""Smallest f16-KV context worth trading Q8_0 KV headroom for. Below this the
KV savings buy more useful context than the XMX prefill path buys speed."""

XMX_FA_UBATCH = 1024
XMX_FA_BATCH = 2048
"""Prefill batch sizing for the XMX path. llama.cpp's default -ub 512 leaves
the systolic arrays underfed on Arc dGPUs; 1024/2048 measured fastest on
Battlemage without blowing up compute buffers."""

XMX_FA_COMPUTE_BUFFER_MB = 1536
"""Compute-buffer estimate when ubatch is raised to 1024 — roughly double the
768 MB default-ubatch figure. Deliberately pessimistic so ctx sizing stays safe."""

DEFAULT_CACHE_REUSE = 256
"""--cache-reuse threshold applied to non-SWA models: multi-turn chats re-use
the common prefix KV instead of re-prefilling the whole conversation."""


def default_recipe(
    arch: Arch,
    vram_mb: int,
    model_file_mb: int,
    kv_class: str = "default",
    prefer_q8_kv: bool = True,
) -> LaunchRecipe:
    """A safe starting recipe for a freshly added model on a given arch.

    On Xe2 (Battlemage / Lunar Lake) the oneDNN XMX flash-attention path is
    preferred when VRAM allows: it needs an f16 KV cache (quantized KV falls
    back to shader kernels), so we take f16 + `-fa on` whenever that still
    leaves ≥ XMX_FA_MIN_F16_CTX of context, and only drop to q8_0 KV on
    VRAM-tight setups where context matters more than prefill speed.
    """
    profile: ArchProfile = profile_for(arch)
    cache_reuse = None if kv_class == "gemma_swa" else DEFAULT_CACHE_REUSE

    if profile.supports_xmx_fa:
        ctx_f16 = suggest_ctx(
            vram_mb=vram_mb,
            model_file_mb=model_file_mb,
            kv_type=KVCacheType.F16,
            kv_class=kv_class,
            compute_buffer_mb=XMX_FA_COMPUTE_BUFFER_MB,
        )
        if ctx_f16 >= XMX_FA_MIN_F16_CTX:
            return LaunchRecipe(
                n_gpu_layers=999,
                ctx=ctx_f16,
                parallel=1,
                cache_type_k=KVCacheType.F16,
                cache_type_v=KVCacheType.F16,
                ubatch_size=XMX_FA_UBATCH,
                batch_size=XMX_FA_BATCH,
                flash_attn="on",
                cache_reuse=cache_reuse,
            )

    kv_type = KVCacheType.Q8_0 if (prefer_q8_kv and profile.safe_kv_q8) else KVCacheType.F16
    ctx = suggest_ctx(
        vram_mb=vram_mb,
        model_file_mb=model_file_mb,
        kv_type=kv_type,
        kv_class=kv_class,
    )
    return LaunchRecipe(
        n_gpu_layers=999,
        ctx=ctx,
        parallel=1,
        cache_type_k=kv_type,
        cache_type_v=kv_type,
        cache_reuse=cache_reuse,
    )
