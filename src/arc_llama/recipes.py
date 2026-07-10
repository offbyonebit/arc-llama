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


FLASH_ATTN_VALUES = ("on", "off", "auto")


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
    """Logical batch size (-b). Must be >= ubatch_size when both are set."""
    flash_attn: str | None = None
    """Flash Attention: 'on' | 'off' | 'auto' | None (binary default).

    Old llama-server builds (pre ~b6300) expose -fa as a boolean that defaults
    to off; new builds take -fa {on,off,auto} and default to auto. `to_argv`
    translates per the probed binary style — see launcher.probe_server_caps.
    """
    no_mmap: bool = False
    """Disable mmap (--no-mmap): slower reload, but the whole model is read
    up-front — avoids page-cache thrash when VRAM spill keeps tensors host-side."""
    mlock: bool = False
    """--mlock: pin host-side weights in RAM so they can't be swapped out."""
    extra_flags: list[str] = field(default_factory=list)
    """Anything else the user wants appended to the command line verbatim."""

    def to_argv(self, fa_takes_value: bool = True) -> list[str]:
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
        if self.flash_attn in FLASH_ATTN_VALUES:
            if fa_takes_value:
                argv += ["-fa", self.flash_attn]
            elif self.flash_attn == "on":
                # Old boolean-style flag; 'off' is that style's default and
                # 'auto' is inexpressible, so both fall through to no flag.
                argv += ["-fa"]
        if self.no_mmap:
            argv += ["--no-mmap"]
        if self.mlock:
            argv += ["--mlock"]
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


PERF_UBATCH_MIN_VRAM_MB = 16384
"""Only default to a large ubatch on cards with real VRAM headroom — the
compute buffer grows roughly linearly with ubatch, and on 8–12 GB cards a
previously-fitting model could stop fitting."""

PERF_UBATCH = 1024
PERF_BATCH = 2048
PERF_COMPUTE_BUFFER_MB = 1536
"""Compute-buffer estimate used for ctx sizing when the perf ubatch applies
(vs. the conservative 768 MiB default at llama.cpp's stock ubatch of 512)."""


def default_recipe(
    arch: Arch,
    vram_mb: int,
    model_file_mb: int,
    kv_class: str = "default",
    prefer_q8_kv: bool = True,
) -> LaunchRecipe:
    """A safe starting recipe for a freshly added model on a given arch."""
    profile: ArchProfile = profile_for(arch)
    kv_type = KVCacheType.Q8_0 if (prefer_q8_kv and profile.safe_kv_q8) else KVCacheType.F16
    # Prompt processing on Arc is very sensitive to ubatch; default up from
    # llama.cpp's 512 when the card can absorb the bigger compute buffer.
    perf_batching = vram_mb >= PERF_UBATCH_MIN_VRAM_MB
    ctx = suggest_ctx(
        vram_mb=vram_mb,
        model_file_mb=model_file_mb,
        kv_type=kv_type,
        kv_class=kv_class,
        compute_buffer_mb=PERF_COMPUTE_BUFFER_MB if perf_batching else 768,
    )
    return LaunchRecipe(
        n_gpu_layers=999,
        ctx=ctx,
        parallel=1,
        cache_type_k=kv_type,
        cache_type_v=kv_type,
        flash_attn="auto",
        ubatch_size=PERF_UBATCH if perf_batching else None,
        batch_size=PERF_BATCH if perf_batching else None,
    )


def recipe_to_dict(recipe: LaunchRecipe) -> dict:
    """Serialise a recipe to the TOML-friendly dict stored in ModelConfig.recipe.

    Only always-meaningful fields are emitted unconditionally; optional fields
    are included only when set, so configs stay small and None never reaches
    the TOML writer.
    """
    d: dict = {
        "n_gpu_layers": recipe.n_gpu_layers,
        "ctx": recipe.ctx,
        "parallel": recipe.parallel,
        "cache_type_k": recipe.cache_type_k.value,
        "cache_type_v": recipe.cache_type_v.value,
    }
    if recipe.flash_attn is not None:
        d["flash_attn"] = recipe.flash_attn
    if recipe.ubatch_size is not None:
        d["ubatch_size"] = recipe.ubatch_size
    if recipe.batch_size is not None:
        d["batch_size"] = recipe.batch_size
    if recipe.spec_type is not None:
        d["spec_type"] = recipe.spec_type
    if recipe.no_mmap:
        d["no_mmap"] = True
    if recipe.mlock:
        d["mlock"] = True
    return d
