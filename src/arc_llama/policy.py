"""Inference policy for Arc launch recipes.

Only encodes requirements verified on real hardware — not literature claims
or unbenchmarked folklore.

Verified on Battlemage B60 + SYCL llama-server + Qwen3.6-27B-MTP
(see ``bench_results/SUMMARY.md``):

* **Vulkan** + quantized V-cache needs ``--flash-attn`` for the quant path
  (llama.cpp requirement; already documented in arch profiles).
* **SYCL** + quantized V-cache does **not** require injecting flash-attn.
  Production configs with ``--cache-type-v q8_0`` and no ``--flash-attn``
  serve correctly. An earlier "hard abort" conclusion conflated Vulkan's
  constraint (and/or explicit ``-fa off``) onto SYCL — that claim is dropped.
* draft-mtp with ``--spec-draft-n-max`` 1–4: gen win vs no-MTP; n_max 5–6
  regresses gen well outside run noise. Policy warns (does not force) when
  n_max > 4.
* ``--fit on`` does **not** improve prompt-eval vs manual ``-c`` on this
  model; it only auto-sizes a larger context. Not auto-enabled here.
* Gen tok/s is noisy (~20% spread on identical config over 5 runs) and
  prompt-sensitive under MTP. Prefer ≥5 repeats before ranking configs.

Adoption rule: measure with ``scripts/empirical_throughput.py`` before new
defaults.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from arc_llama.arch import Arch, Backend
from arc_llama.recipes import KVCacheType, LaunchRecipe

log = logging.getLogger("arc_llama.policy")

# Quantized V-cache types that need flash-attn on the *Vulkan* backend.
_QUANT_KV: frozenset[KVCacheType] = frozenset(
    {
        KVCacheType.Q8_0,
        KVCacheType.Q5_1,
        KVCacheType.Q5_0,
        KVCacheType.Q4_1,
        KVCacheType.Q4_0,
    }
)


def _has_flag(flags: list[str], flag: str) -> bool:
    """True if flag or flag=value / flag value pair is present."""
    if flag in flags:
        return True
    if any(f.startswith(f"{flag}=") for f in flags):
        return True
    # Handle "-fa" / "--flash-attn" followed by on|off|auto
    for i, f in enumerate(flags):
        if f in (flag, "-fa") and i + 1 < len(flags):
            return True
    return False


def _flash_attn_disabled(flags: list[str]) -> bool:
    """True if the user explicitly turned flash-attn off."""
    for i, f in enumerate(flags):
        if f in ("--flash-attn", "-fa"):
            if i + 1 < len(flags) and flags[i + 1] in ("off", "0", "false"):
                return True
            if f.endswith("=off") or f.endswith("=0") or f.endswith("=false"):
                return True
        if f in ("--flash-attn=off", "-fa=off", "--flash-attn=0", "-fa=0"):
            return True
    return False


def _ensure_flash_attn_on(flags: list[str]) -> list[str]:
    """Return flags with flash-attn enabled (modern ``on`` form preferred)."""
    out = list(flags)
    # Strip any existing -fa / --flash-attn tokens so we don't duplicate.
    cleaned: list[str] = []
    skip_next = False
    for i, f in enumerate(out):
        if skip_next:
            skip_next = False
            continue
        if f in ("--flash-attn", "-fa"):
            # skip optional following on|off|auto
            if i + 1 < len(out) and out[i + 1] in ("on", "off", "auto", "0", "1", "true", "false"):
                skip_next = True
            continue
        if f.startswith("--flash-attn=") or f.startswith("-fa="):
            continue
        cleaned.append(f)
    cleaned.extend(["--flash-attn", "on"])
    return cleaned


def needs_flash_attn(backend: Backend, recipe: LaunchRecipe) -> bool:
    """True when Vulkan + quantized V/K cache needs --flash-attn injected.

    SYCL production configs run fine with q8 V-cache and no flash-attn flag.
    Only the Vulkan backend requires the flag for quantized V-cache.
    """
    if backend != Backend.VULKAN:
        return False
    return recipe.cache_type_v in _QUANT_KV or recipe.cache_type_k in _QUANT_KV


def apply_launch_policy(
    recipe: LaunchRecipe,
    *,
    arch: Arch,
    backend: Backend,
    model_path: str | Path,
    model_name: str = "",
) -> LaunchRecipe:
    """Return a recipe adjusted for verified launch requirements.

    Does not mutate *recipe*. Safe to call on every launch.

    Does **not** disable draft-MTP or auto-switch backends based on unmeasured
    arch heuristics — use ``scripts/empirical_throughput.py`` for those calls.
    """
    del arch, model_path  # reserved for future measured, opt-in policies
    label = model_name or "model"
    out = replace(recipe, extra_flags=list(recipe.extra_flags))

    # Measured: draft n_max > 4 regresses gen on B60 hybrid MTP (well outside
    # the ~20% same-config noise band).
    if out.spec_type == "draft-mtp" and out.spec_draft_n_max is not None:
        if out.spec_draft_n_max > 4:
            log.warning(
                "[%s] spec_draft_n_max=%d is above the measured sweet band "
                "(1–4 on B60/Qwen3.6-27B-MTP); gen regressed at 5–6 in "
                "bench_results/SUMMARY.md — consider lowering to 3",
                label,
                out.spec_draft_n_max,
            )

    # Vulkan-only: quantized V-cache needs --flash-attn.
    if needs_flash_attn(backend, out):
        if _flash_attn_disabled(out.extra_flags):
            log.warning(
                "[%s] Vulkan quantized KV-cache needs --flash-attn; "
                "overriding explicit flash-attn=off",
                label,
            )
        if not _has_flag(out.extra_flags, "--flash-attn") and "-fa" not in out.extra_flags:
            log.info(
                "[%s] injecting --flash-attn on (Vulkan quantized KV)",
                label,
            )
            out = replace(out, extra_flags=_ensure_flash_attn_on(out.extra_flags))
        elif _flash_attn_disabled(out.extra_flags):
            out = replace(out, extra_flags=_ensure_flash_attn_on(out.extra_flags))

    return out
