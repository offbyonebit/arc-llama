"""Safe discovery helpers for llama.cpp speculative decoding.

This deliberately ranks only registered models.  It does not guess that two
unrelated tokenizers are compatible: a bad draft is worse than no draft.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arc_llama.config import Config, ModelConfig
from arc_llama.recipes import estimate_kv_bytes

_FAMILY_RE = re.compile(r"(qwen[\w.-]*|llama[\w.-]*|gemma[\w.-]*|mistral[\w.-]*|phi[\w.-]*)", re.I)
_SAFETY_MB = 1024
_COMPUTE_MB = 1024


@dataclass(frozen=True)
class DraftCandidate:
    name: str
    path: str
    family: str
    estimated_mb: int
    fits: bool
    reason: str


def _family(model: ModelConfig) -> str:
    haystack = f"{model.name} {model.display_name} {Path(model.path).name}"
    match = _FAMILY_RE.search(haystack)
    return match.group(1).lower().split("-")[0] if match else ""


def _weight_mb(model: ModelConfig) -> int:
    try:
        return max(1, Path(model.path).stat().st_size // (1024 * 1024))
    except OSError:
        return 0


def discover_drafts(cfg: Config, target: ModelConfig) -> list[DraftCandidate]:
    """Return conservative same-family draft candidates, smallest first."""
    gpu = cfg.find_gpu(target.gpu_pci_slot)
    family = _family(target)
    recipe = target.launch_recipe()
    kv_mb = estimate_kv_bytes(recipe.ctx, recipe.cache_type_k, target.kv_class) // (1024 * 1024)
    target_mb = _weight_mb(target)
    result: list[DraftCandidate] = []
    for candidate in cfg.models:
        if candidate.name == target.name:
            continue
        candidate_family = _family(candidate)
        weight_mb = _weight_mb(candidate)
        if not family or candidate_family != family:
            continue
        if not weight_mb or weight_mb >= target_mb:
            continue
        required = target_mb + weight_mb + kv_mb + _COMPUTE_MB + _SAFETY_MB
        fits = bool(gpu and gpu.vram_mb and required <= gpu.vram_mb)
        result.append(DraftCandidate(
            name=candidate.name, path=candidate.path, family=family,
            estimated_mb=required, fits=fits,
            reason=("same family and fits VRAM budget" if fits else "same family but exceeds/unknown VRAM budget"),
        ))
    return sorted(result, key=lambda c: c.estimated_mb)
