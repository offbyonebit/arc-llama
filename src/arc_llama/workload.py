"""Workload profile: what the declared usage answers change for tuning.

The tuner measured at 1024 prompt / 128 gen and concluded f16 KV beats q8_0 —
true at that measurement point and wrong for a model actually used at 131k
context, where f16 KV does not fit in 24 GB at all. The answers gathered by
`arc-llama init` (persisted in the [workload] config section) exist so the
tuner optimises the workload rather than the benchmark:

  * context_length sets the tuning context target and prunes KV candidates
    that cannot hold it in VRAM,
  * style maps agentic -> prompt target, conversational -> generation,
  * priority weights the score between prompt-eval and generation.

Every field may be empty ("not sure"); an empty profile reproduces the
pre-profile behaviour exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arc_llama.recipes import KVCacheType

if TYPE_CHECKING:
    from arc_llama.config import Config, GPUConfig, ModelConfig, WorkloadConfig

#: Tuning context target per declared conversation length. These are the
#: depths at which the KV-fit question must be answered and (for long
#: contexts) at which the depth-sensitive axes are measured.
CONTEXT_TARGET_TOKENS: dict[str, int] = {
    "short": 8192,
    "long": 32768,
    "very_long": 131072,
}

CONTEXT_LENGTHS = tuple(CONTEXT_TARGET_TOKENS)
STYLES = ("agentic", "conversational")
PRIORITIES = ("first_token", "throughput")

#: Contexts at or above this change measurement depth: KV and ubatch rankings
#: at 1k prompt tokens do not necessarily hold at 32k, so those stages are
#: measured at the declared depth instead of the shallow default.
DEEP_MEASUREMENT_MIN_TOKENS = 32768

#: Tokens reserved below the model's configured n_ctx when building a deep
#: prompt-eval benchmark. The request adds max_tokens=1 on top of the
#: prompt, and llama-server needs a little headroom for its own slot
#: bookkeeping; without this margin the total can exceed n_ctx and the
#: server rejects the request with HTTP 400.
DEEP_PROMPT_CTX_RESERVE_TOKENS = 128

#: Axes whose ordering changes with context depth. Flash attention is not
#: among them — it stays at the shallow measurement depth.
_DEPTH_SENSITIVE_EDITS = frozenset({"cache_type_k", "cache_type_v", "ubatch_size", "batch_size"})


def target_ctx(workload: WorkloadConfig) -> int | None:
    """The context the tuner must plan for, or None when undeclared."""
    return CONTEXT_TARGET_TOKENS.get(workload.context_length)


def tune_target(cfg: Config) -> str:
    """Effective sweep target: agentic -> prompt, conversational -> generation.

    Falls back to the explicit [tune] target ("balanced" by default) when the
    style question is unanswered.
    """
    if cfg.workload.style == "agentic":
        return "prompt"
    if cfg.workload.style == "conversational":
        return "generation"
    return cfg.tune.target


def score_priority(cfg: Config) -> str | None:
    """Score weighting from the first-token-vs-throughput answer, if given."""
    p = cfg.workload.priority
    return p if p in PRIORITIES else None


def fingerprint_key(workload: WorkloadConfig) -> str:
    """Canonical string of the profile, mixed into the tune fingerprint.

    Changing any answer must invalidate tuned recipes and schedule a retune,
    because the profile changes both the search space and the measurement
    depth the recipe was chosen under.
    """
    return f"ctx={workload.context_length};style={workload.style};priority={workload.priority}"


def kv_fits_at_ctx(
    model: ModelConfig,
    gpu: GPUConfig | None,
    kv_type: str,
    ctx: int,
) -> bool:
    """True when the model with this KV type is estimated to fit at ``ctx``.

    Reuses the router's VRAM estimation (GGUF weight metadata + KV-per-token
    + compute-buffer and safety margins), which accounts for any expert
    offload in the model's recipe. Without a known VRAM budget — or when the
    footprint cannot be estimated at all (expert offload bytes unknown) —
    the answer is optimistic: pruning must never empty the search space on
    cards we cannot size.
    """
    from arc_llama.router import _estimate_model_vram_mb

    if gpu is None or not gpu.vram_mb:
        return True
    needed = _estimate_model_vram_mb(model, ctx=ctx, kv_type=KVCacheType(kv_type))
    if needed is None:
        return True
    return needed <= gpu.vram_mb


def deep_prompt_tokens(cfg: Config, recipe: dict) -> int | None:
    """Prompt depth for depth-sensitive stages, or None for shallow.

    Only declared long contexts trigger deep measurement. When the model's
    configured ctx is known, the depth is capped below it by
    ``DEEP_PROMPT_CTX_RESERVE_TOKENS`` so the prompt plus the benchmark's
    generation budget and server bookkeeping fit inside n_ctx. If the
    capped depth would fall below ``DEEP_MEASUREMENT_MIN_TOKENS``, the
    stage is measured shallow instead.
    """
    target = target_ctx(cfg.workload)
    if target is None or target < DEEP_MEASUREMENT_MIN_TOKENS:
        return None
    recipe_ctx = int((recipe or {}).get("ctx", 0))
    if recipe_ctx > 0:
        depth = min(target, recipe_ctx - DEEP_PROMPT_CTX_RESERVE_TOKENS)
        if depth < DEEP_MEASUREMENT_MIN_TOKENS:
            return None
        return depth
    return target


def stage_is_depth_sensitive(edits_keys: set[str]) -> bool:
    """True when a stage's edits touch an axis whose ordering changes with depth."""
    return not _DEPTH_SENSITIVE_EDITS.isdisjoint(edits_keys)
