"""Opt-in, staged MTP tuning through the existing router and edit API.

Trials temporarily persist recipes just like the main tuner. Every exit restores
the original axes unless a winner passes fresh confirmation measurements.
"""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from arc_llama.benchmark import _complete
from arc_llama.config import Config, ModelConfig, load_config
from arc_llama.gguf_meta import has_mtp_heads
from arc_llama.server_caps import ServerCaps
from arc_llama.speculation import effective_confidence
from arc_llama.tune import _restore_final_state

MTP_KEYS = (
    "spec_type",
    "spec_draft_name",
    "spec_draft_model",
    "spec_draft_ngl",
    "spec_draft_n_max",
    "spec_draft_p_min",
    "speculation_result",
)
DEFAULT_PROMPT = """Continue this Python module with a robust implementation of
merge_intervals(intervals). Return sorted non-overlapping intervals, merging
touching endpoints. Validate malformed intervals, preserve the input, and then
write tests covering empty input, nesting, duplicate intervals, negative values,
and unsorted input. Explain time complexity briefly in comments.

from collections.abc import Iterable

def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    \"\"\"Merge overlapping or touching intervals without modifying the input.\"\"\"
"""


@dataclass
class MTPMeasurement:
    generation_samples: list[float] = field(default_factory=list)
    wall_samples_ms: list[float] = field(default_factory=list)
    prompt_samples_ms: list[float] = field(default_factory=list)
    draft_tokens: int | None = None
    accepted_tokens: int | None = None
    output_fingerprints: list[str] = field(default_factory=list)

    @property
    def generation_tok_s(self) -> float:
        return float(statistics.median(self.generation_samples))

    @property
    def wall_ms(self) -> float:
        return float(statistics.median(self.wall_samples_ms))

    @property
    def acceptance(self) -> float | None:
        if self.draft_tokens and self.accepted_tokens is not None:
            return self.accepted_tokens / self.draft_tokens
        return None


@dataclass
class MTPReport:
    model: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    applied: bool = False
    selected: str = "original"
    reason: str = "No verified improvement; original recipe restored."
    sampling: dict[str, float | int] = field(
        default_factory=lambda: {"temperature": 0.0, "seed": 42}
    )


def snapshot(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(recipe.get(key)) for key in MTP_KEYS}


def mtp_state(original: dict[str, Any], length: int, confidence: float) -> dict[str, Any]:
    return dict(
        original,
        spec_type="draft-mtp",
        spec_draft_n_max=length,
        spec_draft_p_min=confidence,
        speculation_result=None,
    )


def validate_mtp(model: ModelConfig, caps: ServerCaps) -> None:
    if not (caps.probed and caps.supports_mtp and caps.supports_draft_confidence):
        raise ValueError("MTP tuning requires a probed build with draft-mtp and confidence support")
    recipe = model.recipe
    if recipe.get("spec_type") not in (None, "draft-mtp") or recipe.get("spec_draft_name"):
        raise ValueError("MTP tuning requires embedded MTP heads or an MTP sidecar recipe")
    # Unstructured overrides could silently defeat the candidate or off baseline.
    for flag in recipe.get("extra_flags", []):
        key = flag.split("=", 1)[0]
        if key.startswith("--spec-") and key != "--spec-draft-p-min":
            raise ValueError(f"Move {key} from extra_flags to recipe fields before MTP tuning")
        if key in (
            "-md",
            "--model-draft",
            "--draft-max",
            "--draft-n",
            "--draft",
            "-ngld",
            "--gpu-layers-draft",
            "--n-gpu-layers-draft",
            "--draft-min",
            "-hfd",
            "--hf-repo-draft",
            "--draft-p-split",
        ):
            raise ValueError(f"Remove legacy draft override {key} before MTP tuning")
    sidecar = recipe.get("spec_draft_model")
    if sidecar and not caps.supports_draft_model:
        raise ValueError("Installed build does not support a sidecar draft")
    if not has_mtp_heads(sidecar or model.path):
        raise ValueError("No MTP heads found in the configured model or sidecar")
    effective_confidence(recipe)


def draft_counts(timings: dict[str, Any]) -> tuple[int, int] | None:
    total, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
    if (
        isinstance(total, int)
        and not isinstance(total, bool)
        and total >= 0
        and isinstance(accepted, int)
        and not isinstance(accepted, bool)
        and 0 <= accepted <= total
    ):
        return total, accepted
    return None


async def _apply_edits(
    client: httpx.AsyncClient, model_name: str, state: dict[str, Any]
) -> str | None:
    """Require the server to acknowledge every axis, including cleared sidecars.

    An older server can return 200 while ignoring newly introduced edit fields.
    Never benchmark that as if the candidate had actually been installed.
    """
    try:
        response = await client.post(f"/admin/models/{model_name}/edit", json=state)
        if response.status_code != 200:
            return f"edit failed: {response.status_code} {response.text}"
        recipe = response.json().get("recipe")
        if not isinstance(recipe, dict) or any(recipe.get(k) != v for k, v in state.items()):
            return "Server did not apply the complete MTP recipe; restart Arc Llama after updating"
    except (httpx.HTTPError, ValueError) as e:
        return f"edit failed: {e}"
    return None


async def measure_mtp(
    client: httpx.AsyncClient,
    model: str,
    *,
    repeats: int,
    gen_tokens: int,
    prompt: str,
) -> MTPMeasurement:
    # Absorb JIT/model load outside the measured requests; fail if warmup fails.
    await _complete(
        client, model, prompt, 8, is_chat=False, cache_prompt=False, temperature=0.0, seed=42
    )
    result = MTPMeasurement()
    for _ in range(repeats):
        wall, obj = await _complete(
            client,
            model,
            prompt,
            gen_tokens,
            ignore_eos=True,
            is_chat=False,
            cache_prompt=False,
            temperature=0.0,
            seed=42,
        )
        timings = obj.get("timings") or {}
        choices = obj.get("choices") or []
        if choices and isinstance(choices[0].get("text"), str):
            result.output_fingerprints.append(
                hashlib.sha256(choices[0]["text"].encode()).hexdigest()
            )
        count = timings.get("predicted_n", (obj.get("usage") or {}).get("completion_tokens"))
        if not isinstance(count, int) or isinstance(count, bool) or count != gen_tokens:
            raise RuntimeError("Benchmark did not generate the requested token count")
        # Decode timings must exist: wall-based throughput cannot be labelled decode speed.
        rate = timings.get("predicted_per_second")
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate <= 0
            or not math.isfinite(wall)
            or wall <= 0
        ):
            raise RuntimeError("Benchmark response omitted valid generation timings")
        result.generation_samples.append(float(rate))
        result.wall_samples_ms.append(wall * 1000)
        prompt_ms = timings.get("prompt_ms")
        if isinstance(prompt_ms, (int, float)) and math.isfinite(prompt_ms) and prompt_ms >= 0:
            result.prompt_samples_ms.append(float(prompt_ms))
        counts = draft_counts(timings)
        if counts is not None:
            result.draft_tokens = (result.draft_tokens or 0) + counts[0]
            result.accepted_tokens = (result.accepted_tokens or 0) + counts[1]
    return result


def improves(candidate: MTPMeasurement, baseline: MTPMeasurement, threshold: float) -> bool:
    # Different continuations can have very different MTP acceptance. Missing
    # output evidence must never turn into a claimed comparable speedup.
    if (
        len(candidate.output_fingerprints) != len(candidate.generation_samples)
        or len(baseline.output_fingerprints) != len(baseline.generation_samples)
        or not candidate.output_fingerprints
        or candidate.output_fingerprints != baseline.output_fingerprints
    ):
        return False
    return (
        candidate.generation_tok_s >= baseline.generation_tok_s * (1 + threshold)
        and baseline.wall_ms / candidate.wall_ms >= 1 + threshold
    )


async def tune_mtp(
    cfg: Config,
    model: ModelConfig,
    caps: ServerCaps,
    *,
    server_url: str,
    config_path: Path,
    repeats: int = 5,
    gen_tokens: int = 128,
    min_speedup: float = 0.05,
    prompt: str = DEFAULT_PROMPT,
    progress: Callable[[str], None] | None = None,
) -> MTPReport:
    validate_mtp(model, caps)
    if repeats < 5 or gen_tokens < 16 or not prompt.strip():
        raise ValueError("Use at least 5 repeats, 16 generated tokens, and a nonempty prompt")
    if not math.isfinite(min_speedup) or min_speedup < 0.05:
        raise ValueError("MTP tuning requires a minimum speedup of at least 5%")
    original = snapshot(model.recipe)
    confidence = effective_confidence(model.recipe)
    off = dict.fromkeys(MTP_KEYS)
    report = MTPReport(model=model.name)
    headers = (
        {"Authorization": f"Bearer {cfg.server.admin_token}"} if cfg.server.admin_token else {}
    )

    async def measure(client: httpx.AsyncClient) -> MTPMeasurement:
        return await measure_mtp(
            client, model.name, repeats=repeats, gen_tokens=gen_tokens, prompt=prompt
        )

    async with httpx.AsyncClient(base_url=server_url, timeout=600, headers=headers) as client:
        # Fail before any edits when the service is still starting or unreachable.
        readiness = await client.get("/health", timeout=10)
        readiness.raise_for_status()
        changed = False
        committed = False

        async def trial(label: str, state: dict[str, Any]) -> MTPMeasurement:
            nonlocal changed
            if progress:
                progress(f"Measuring {label} ({repeats} repeats)")
            changed = True  # POST may persist before a timeout or rebuild failure.
            error = await _apply_edits(client, model.name, state)
            if error:
                raise RuntimeError(error)
            result = await measure(client)
            row = dict(
                label=label,
                recipe=state,
                **asdict(result),
                generation_tok_s=result.generation_tok_s,
                wall_ms=result.wall_ms,
                acceptance=result.acceptance,
            )
            report.trials.append(row)
            if progress:
                acceptance = (
                    f"{result.acceptance:.1%}" if result.acceptance is not None else "unknown"
                )
                progress(
                    f"{label}: {result.generation_tok_s:.2f} tok/s, "
                    f"{result.wall_ms:.0f} ms/request, draft acceptance {acceptance}"
                )
            return result

        try:
            baseline = await trial("original", original)
            target_only = await trial("target-only", off)
            best = target_only
            best_state = off
            best_label = "target-only"
            # Keep the fastest length even when it has not yet cleared the threshold;
            # confidence tuning may rescue a promising but inefficient draft length.
            length_results: list[tuple[MTPMeasurement, int]] = []
            seen: set[tuple[int, float]] = set()

            async def candidate(length: int, cutoff: float) -> None:
                nonlocal best, best_state, best_label
                if (length, cutoff) in seen:
                    return
                seen.add((length, cutoff))
                state = mtp_state(original, length, cutoff)
                label = f"mtp length={length} confidence={cutoff:g}"
                try:
                    result = await trial(label, state)
                except (RuntimeError, httpx.HTTPError) as e:
                    report.trials.append({"label": label, "error": str(e)})
                    if progress:
                        progress(f"{label}: failed ({e})")
                    return
                if cutoff == confidence:
                    length_results.append((result, length))
                if (
                    improves(result, target_only, min_speedup)
                    and improves(result, baseline, min_speedup)
                    and (best_state == off or result.wall_ms < best.wall_ms)
                ):
                    best, best_state, best_label = result, state, label

            for length in (1, 2, 3, 4):
                await candidate(length, confidence)
            if length_results:
                _, best_length = min(length_results, key=lambda pair: pair[0].wall_ms)
                for cutoff in (0.0, 0.5, 0.75, 0.9):
                    await candidate(best_length, cutoff)
            if not improves(best, baseline, min_speedup):
                return report
            # Remeasure baselines and winner to reject drift and one-off improvements.
            original_check = await trial("confirm original", original)
            off_check = await trial("confirm target-only", off)
            winner_check = await trial(f"confirm {best_label}", best_state)
            if not improves(winner_check, original_check, min_speedup) or (
                best_state != off and not improves(winner_check, off_check, min_speedup)
            ):
                report.reason = "Winner failed fresh confirmation; original recipe restored."
                return report
            final = dict(best_state)
            gain = winner_check.generation_tok_s / original_check.generation_tok_s - 1
            wall_gain = original_check.wall_ms / winner_check.wall_ms - 1
            final["speculation_result"] = (
                f"{best_label}; confirmed {gain:.1%} decode and {wall_gain:.1%} request "
                f"throughput improvement ({repeats} repeats; selected benchmark prompt)"
            )
            error = await _apply_edits(client, model.name, final)
            if error:
                raise RuntimeError(f"Could not apply confirmed winner: {error}")
            committed = True
            report.applied = True
            report.selected = best_label
            report.reason = str(final["speculation_result"])
            return report
        finally:
            if changed and not committed:
                error = await _restore_final_state(client, model.name, original, cfg=None)
                if error:
                    # Repair the selected config, never the default or a stale full config.
                    fresh = load_config(config_path)
                    target = fresh.find_model(model.name)
                    if target is None:
                        raise RuntimeError(
                            f"Live rollback failed: {error}; model missing from config"
                        )
                    if target is not None:
                        for key, value in original.items():
                            if value is None:
                                target.recipe.pop(key, None)
                            else:
                                target.recipe[key] = value
                        fresh.save(config_path)
                    raise RuntimeError(
                        f"Live rollback failed: {error}; config repaired; restart Arc Llama"
                    )
