"""Autotuner: measure, pick, persist.

`arc-llama tune MODEL` closes the loop between the benchmark harness and the
per-model recipe. Static defaults can't know whether *your* card/model/build
combination prefers f16 or q8_0 KV, a 512 or 2048 ubatch, or flash attention
on or off — the SYCL backend's answer genuinely differs per SKU and per
llama.cpp revision. A ~10-minute staged sweep answers it empirically and
writes the winner into the model's recipe.

Search is greedy and staged rather than exhaustive (the full grid would be
18+ cold starts):

  stage 1: KV cache type   f16 vs q8_0
  stage 2: ubatch size     one step down / up from current
  stage 3: flash attention auto / on / off

Each stage keeps its winner and carries it into the next. A candidate that
fails to launch (OOM from a bigger compute buffer, V-quant without FA on a
build that requires it) simply records an error and loses the stage — the
tuner never leaves the model in a broken config: the winning (or original)
recipe is re-applied at the end.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import httpx

from arc_llama.benchmark import DEFAULT_GEN_TOKENS, BenchmarkResult, benchmark_model
from arc_llama.config import Config, load_config
from arc_llama.recipes import FLASH_ATTN_VALUES

log = logging.getLogger("arc_llama.tune")

TUNE_PROMPT_TOKENS = 1024
"""Longer than the benchmark default: prompt-eval differences between ubatch
settings only show up once the prompt spans several ubatches."""

TARGETS = ("balanced", "generation", "prompt")

_UBATCH_LADDER = [256, 512, 1024, 2048]


@dataclass
class TuneStep:
    """One measured candidate configuration."""
    label: str
    edits: dict[str, Any]
    result: BenchmarkResult | None = None
    score: float | None = None
    chosen: bool = False
    skipped_reason: str | None = None


@dataclass
class TuneReport:
    model: str
    target: str
    steps: list[TuneStep] = field(default_factory=list)
    best_edits: dict[str, Any] = field(default_factory=dict)
    baseline: BenchmarkResult | None = None
    best: BenchmarkResult | None = None
    applied: bool = False
    error: str | None = None

    @property
    def improvement_pct(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {"prompt_eval": None, "generation": None}
        if self.baseline is None or self.best is None:
            return out
        for key, attr in (("prompt_eval", "prompt_eval_tok_s"), ("generation", "generation_tok_s")):
            a = getattr(self.baseline, attr)
            b = getattr(self.best, attr)
            if a and b:
                out[key] = round((b / a - 1.0) * 100, 1)
        return out


def score_result(result: BenchmarkResult, target: str = "balanced") -> float | None:
    """Higher is better. Errors and empty measurements score None (lose)."""
    if result.error:
        return None
    pp = result.prompt_eval_tok_s or 0.0
    gen = result.generation_tok_s or 0.0
    if target == "generation":
        return gen if gen > 0 else None
    if target == "prompt":
        return pp if pp > 0 else None
    if pp <= 0 or gen <= 0:
        return None
    # Balanced: geometric mean, so a 2x regression on one axis can't be bought
    # back by a 2x win on the other plus rounding.
    return math.sqrt(pp * gen)


def _ubatch_candidates(current: int | None, vram_mb: int | None) -> list[int]:
    """Current value plus one ladder step down and up, VRAM-permitting."""
    cur = current or 512  # llama.cpp default
    ladder = [u for u in _UBATCH_LADDER if u != cur]
    below = max((u for u in ladder if u < cur), default=None)
    above = min((u for u in ladder if u > cur), default=None)
    out = [cur]
    if below:
        out.append(below)
    # Bigger ubatch costs compute-buffer VRAM; don't even try 2048 on <12 GB.
    if above and not (above > 1024 and (vram_mb or 0) < 12288):
        out.append(above)
    return out


def build_stages(
    recipe: dict[str, Any],
    *,
    safe_kv_q8: bool = True,
    fa_supported: bool = True,
    fa_takes_value: bool = True,
    vram_mb: int | None = None,
) -> list[list[TuneStep]]:
    """The staged candidate grid for one model, given its current recipe."""
    stages: list[list[TuneStep]] = []

    kv_options = ["f16", "q8_0"] if safe_kv_q8 else ["f16"]
    stages.append([
        TuneStep(label=f"kv={v}", edits={"cache_type_k": v, "cache_type_v": v})
        for v in kv_options
    ])

    # MTP models are pinned to ubatch=8 (SSM compute-buffer OOM otherwise) —
    # sweeping ubatch there would break them, so skip the stage.
    if recipe.get("spec_type") != "draft-mtp":
        stages.append([
            TuneStep(label=f"ubatch={u}", edits={"ubatch_size": u, "batch_size": max(2048, u)})
            for u in _ubatch_candidates(recipe.get("ubatch_size"), vram_mb)
        ])

    if fa_supported:
        fa_options = list(FLASH_ATTN_VALUES) if fa_takes_value else ["on", "off"]
        stages.append([
            TuneStep(label=f"fa={v}", edits={"flash_attn": v}) for v in fa_options
        ])

    return stages


def _state_key(state: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in state.items()))


async def _apply_edits(
    client: httpx.AsyncClient, model_name: str, edits: dict[str, Any]
) -> str | None:
    """POST a partial recipe to /admin/models/{name}/edit. Returns error or None."""
    if not edits:
        return None
    try:
        r = await client.post(f"/admin/models/{model_name}/edit", json=edits)
    except httpx.HTTPError as e:
        return f"edit failed: {e}"
    if r.status_code != 200:
        return f"edit failed: {r.status_code} {r.text}"
    return None


def _restore_edits(original: dict[str, Any], touched: set[str]) -> dict[str, Any]:
    """Build the edit body that puts every touched axis back to its original value.

    Axes the original recipe didn't set are restored to their llama.cpp
    defaults explicitly (ubatch 512, batch 2048) or cleared (flash_attn null) —
    the edit endpoint has no generic "unset" operation.
    """
    defaults: dict[str, Any] = {
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "ubatch_size": 512,
        "batch_size": 2048,
        "flash_attn": None,
    }
    return {k: original.get(k, defaults.get(k)) for k in touched}


async def tune_model(
    server_url: str,
    model_name: str,
    *,
    target: str = "balanced",
    prompt_tokens: int = TUNE_PROMPT_TOKENS,
    gen_tokens: int = DEFAULT_GEN_TOKENS,
    apply: bool = True,
    cfg: Config | None = None,
) -> TuneReport:
    """Run the staged sweep against a live `arc-llama serve` instance.

    Every measurement goes through the server (same SYCL env, same router
    policy the user actually runs with). The model's recipe is edited in
    place between measurements and always left in a valid state: the winner
    when `apply` is True, the original otherwise.
    """
    if cfg is None:
        cfg = load_config()
    report = TuneReport(model=model_name, target=target)

    model = cfg.find_model(model_name)
    if model is None:
        report.error = f"Model '{model_name}' not found in config"
        return report
    gpu = cfg.find_gpu(model.gpu_pci_slot)

    from arc_llama.arch import Arch, profile_for
    from arc_llama.server_caps import probe_server_caps

    profile = profile_for(Arch(gpu.arch) if gpu and gpu.arch else Arch.UNKNOWN)
    caps = probe_server_caps(cfg.paths.llama_server)

    original = dict(model.recipe or {})
    stages = build_stages(
        original,
        safe_kv_q8=profile.safe_kv_q8,
        fa_supported=caps.supports_flash_attn,
        fa_takes_value=caps.flash_attn_takes_value,
        vram_mb=gpu.vram_mb if gpu else None,
    )

    touched: set[str] = set()
    for stage in stages:
        for step in stage:
            touched.update(step.edits.keys())
    # Canonical value for every axis the sweep will touch, as the model runs
    # today. Every candidate application sends the FULL state for these axes
    # (base + winners-so-far + this candidate), never a partial diff — the
    # edit endpoint persists whatever it's sent, so partial diffs would leave
    # a losing candidate's value in force and contaminate later measurements.
    base_state = _restore_edits(original, touched)

    measured: dict[tuple, tuple[BenchmarkResult, float | None]] = {}

    async def measure(state: dict[str, Any]) -> tuple[BenchmarkResult, float | None, str | None]:
        key = _state_key(state)
        if key in measured:
            res, sc = measured[key]
            return res, sc, "same config as an earlier run"
        err = await _apply_edits(client, model_name, state)
        if err:
            res = BenchmarkResult(
                model=model_name, ctx=0, cache_type_k="?", cache_type_v="?",
                prompt_tokens=prompt_tokens, gen_tokens=gen_tokens, error=err,
            )
            return res, None, None
        res = await benchmark_model(
            server_url, model_name,
            prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
            load=True, cfg=cfg,
        )
        sc = score_result(res, target)
        measured[key] = (res, sc)
        return res, sc, None

    async with httpx.AsyncClient(base_url=server_url, timeout=600.0) as client:
        # Baseline = the recipe as it stands today (base_state is exactly the
        # current effective config, just spelled explicitly).
        log.info("tune %s: measuring baseline", model_name)
        baseline_step = TuneStep(label="baseline", edits={})
        baseline_step.result, baseline_step.score, _ = await measure(dict(base_state))
        report.steps.append(baseline_step)
        report.baseline = baseline_step.result
        if baseline_step.score is None:
            report.error = (
                "baseline measurement failed"
                + (f": {baseline_step.result.error}" if baseline_step.result.error else "")
            )
            return report

        best_edits: dict[str, Any] = {}
        best_score: float = baseline_step.score
        best_result: BenchmarkResult = baseline_step.result

        for stage in stages:
            stage_winner: TuneStep | None = None
            for step in stage:
                state = {**base_state, **best_edits, **step.edits}
                log.info("tune %s: trying %s", model_name, step.label)
                step.result, step.score, step.skipped_reason = await measure(state)
                report.steps.append(step)
                if step.score is not None and step.score > best_score:
                    if stage_winner is None or step.score > (stage_winner.score or 0):
                        stage_winner = step
            if stage_winner is not None:
                stage_winner.chosen = True
                best_edits.update(stage_winner.edits)
                best_score = stage_winner.score  # type: ignore[assignment]
                if stage_winner.result is not None:
                    best_result = stage_winner.result

        report.best_edits = dict(best_edits)
        report.best = best_result

        # Leave the recipe in its final state: winner if applying, original
        # otherwise. Restore is needed either way because the last-tried
        # candidate's values are what the edit endpoint persisted.
        final_state = {**base_state, **best_edits} if apply else dict(base_state)
        err = await _apply_edits(client, model_name, final_state)
        if err:
            report.error = f"failed to write final recipe: {err}"
            return report
        report.applied = apply and bool(best_edits)

    return report


# ------------------------------------------------------------------
# Console output
# ------------------------------------------------------------------

def print_report(report: TuneReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if report.error:
        console.print(f"[red]tune failed for {report.model}: {report.error}[/red]")
        if not report.steps:
            return

    table = Table(title=f"Tune: {report.model} (target: {report.target})")
    table.add_column("Config")
    table.add_column("Prompt-eval")
    table.add_column("Generation")
    table.add_column("Score")
    table.add_column("")

    def _fmt(v: float | None) -> str:
        return f"{v:.1f} tok/s" if v else "—"

    for step in report.steps:
        r = step.result
        if r is None:
            continue
        if r.error:
            table.add_row(step.label, "[red]failed[/red]", "", "", "")
            continue
        note = "◀ chosen" if step.chosen else ("(cached)" if step.skipped_reason else "")
        table.add_row(
            step.label,
            _fmt(r.prompt_eval_tok_s),
            _fmt(r.generation_tok_s),
            f"{step.score:.1f}" if step.score is not None else "—",
            note,
        )
    console.print(table)

    if not report.best_edits:
        console.print("[dim]Baseline recipe is already the best of the tried configs.[/dim]")
        return
    imp = report.improvement_pct
    parts = []
    if imp["prompt_eval"] is not None:
        parts.append(f"prompt-eval {imp['prompt_eval']:+.1f}%")
    if imp["generation"] is not None:
        parts.append(f"generation {imp['generation']:+.1f}%")
    console.print(
        f"\n[bold]Best config:[/bold] {report.best_edits}"
        + (f"  ({', '.join(parts)} vs baseline)" if parts else "")
    )
    if report.applied:
        console.print("[green]Applied to the model's recipe and persisted.[/green]")
    else:
        console.print("[yellow]Dry run — original recipe restored. Re-run with --apply to keep it.[/yellow]")
