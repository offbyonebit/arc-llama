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

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from arc_llama.benchmark import DEFAULT_GEN_TOKENS, BenchmarkResult, benchmark_model
from arc_llama.config import Config, ModelConfig, load_config
from arc_llama.gguf_meta import (
    override_tensor_saved_bytes,
    propose_override_tensor_patterns,
    validate_override_patterns,
    weight_tensor_table,
)
from arc_llama.recipes import FLASH_ATTN_VALUES, PERF_COMPUTE_BUFFER_MB, KVCacheType
from arc_llama.router import (
    _VRAM_COMPUTE_BUFFER_MB,
    _estimate_model_vram_mb,
    min_moe_offload_layers,
)

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


_DeferredRestore = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class TuneReport:
    model: str
    target: str
    steps: list[TuneStep] = field(default_factory=list)
    best_edits: dict[str, Any] = field(default_factory=dict)
    baseline: BenchmarkResult | None = None
    best: BenchmarkResult | None = None
    applied: bool = False
    aborted: bool = False
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


def score_result(
    result: BenchmarkResult,
    target: str = "balanced",
    priority: str | None = None,
) -> float | None:
    """Higher is better. Errors and empty measurements score None (lose).

    ``priority`` comes from the workload profile's first-token-vs-throughput
    answer and tilts the balanced geometric mean 3:1 toward the axis the user
    said hurts more. It never lets a missing measurement win, and it does not
    change the single-axis "prompt"/"generation" targets.
    """
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
    # Weighted geometric mean, so a 2x regression on one axis can't be bought
    # back by a 2x win on the other plus rounding.
    w_pp, w_gen = {
        "first_token": (0.75, 0.25),
        "throughput": (0.25, 0.75),
    }.get(priority or "", (0.5, 0.5))
    return (pp ** w_pp) * (gen ** w_gen)


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
    kv_candidates: list[str] | None = None,
) -> list[list[TuneStep]]:
    """The staged candidate grid for one model, given its current recipe.

    ``kv_candidates`` overrides the default f16/q8_0 pair — the workload
    profile uses it to prune KV types that cannot hold the declared context
    within the GPU's VRAM, so the tuner never offers a candidate that cannot
    boot the user's actual workload.
    """
    stages: list[list[TuneStep]] = []

    if kv_candidates is not None:
        kv_options = [v for v in kv_candidates if v != "q8_0" or safe_kv_q8]
    else:
        kv_options = ["f16", "q8_0"] if safe_kv_q8 else ["f16"]
    if kv_options:
        stages.append([
            TuneStep(label=f"kv={v}", edits={"cache_type_k": v, "cache_type_v": v})
            for v in kv_options
        ])

    # ubatch is free to sweep even for draft-mtp models: the earlier forced
    # -ub 8 for MTP was folklore; B60 measurements run MTP fine at 1024.
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
    def _norm(v):
        if isinstance(v, list):
            return tuple(v)
        return v
    return tuple(sorted((k, str(_norm(v))) for k, v in state.items()))


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
        "n_cpu_moe": None,
        "override_tensor": None,
    }
    return {k: original.get(k, defaults.get(k)) for k in touched}


def _apply_state_to_recipe(model: ModelConfig, state: dict[str, Any]) -> None:
    """Write a sweep state directly into the model's in-memory recipe.

    Mirrors the edit endpoint's field rules (None clears, n_cpu_moe and
    override_tensor are mutually exclusive, n_cpu_moe=0 clears) so the
    last-resort restore path below persists exactly what the endpoint would
    have. Only used when the endpoint itself cannot be reached — the values
    here were already validated when the sweep applied them earlier.
    """
    recipe = dict(model.recipe or {})
    for key, value in state.items():
        if value is None:
            recipe.pop(key, None)
        elif key == "n_cpu_moe":
            if int(value) == 0:
                recipe.pop("n_cpu_moe", None)
            else:
                recipe["n_cpu_moe"] = int(value)
                recipe.pop("override_tensor", None)
        elif key == "override_tensor":
            recipe["override_tensor"] = list(value)
            recipe.pop("n_cpu_moe", None)
        else:
            recipe[key] = value
    model.recipe = recipe


async def _restore_final_state(
    client: httpx.AsyncClient,
    model_name: str,
    final_state: dict[str, Any],
    cfg: Config | None,
) -> str | None:
    """Apply the end-of-sweep recipe, retrying, with a direct-write fallback.

    The finally block that calls this is the only thing standing between the
    user and a config file that still holds the last *candidate* the sweep
    measured — a recipe that may OOM or fail to load on the next request. A
    single failed POST must therefore not be the end of the story: retry the
    endpoint a few times, and if it stays broken, write the intended recipe
    to the local config directly (in the background autotuner this cfg IS the
    server's live Config; from the CLI it at least repairs the on-disk file
    the next server start will read). Returns None on success, else the
    error string from the last endpoint attempt.
    """
    err: str | None = None
    for attempt in range(3):
        err = await _apply_edits(client, model_name, final_state)
        if err is None:
            return None
        if attempt < 2:
            await asyncio.sleep(0.5 * (attempt + 1))
    log.error(
        "tune %s: final recipe restore failed after 3 attempts: %s",
        model_name,
        err,
    )
    if cfg is not None:
        model = cfg.find_model(model_name)
        if model is not None:
            try:
                from arc_llama.config import default_config_path

                _apply_state_to_recipe(model, final_state)
                cfg.save(default_config_path())
                log.warning(
                    "tune %s: restored the final recipe via a direct config "
                    "write after the edit endpoint failed",
                    model_name,
                )
                return None
            except Exception as e:  # noqa: BLE001
                log.critical(
                    "tune %s: direct config fallback also failed: %s. The "
                    "persisted recipe still holds the last sweep candidate, "
                    "which may not load. Intended recipe: %s",
                    model_name,
                    e,
                    final_state,
                )
    return err


# ---------------------------------------------------------------------------
# MoE expert offload as a measured axis
# ---------------------------------------------------------------------------

_OFFLOAD_STAGE = "offload"
"""Sentinel in the stage plan: the offload stage is resolved lazily, after
the KV stage, because its minimum feasible N must be computed against the KV
type that *won* — KV and offload draw from the same VRAM budget."""


@dataclass
class _OffloadInfo:
    """What the offload stage needs to know about one MoE model."""
    n_layers: int
    vram_mb: int


def _probe_offload_info(model: ModelConfig, gpu: Any) -> _OffloadInfo | None:
    """Return offload relevance for this model, or None when the stage must
    not exist at all: non-MoE (no routed-expert tensors), unknown VRAM
    budget, or an unreadable GGUF. Runs synchronously — callers wrap it in
    asyncio.to_thread (GGUF tensor-table read).
    """
    if gpu is None or not gpu.vram_mb:
        return None
    from arc_llama.gguf_meta import scan_weight_tensors

    scan = scan_weight_tensors(model.path)
    if scan is None:
        return None
    _, expert_by_layer = scan
    if not expert_by_layer:
        return None
    return _OffloadInfo(n_layers=max(expert_by_layer) + 1, vram_mb=gpu.vram_mb)


class _OverrideTensorInfo:
    """Carry real tensor data for the override-tensor refinement step."""

    def __init__(self, table: dict[str, int], candidates: list[str]) -> None:
        self.table = table
        self.candidates = candidates


def _offload_probe_step(n_layers: int) -> int:
    """Probe distance in layers around the estimated minimum offload.

    max(1, n_layers // 8): a single layer is only ~1-2% of an MoE model's
    expert bytes, smaller than the estimator's noise from the KV-per-token
    and compute-buffer terms, so probing one layer off would mostly measure
    noise. An eighth of the stack is large enough for a real correction and
    small enough that we don't leave several layers of avoidable per-token
    PCIe expert traffic on the table.
    """
    return max(1, n_layers // 8)


def _prune_ubatch_stage(
    stage: list[TuneStep],
    *,
    model: ModelConfig,
    vram_mb: int,
    ctx: int,
    kv_type: KVCacheType,
    n_cpu_moe: int | None = None,
    override_tensor: list[str] | None = None,
) -> None:
    """Mark ubatch candidates that no longer fit under the chosen offload.

    A larger ubatch grows the compute buffer (~768 MiB at the stock 512,
    ~1536 MiB at 1024+); with expert offload in force the VRAM budget is
    tight by construction, so candidates that would OOM are skipped here
    rather than measured into a load failure. Synchronous GGUF math — wrap
    in asyncio.to_thread.
    """
    for step in stage:
        u = int(step.edits.get("ubatch_size", 512))
        buffer_mb = PERF_COMPUTE_BUFFER_MB if u > 512 else _VRAM_COMPUTE_BUFFER_MB
        est = _estimate_model_vram_mb(
            model, ctx=ctx, kv_type=kv_type, n_cpu_moe=n_cpu_moe,
            override_tensor=override_tensor,
            compute_buffer_mb=buffer_mb,
        )
        if est is not None and est > vram_mb:
            step.skipped_reason = (
                f"ubatch {u} compute buffer would not fit in {vram_mb} MiB "
                f"with offload={override_tensor or n_cpu_moe}"
            )


async def tune_model(
    server_url: str,
    model_name: str,
    *,
    target: str = "balanced",
    prompt_tokens: int = TUNE_PROMPT_TOKENS,
    gen_tokens: int = DEFAULT_GEN_TOKENS,
    apply: bool = True,
    cfg: Config | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_stage: Callable[[str, int, int], None] | None = None,
    on_deferred_restore: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> TuneReport:
    """Run the staged sweep against a live `arc-llama serve` instance.

    Every measurement goes through the server (same SYCL env, same router
    policy the user actually runs with). The model's recipe is edited in
    place between measurements and always left in a valid state: the winner
    when `apply` is True, the original otherwise.

    The ``should_abort`` hook is checked before every measurement and between
    stages. When it returns True, the sweep stops, ``report.aborted`` is set,
    and the best fully-completed stage recipe is restored. This lets the
    background auto-tuner abandon instantly when a real request arrives.

    If ``on_deferred_restore`` is provided and the sweep is aborted, the caller
    is responsible for applying ``final_state`` once it is safe to do so. The
    manual CLI path omits this callback and restores immediately.

    Round 7: ``override_tensor`` is a refinement of the offload stage.
    It is only evaluated when offload is needed at all, and only if the
    generated pattern frees enough bytes to beat ``--n-cpu-moe``. At most
    two extra measurements are allowed there: one cheaper projection class
    and one that matches every routed-expert tensor (the `-ot` equivalent of
    the full ``--n-cpu-moe`` sweep winner). If no generated pattern fits the
    model or none beats the ``n_cpu_moe`` winner, the simpler flag is kept.
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
    # probe_server_caps shells out to `llama-server --help` (subprocess.run,
    # up to a 10 s timeout on a cold cache). Running it inline here blocks the
    # event loop at sweep start — which is exactly when /admin/tune/status was
    # observed stalling — so push it to a thread.
    caps = await asyncio.to_thread(probe_server_caps, cfg.paths.llama_server)

    original = dict(model.recipe or {})

    from arc_llama import workload

    # Workload profile: prune KV candidates that cannot hold the declared
    # context in VRAM (an f16 ranking taken at 1k depth is wrong when the
    # user's real context cannot even boot with f16), weight scoring toward
    # the declared priority, and measure depth-sensitive axes at the declared
    # depth rather than at the shallow default.
    wl_ctx = workload.target_ctx(cfg.workload)
    kv_candidates: list[str] | None = None
    if wl_ctx is not None:
        # kv_fits_at_ctx reads GGUF metadata (sync file I/O); keep it off the
        # event loop for the same reason as the caps probe above.
        def _fitting_kv_types() -> list[str]:
            return [
                kv for kv in ("f16", "q8_0")
                if workload.kv_fits_at_ctx(model, gpu, kv, wl_ctx)
            ]

        fitting = await asyncio.to_thread(_fitting_kv_types)
        if not fitting:
            # Nothing fits at the declared context: offer only the KV type
            # the model currently runs rather than an empty stage or a
            # candidate that cannot boot the workload.
            current_kv = str(original.get("cache_type_k", "f16"))
            log.warning(
                "tune %s: no KV type is estimated to fit ctx=%d in %s MiB; "
                "restricting the sweep to the current KV type %s",
                model_name, wl_ctx, gpu.vram_mb if gpu else 0, current_kv,
            )
            fitting = [current_kv]
        kv_candidates = fitting
    priority = workload.score_priority(cfg)
    deep_tokens = workload.deep_prompt_tokens(cfg, original)

    stages = build_stages(
        original,
        safe_kv_q8=profile.safe_kv_q8,
        fa_supported=caps.supports_flash_attn,
        fa_takes_value=caps.flash_attn_takes_value,
        vram_mb=gpu.vram_mb if gpu else None,
        kv_candidates=kv_candidates,
    )

    # Partition the static stages. The MoE offload stage sits between KV and
    # ubatch and is resolved lazily: its minimum feasible N must be computed
    # against the KV type that *won*, because KV and offload draw from the
    # same VRAM budget — fixing offload before KV would pin it to a KV
    # choice that later changes.
    kv_stages = [s for s in stages if any("cache_type_k" in step.edits for step in s)]
    ubatch_stages = [s for s in stages if any("ubatch_size" in step.edits for step in s)]
    fa_stages = [s for s in stages if any("flash_attn" in step.edits for step in s)]

    # MoE offload relevance reads the GGUF tensor table; keep it off the
    # event loop like the caps probe above.
    # Tiny or malformed placeholder files are common in configuration tests;
    # parse those inline so they cannot strand an executor worker in a GGUF
    # reader while the event loop waits for the result.
    try:
        tiny_placeholder = Path(model.path).stat().st_size < 1024
    except OSError:
        tiny_placeholder = True
    if tiny_placeholder:
        offload_info = _probe_offload_info(model, gpu)
    else:
        offload_info = await asyncio.to_thread(_probe_offload_info, model, gpu)

    # Round 7: gather the real tensor table and generate candidate override-tensor
    # patterns. These are derived from the model's actual tensor names, not
    # from a hardcoded naming convention, so a pattern that matches zero tensors
    # can be caught before it is ever passed to llama-server.
    ot_info: _OverrideTensorInfo | None = None
    if offload_info is not None:
        table = weight_tensor_table(model.path) if tiny_placeholder else await asyncio.to_thread(
            weight_tensor_table, model.path
        )
        if table:
            ot_info = _OverrideTensorInfo(
                table=table,
                candidates=await asyncio.to_thread(propose_override_tensor_patterns, table),
            )

    touched: set[str] = set()
    for stage in stages:
        for step in stage:
            touched.update(step.edits.keys())
    if offload_info is not None:
        touched.add("n_cpu_moe")
    if ot_info is not None:
        # Both for the refinement's own gate and for the restore: -ot is set
        # inside the offload stage rather than by a pre-declared step, so it
        # never reaches `touched` via the loop above, and an untouched axis is
        # one `_restore_edits` will not clear afterwards.
        touched.add("override_tensor")
    # Canonical value for every axis the sweep will touch, as the model runs
    # today. Every candidate application sends the FULL state for these axes
    # (base + winners-so-far + this candidate), never a partial diff — the
    # edit endpoint persists whatever it's sent, so partial diffs would leave
    # a losing candidate's value in force and contaminate later measurements.
    base_state = _restore_edits(original, touched)

    # Seed the sweep from the minimum feasible offload when the recipe's own
    # offload is below it — critically when registration set none and the
    # model does not load at zero offload at all. Without this the baseline
    # measurement itself OOMs and the sweep dies with "baseline measurement
    # failed" before the offload stage can rescue it. The seed lives in
    # best_edits (not base_state) so a dry-run restore still returns the
    # recipe to its true original state.
    best_edits: dict[str, Any] = {}
    if offload_info is not None:
        current_offload = int(original.get("n_cpu_moe") or 0)
        seed_vram = gpu.vram_mb if gpu is not None else None
        if seed_vram is None:
            log.warning(
                "tune %s: cannot seed minimum offload because GPU VRAM is unknown",
                model_name,
            )
        else:
            n0 = await asyncio.to_thread(
                min_moe_offload_layers, model, seed_vram,
                ctx=int(base_state.get("ctx", 8192)),
                kv_type=KVCacheType(str(base_state.get("cache_type_k", "f16"))),
            )
            if n0 is not None and n0 > current_offload:
                log.info(
                    "tune %s: recipe offload (%d) is below the estimated minimum "
                    "(%d layer(s)); starting the sweep from the minimum",
                    model_name, current_offload, n0,
                )
                best_edits["n_cpu_moe"] = n0

    measured: dict[tuple, tuple[BenchmarkResult, float | None]] = {}

    headers = (
        {"Authorization": f"Bearer {cfg.server.admin_token}"}
        if cfg.server.admin_token
        else {}
    )

    async def measure(
        client: httpx.AsyncClient,
        state: dict[str, Any],
        *,
        depth_tokens: int | None = None,
    ) -> tuple[BenchmarkResult, float | None, str | None]:
        # depth_tokens overrides prompt_tokens for stages measured at the
        # declared workload depth; the cache key must include it because the
        # same config at 1k and at 32k is two different measurements.
        pt = depth_tokens or prompt_tokens
        key = (_state_key(state), pt)
        if key in measured:
            res, sc = measured[key]
            return res, sc, "same config as an earlier run"
        err = await _apply_edits(client, model_name, state)
        if err:
            res = BenchmarkResult(
                model=model_name, ctx=0, cache_type_k="?", cache_type_v="?",
                prompt_tokens=pt, gen_tokens=gen_tokens, error=err,
            )
            return res, None, None
        if should_abort and should_abort():
            # The edit above awaited, which is exactly the window in which a
            # real request registers (the caller's own abort check ran in the
            # same synchronous segment as this function's entry, so checking
            # earlier here could never observe it). Bail before the benchmark:
            # its load=True would evict the user's just-started model.
            report.aborted = True
            res = BenchmarkResult(
                model=model_name, ctx=0, cache_type_k="?", cache_type_v="?",
                prompt_tokens=pt, gen_tokens=gen_tokens,
                error="aborted before measurement",
            )
            return res, None, "aborted before measurement"
        try:
            res = await benchmark_model(
                server_url, model_name,
                prompt_tokens=pt, gen_tokens=gen_tokens,
                load=True, cfg=cfg,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A candidate that crashes the backend (OOM, SIGSEGV, hung health
            # check, etc.) must lose the stage rather than aborting the whole
            # sweep. Record the failure and let the tuner move on.
            log.warning(
                "tune %s: measurement failed for %s at %d tokens: %s",
                model_name, _state_key(state), pt, exc,
            )
            res = BenchmarkResult(
                model=model_name,
                ctx=int(state.get("ctx", 0)),
                cache_type_k=str(state.get("cache_type_k", "?")),
                cache_type_v=str(state.get("cache_type_v", "?")),
                prompt_tokens=pt,
                gen_tokens=gen_tokens,
                error=f"measurement failed: {exc}",
            )
        sc = score_result(res, target, priority)
        measured[key] = (res, sc)
        return res, sc, None

    async with httpx.AsyncClient(
        base_url=server_url, timeout=600.0, headers=headers
    ) as client:
        # Baseline = the recipe as it stands today (base_state is exactly the
        # current effective config, just spelled explicitly — plus the
        # minimum-feasible-offload seed when one was needed to load at all).
        log.info("tune %s: measuring baseline", model_name)
        baseline_step = TuneStep(label="baseline", edits={})
        best_result: BenchmarkResult | None = None
        # Score of whatever produced best_result. BenchmarkResult carries raw
        # tok/s only; the workload-weighted score lives on TuneStep, so the
        # winner's score has to be carried alongside it for later stages to
        # compare against.
        best_score: float | None = None

        async def _try_offload(n: int) -> TuneStep:
            st = TuneStep(label=f"n_cpu_moe={n}", edits={"n_cpu_moe": n})
            log.info("tune %s: trying n_cpu_moe=%d", model_name, n)
            st.result, st.score, st.skipped_reason = await measure(
                client, {**base_state, **best_edits, "n_cpu_moe": n}
            )
            report.steps.append(st)
            return st

        async def run_offload_stage(stage_index: int, total: int) -> bool:
            """Minimum-feasible-offload search. Returns True when the stage
            measured something (False = skipped, reason recorded in report).

            Deliberately not a ladder: more offload is monotonically worse
            for throughput and monotonically better for fit, so the optimum
            is the minimum that actually loads. Measure the estimate, probe
            one step below on success (the estimator may be conservative and
            less offload may fit), escalate upward on load failure (it may be
            optimistic and OOM). Three measurements maximum, each paying a
            cold start.
            """
            nonlocal best_result, best_score
            assert offload_info is not None
            kv_winner = KVCacheType(
                str(best_edits.get("cache_type_k", base_state.get("cache_type_k", "f16")))
            )
            eff_ctx = wl_ctx if wl_ctx is not None else int(base_state.get("ctx", 8192))
            n_min = await asyncio.to_thread(
                min_moe_offload_layers, model, offload_info.vram_mb,
                ctx=eff_ctx, kv_type=kv_winner,
            )
            if n_min is None:
                report.steps.append(TuneStep(
                    label="n_cpu_moe", edits={},
                    skipped_reason="expert tensor bytes unknown; offload cannot be measured",
                ))
                return False
            if n_min == 0:
                log.info(
                    "tune %s: fits with zero offload at ctx=%d kv=%s; offload stage skipped",
                    model_name, eff_ctx, kv_winner.value,
                )
                report.steps.append(TuneStep(
                    label="n_cpu_moe=0", edits={},
                    skipped_reason="fits without offload under the winning KV type; stage skipped",
                ))
                return False
            if stage_index > 0 and on_stage is not None:
                on_stage(f"n_cpu_moe~{n_min}", stage_index, total)
            step = _offload_probe_step(offload_info.n_layers)
            anchor = await _try_offload(n_min)
            winner_step: TuneStep | None = None
            if anchor.score is None:
                # Estimator optimistic: the minimum OOM'd at load. More
                # offload only helps fit, so escalate — and never probe
                # below a level that failed to load.
                escalations: list[int] = []
                n = n_min + step
                while n <= offload_info.n_layers and len(escalations) < 2:
                    escalations.append(n)
                    n += step
                for n in escalations:
                    if should_abort and should_abort():
                        report.aborted = True
                        return True
                    st = await _try_offload(n)
                    if st.score is not None:
                        st.chosen = True
                        winner_step = st
                        break
                if winner_step is None:
                    log.warning(
                        "tune %s: no offload level up to %d layers loads; "
                        "leaving the recipe offload unchanged",
                        model_name, escalations[-1] if escalations else n_min,
                    )
                    return True
            else:
                anchor.chosen = True
                winner_step = anchor
                below = n_min - step
                if 0 <= below < n_min:
                    if should_abort and should_abort():
                        report.aborted = True
                        return True
                    st = await _try_offload(below)
                    if st.score is not None and st.score > (anchor.score or 0):
                        # Estimator was conservative: less offload fits and
                        # measures faster. One step only — do not chase the
                        # floor with more cold starts.
                        anchor.chosen = False
                        st.chosen = True
                        winner_step = st
                    # A failing below-probe is information, not an error:
                    # n_min stands and we stop probing downward.
                # No upward probe on the success path: more offload is
                # monotonically worse for throughput, so n_min+step could
                # never win once n_min is proven to load.
            best_edits["n_cpu_moe"] = winner_step.edits["n_cpu_moe"]
            if winner_step.result is not None:
                best_result = winner_step.result
                best_score = winner_step.score
            return True

        async def _try_override_tensor(pats: list[str]) -> TuneStep:
            label = "-ot " + ",".join(pats)
            st = TuneStep(label=label, edits={"override_tensor": list(pats)})
            log.info("tune %s: trying %s", model_name, label)
            state = {**base_state, **best_edits, "override_tensor": list(pats)}
            # Enforce mutual exclusion: -ot and --n-cpu-moe are alternatives.
            state.pop("n_cpu_moe", None)
            st.result, st.score, st.skipped_reason = await measure(client, state)
            report.steps.append(st)
            return st

        async def run_override_tensor_refinement() -> bool:
            """Optional refinement of the offload winner with -ot patterns.

            Only runs when the offload stage found a non-zero --n-cpu-moe that
            fits, because without an offload need there is no point paying a
            cold start for a finer-grained alternative. At most two extra
            measurements are performed:

              1. The cheapest projection-class pattern (smallest bytes moved).
                 If this alone frees enough bytes to fit, it will usually be
                 faster than full layer offload and is the common win.
              2. The catch-all routed-expert pattern (the -ot equivalent of
                 the current --n-cpu-moe winner). This provides an upper bound
                 on the bytes a generated -ot pattern can free.

            The refinement only replaces the --n-cpu-moe winner when an -ot
            candidate both loads and scores higher, keeping the simpler flag
            when the refinement does not beat it.
            """
            nonlocal best_result, best_score
            if ot_info is None or not ot_info.candidates:
                return False
            if not best_edits.get("n_cpu_moe"):
                return False
            if "override_tensor" not in touched:
                return False
            n_min = int(best_edits.get("n_cpu_moe", 0))
            if n_min <= 0:
                return False
            n_moe_winner = n_min
            moe_winner_score = best_score if best_score is not None else 0.0
            # Pick up to two candidates: cheapest projection and the catch-all.
            unique: list[str] = []
            for c in ot_info.candidates:
                if c not in unique:
                    unique.append(c)
            candidates = unique[:1]
            if unique[-1] not in candidates:
                candidates.append(unique[-1])
            # Validate every generated pattern before measuring.
            ok, err = validate_override_patterns(ot_info.table, candidates)
            if not ok:
                log.warning("tune %s: override_tensor patterns rejected: %s", model_name, err)
                report.steps.append(TuneStep(
                    label="-ot rejected", edits={},
                    skipped_reason=f"generated -ot patterns invalid: {err}",
                ))
                return False
            # A candidate only wins if it actually fits:
            # its freed bytes must make _estimate_model_vram_mb <= vram_mb.
            vram_mb = (gpu.vram_mb or 0) if gpu is not None else 0
            eff_ctx = wl_ctx if wl_ctx is not None else int(base_state.get("ctx", 8192))
            kv_winner = KVCacheType(
                str(best_edits.get("cache_type_k", base_state.get("cache_type_k", "f16")))
            )
            viable: list[str] = []
            for c in candidates:
                saved_mb = override_tensor_saved_bytes(ot_info.table, [c]) // (1_048_576)
                est = _estimate_model_vram_mb(
                    model, ctx=eff_ctx, kv_type=kv_winner,
                    override_tensor=[c], compute_buffer_mb=_VRAM_COMPUTE_BUFFER_MB,
                )
                # Estimator must think it fits; also require it frees at least
                # as many bytes as the n_cpu_moe winner freed, otherwise it
                # cannot possibly beat fit in the same budget.
                if est is not None and est <= vram_mb and saved_mb > 0:
                    viable.append(c)
            if not viable:
                log.info(
                    "tune %s: no generated -ot pattern estimated to fit; keeping n_cpu_moe=%d",
                    model_name, n_moe_winner,
                )
                return False
            ot_winner: TuneStep | None = None
            for pat in viable:
                if should_abort and should_abort():
                    report.aborted = True
                    return True
                st = await _try_override_tensor([pat])
                if st.score is not None and (ot_winner is None or st.score > (ot_winner.score or 0)):
                    ot_winner = st
            if ot_winner is not None and (ot_winner.score or 0) > moe_winner_score:
                chosen = ot_winner.edits["override_tensor"]
                log.info(
                    "tune %s: override_tensor %s beats n_cpu_moe=%d (score %.1f > %.1f); using -ot",
                    model_name, chosen, n_moe_winner,
                    ot_winner.score or 0, moe_winner_score,
                )
                best_edits.pop("n_cpu_moe", None)
                best_edits["override_tensor"] = chosen
                if ot_winner.result is not None:
                    best_result = ot_winner.result
                    best_score = ot_winner.score
                return True
            log.info(
                "tune %s: override_tensor did not beat n_cpu_moe=%d; keeping simpler flag",
                model_name, n_moe_winner,
            )
            return False

        try:
            if should_abort and should_abort():
                report.aborted = True
                # Do not return here; fall through to the finally so the
                # restore (deferred or immediate) always runs.
            else:
                baseline_step.result, baseline_step.score, _ = await measure(
                    client, {**base_state, **best_edits}
                )
            report.steps.append(baseline_step)
            if (
                baseline_step.score is None
                and not report.aborted
                and offload_info is not None
            ):
                # MoE model whose recipe does not load even from the seeded
                # minimum (the estimator was optimistic). Run the offload
                # search now, ahead of KV, to find a level that loads; the
                # post-KV offload stage still recomputes against the KV
                # winner afterwards.
                log.warning(
                    "tune %s: baseline failed (%s); searching for a loadable "
                    "offload level first",
                    model_name,
                    baseline_step.result.error if baseline_step.result else "unknown",
                )
                await run_offload_stage(0, 0)
                rescued_n = best_edits.get("n_cpu_moe")
                if not report.aborted and rescued_n is not None:
                    res, sc, _ = await measure(client, {**base_state, **best_edits})
                    if sc is not None:
                        baseline_step.label = f"baseline (n_cpu_moe={rescued_n})"
                        baseline_step.result, baseline_step.score = res, sc
            report.baseline = baseline_step.result
            if baseline_step.score is None:
                report.error = (
                    "baseline measurement failed"
                    + (f": {baseline_step.result.error}" if baseline_step.result and baseline_step.result.error else "")
                )
                # Fall through to finally so the original recipe is restored.
                return report

            best_result = baseline_step.result
            best_score = baseline_step.score

            stage_plan: list[Any] = list(kv_stages)
            if offload_info is not None:
                stage_plan.append(_OFFLOAD_STAGE)
            stage_plan.extend(ubatch_stages)
            stage_plan.extend(fa_stages)
            total_stages = len(stage_plan)
            stage_index = 0
            for item in stage_plan:
                if should_abort and should_abort():
                    report.aborted = True
                    break
                if item is _OFFLOAD_STAGE:
                    ran = await run_offload_stage(stage_index + 1, total_stages)
                    if ran:
                        stage_index += 1
                        # Round 7: give -ot a chance to refine the n_cpu_moe winner.
                        if ot_info is not None and ot_info.candidates:
                            await run_override_tensor_refinement()
                    else:
                        total_stages -= 1
                    if report.aborted:
                        break
                    continue
                stage = item
                if stage in ubatch_stages:
                    # The ubatch stage must respect the chosen offload: a
                    # larger ubatch grows the compute buffer, and with expert
                    # offload in force the budget is tight by construction.
                    eff_offload = int(
                        best_edits.get("n_cpu_moe") or base_state.get("n_cpu_moe") or 0
                    )
                    eff_ot = best_edits.get("override_tensor") or base_state.get("override_tensor")
                    if (eff_offload > 0 or eff_ot) and gpu is not None and gpu.vram_mb:
                        kv_winner = KVCacheType(
                            str(best_edits.get(
                                "cache_type_k", base_state.get("cache_type_k", "f16")
                            ))
                        )
                        eff_ctx = (
                            wl_ctx if wl_ctx is not None
                            else int(base_state.get("ctx", 8192))
                        )
                        await asyncio.to_thread(
                            _prune_ubatch_stage, stage,
                            model=model, vram_mb=gpu.vram_mb, ctx=eff_ctx,
                            kv_type=kv_winner, n_cpu_moe=eff_offload,
                            override_tensor=eff_ot,
                        )
                stage_index += 1
                if on_stage is not None:
                    on_stage(stage[0].label[:32], stage_index, total_stages)
                # Depth-sensitive axes (KV type, ubatch) are measured at the
                # declared workload depth when one is declared; everything
                # else stays shallow. Rankings taken at 1k do not necessarily
                # hold at 32k, so candidates in a stage are only ever compared
                # against measurements taken at that stage's depth.
                stage_depth = (
                    deep_tokens
                    if deep_tokens is not None
                    and workload.stage_is_depth_sensitive(
                        {k for step in stage for k in step.edits}
                    )
                    else None
                )
                # Anchor: (re)measure the current best state at this stage's
                # depth — almost always a cache hit — so a candidate only wins
                # by beating the incumbent measured at the same depth.
                anchor_state = {**base_state, **best_edits}
                _, anchor_score, _ = await measure(
                    client, anchor_state, depth_tokens=stage_depth
                )
                bar = anchor_score if anchor_score is not None else 0.0
                stage_winner: TuneStep | None = None
                for step in stage:
                    if should_abort and should_abort():
                        report.aborted = True
                        break
                    if step.skipped_reason is not None:
                        # Pre-pruned (e.g. ubatch candidate that would not
                        # fit under the chosen offload): record, don't
                        # measure into an OOM.
                        report.steps.append(step)
                        continue
                    state = {**base_state, **best_edits, **step.edits}
                    log.info("tune %s: trying %s", model_name, step.label)
                    step.result, step.score, step.skipped_reason = await measure(
                        client, state, depth_tokens=stage_depth
                    )
                    report.steps.append(step)
                    if step.score is not None and step.score > bar:
                        if stage_winner is None or step.score > (stage_winner.score or 0):
                            stage_winner = step
                if report.aborted:
                    # Discard any partial stage results; only completed stages
                    # may contribute to the final restored recipe.
                    break
                if stage_winner is not None:
                    stage_winner.chosen = True
                    best_edits.update(stage_winner.edits)
                    if stage_winner.result is not None:
                        best_result = stage_winner.result
                        best_score = stage_winner.score
        finally:
            report.best_edits = dict(best_edits)
            report.best = best_result
            # Leave the recipe in its final state: winner if applying, original
            # otherwise. Restore is needed either way because the last-tried
            # candidate's values are what the edit endpoint persisted. This runs
            # even when CancelledError aborts mid-sweep.
            final_state = {**base_state, **best_edits} if apply else dict(base_state)
            if report.aborted and on_deferred_restore is not None:
                # Hand the restore to the caller so it can wait until real
                # requests have drained. The restore itself is not skipped.
                await on_deferred_restore(final_state)
            else:
                err = await _restore_final_state(client, model_name, final_state, cfg)
                if err and report.error is None:
                    report.error = f"failed to write final recipe: {err}"
                if not report.error:
                    report.applied = apply and bool(best_edits) and not report.aborted

    return report


async def tune_all(
    server_url: str,
    model_names: list[str],
    *,
    target: str = "balanced",
    prompt_tokens: int = TUNE_PROMPT_TOKENS,
    gen_tokens: int = DEFAULT_GEN_TOKENS,
    apply: bool = True,
    cfg: Config | None = None,
    on_start: Callable[[str, int, int], None] | None = None,
    on_done: Callable[[TuneReport], None] | None = None,
) -> list[TuneReport]:
    """Tune every model in *model_names* sequentially, returning one report per attempt.

    A failure on one model never aborts the rest.  ``KeyboardInterrupt`` is
    re-raised so the caller can stop the whole sweep.
    """
    reports: list[TuneReport] = []
    total = len(model_names)
    for idx, name in enumerate(model_names, start=1):
        log.info("tune-all: [%d/%d] %s", idx, total, name)
        if on_start is not None:
            on_start(name, idx, total)
        try:
            report = await tune_model(
                server_url,
                name,
                target=target,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                apply=apply,
                cfg=cfg,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            report = TuneReport(model=name, target=target, error=str(exc))
        reports.append(report)
        if on_done is not None:
            on_done(report)
    return reports


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
        console.print("[yellow]Dry run — original recipe restored. Re-run without --dry-run to keep it.[/yellow]")


def print_multi_summary(reports: list[TuneReport]) -> None:
    """Print a combined rich table summarising multiple ``TuneReport`` results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Tune summary")
    table.add_column("Model")
    table.add_column("Gen tok/s (base→best)")
    table.add_column("Prompt tok/s (base→best)")
    table.add_column("Gen Δ")
    table.add_column("Applied")
    table.add_column("Status")

    def _pair(base: float | None, best: float | None) -> str:
        if base is None or best is None:
            return "-"
        return f"{base:.1f} → {best:.1f}"

    n_ok = 0
    n_applied = 0
    for report in reports:
        if report.error is None:
            n_ok += 1
        if report.applied:
            n_applied += 1

        baseline = report.baseline
        best = report.best
        if report.error is not None or baseline is None or best is None:
            gen_cell = "-"
            prompt_cell = "-"
        else:
            gen_cell = _pair(baseline.generation_tok_s, best.generation_tok_s)
            prompt_cell = _pair(baseline.prompt_eval_tok_s, best.prompt_eval_tok_s)

        gen_delta = report.improvement_pct.get("generation")
        if gen_delta is None:
            delta_cell = "-"
        else:
            delta_cell = f"{gen_delta:+.1f}%"

        applied_cell = "yes" if report.applied else "no"
        if report.error is None:
            status_cell = "ok"
        else:
            status_cell = report.error[:40]

        table.add_row(
            report.model,
            gen_cell,
            prompt_cell,
            delta_cell,
            applied_cell,
            status_cell,
        )

    console.print(table)
    console.print(f"{n_ok}/{len(reports)} tuned OK, {n_applied} recipe(s) updated.")
