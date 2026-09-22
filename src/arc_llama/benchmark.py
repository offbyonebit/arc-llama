"""Benchmark harness for arc-llama models.

Measures prompt-eval throughput, generation throughput, and VRAM usage
for a registered model. Can run a single shot or sweep ctx/KV configs.

All measurements go through the running arc-llama serve instance so the
benchmark inherits the correct SYCL env, arch profile, and router policy.
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config, load_config

log = logging.getLogger("arc_llama.benchmark")

WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog. "
# We repeat this to build arbitrary-length prompts without tokenisation.

DEFAULT_PROMPT_TOKENS = 512
DEFAULT_GEN_TOKENS = 128

# How many repeat runs to summarise by median for stable numbers.
REPEAT_PROMPT_EVAL = 3
REPEAT_GENERATION = 3


@dataclass
class BenchmarkResult:
    """One benchmark measurement."""

    model: str
    ctx: int
    cache_type_k: str
    cache_type_v: str
    prompt_tokens: int
    gen_tokens: int
    prompt_eval_tok_s: float | None = None
    prompt_eval_ms: float | None = None
    generation_tok_s: float | None = None
    generation_ms: float | None = None
    measured_prompt_tokens: int | None = None
    measured_gen_tokens: int | None = None
    prompt_eval_samples: list[float] = field(default_factory=list)
    generation_samples: list[float] = field(default_factory=list)
    prompt_eval_min_tok_s: float | None = None
    prompt_eval_max_tok_s: float | None = None
    generation_min_tok_s: float | None = None
    generation_max_tok_s: float | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    jit_warmup_s: float | None = None
    error: str | None = None

    @property
    def vram_pct(self) -> float | None:
        if self.vram_used_mb is None or self.vram_total_mb is None:
            return None
        return round(self.vram_used_mb / self.vram_total_mb * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# sysfs VRAM helpers
# ------------------------------------------------------------------


def _find_drm_card(pci_slot: str) -> Path | None:
    """Find the /sys/class/drm/cardN path for a given PCI slot."""
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return None
    for entry in drm_path.iterdir():
        if not entry.name.startswith("card"):
            continue
        device_link = entry / "device"
        if not device_link.exists():
            continue
        try:
            real = device_link.resolve()
        except OSError:
            continue
        if real.name == pci_slot:
            return entry
    return None


def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_vram_used(card_path: Path) -> int | None:
    """Card-global VRAM usage in MiB.

    Only amdgpu exposes this; Intel's xe/i915 don't have a global counter, so
    on Arc this returns None and callers fall back to per-process fdinfo
    accounting (`_read_pid_vram_mb`).
    """
    v = _read_int(card_path / "device" / "mem_info_vram_used")
    return None if v is None else v // (1024 * 1024)


def _read_vram_total(card_path: Path) -> int | None:
    """Total VRAM in MiB, trying amdgpu, xe, and i915 sysfs layouts in turn."""
    device = card_path / "device"
    candidates = [
        device / "mem_info_vram_total",  # amdgpu
        device / "tile0" / "physical_vram_size_bytes",  # xe (Battlemage, DG2 on xe)
        device / "lmem_total_bytes",  # i915 dGPU
        card_path / "lmem_total_bytes",  # i915 (older layout)
    ]
    for c in candidates:
        v = _read_int(c)
        if v:
            return v // (1024 * 1024)
    return None


# DRM fdinfo memory-region keys, per kernel driver:
#   xe:     drm-resident-vram0 / drm-total-vram0
#   i915:   drm-resident-local0 / drm-total-local0
#   amdgpu: drm-memory-vram (legacy spelling of drm-total-vram)
_FDINFO_VRAM_RE = re.compile(
    r"^drm-(?P<kind>resident|total|memory)-(?:vram|local)\d*:\s*(?P<num>\d+)\s*(?P<unit>[KMG]iB)?",
)
_FDINFO_UNIT = {None: 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def _read_pid_vram_mb(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """VRAM held by one process in MiB, from DRM fdinfo.

    A process usually has several fds open on the same DRM client (device node
    dup'd by the driver stack); fdinfo repeats identical stats for each, so we
    de-duplicate by `drm-client-id` before summing. Resident is preferred over
    total when the driver reports both.
    """
    fdinfo_dir = proc_root / str(pid) / "fdinfo"
    try:
        entries = list(fdinfo_dir.iterdir())
    except OSError:
        return None
    per_client: dict[str, dict[str, int]] = {}
    anon = 0
    for entry in entries:
        try:
            text = entry.read_text()
        except OSError:
            continue
        if "drm-client-id" not in text and "drm-" not in text:
            continue
        client_id: str | None = None
        sums = {"resident": 0, "total": 0}
        for line in text.splitlines():
            if line.startswith("drm-client-id:"):
                client_id = line.split(":", 1)[1].strip()
                continue
            m = _FDINFO_VRAM_RE.match(line)
            if m:
                kind = "total" if m.group("kind") == "memory" else m.group("kind")
                sums[kind] += int(m.group("num")) * _FDINFO_UNIT[m.group("unit")]
        if sums["resident"] == 0 and sums["total"] == 0:
            continue
        if client_id is None:
            client_id = f"_anon{anon}"
            anon += 1
        per_client[client_id] = sums
    if not per_client:
        return None
    total_bytes = sum(
        s["resident"] if s["resident"] > 0 else s["total"] for s in per_client.values()
    )
    return total_bytes // (1024 * 1024)


# ------------------------------------------------------------------
# Prompt construction (approximate token counts)
# ------------------------------------------------------------------


def _build_prompt(target_tokens: int) -> str:
    """Build a prompt of roughly *target_tokens* tokens.

    We assume ~4 chars per token for English text — good enough for
    benchmarking; we don't need exact tokenisation.
    """
    word = WARMUP_PROMPT.strip()
    needed_chars = target_tokens * 4
    repeats = max(1, needed_chars // len(word))
    prompt = (word + " ") * repeats
    return prompt[:needed_chars]


# ------------------------------------------------------------------
# Core measurement
# ------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementSummary:
    """Robust summary of repeated measurements for one benchmark axis."""

    tok_s: float
    elapsed_ms: float
    token_count: int
    samples: list[float]

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def maximum(self) -> float:
        return max(self.samples)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _measured_tokens(obj: dict[str, Any], timing_key: str, usage_key: str, fallback: int) -> int:
    """Prefer llama-server's exact timing count, then OpenAI usage metadata."""
    timings = obj.get("timings") or {}
    usage = obj.get("usage") or {}
    return _positive_int(timings.get(timing_key)) or _positive_int(usage.get(usage_key)) or fallback


def _summarise_measurements(
    rates: list[float], elapsed_ms: list[float], token_counts: list[int]
) -> MeasurementSummary:
    """Return medians, avoiding the upward bias of best-of-N selection."""
    if not rates:
        return MeasurementSummary(0.0, 0.0, 0, [])
    return MeasurementSummary(
        tok_s=float(statistics.median(rates)),
        elapsed_ms=float(statistics.median(elapsed_ms)),
        token_count=int(statistics.median(token_counts)),
        samples=list(rates),
    )


async def _complete(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool = False,
    cache_prompt: bool = True,
    is_chat: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Send one non-streamed completion. Returns (wall_seconds, response_json).

    Non-streamed on purpose: llama-server reports an exact ``timings`` block
    (prompt/predicted tok/s measured inside the engine), which is far more
    reliable than timing SSE chunks from the client -- llama.cpp batches
    several tokens per chunk, so client-side inter-token timing overstates
    the rate. We keep the wall time only as a fallback for non-llama.cpp
    upstreams that omit ``timings``.

    ``cache_prompt=False`` forces a full re-prefill each call; prompt-eval
    needs it so repeated runs measure real prefill instead of a cache hit.

    ``is_chat`` selects the endpoint and request shape. Generation
    benchmarking uses the raw ``/v1/completions`` endpoint because
    ``ignore_eos`` past a chat template's EOS can produce output that
    ``/v1/chat/completions`` refuses to parse.
    """
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if is_chat:
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if ignore_eos:
        body["ignore_eos"] = True
    if not cache_prompt:
        body["cache_prompt"] = False
    t0 = time.perf_counter()
    resp = await client.post("/v1/chat/completions" if is_chat else "/v1/completions", json=body)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    return wall, resp.json()


async def _measure_prompt_eval(
    client: httpx.AsyncClient,
    model: str,
    prompt_tokens: int,
    repeats: int = REPEAT_PROMPT_EVAL,
) -> tuple[float, float]:
    """Return median (prompt-eval tok/s, elapsed_ms) across *repeats* runs.

    Prefers llama-server's ``timings.prompt_per_second``; falls back to
    ``prompt_tokens / wall`` (with max_tokens=1) for upstreams without it.
    """
    summary = await _measure_prompt_eval_summary(client, model, prompt_tokens, repeats)
    return summary.tok_s, summary.elapsed_ms


async def _measure_prompt_eval_summary(
    client: httpx.AsyncClient,
    model: str,
    prompt_tokens: int,
    repeats: int = REPEAT_PROMPT_EVAL,
) -> MeasurementSummary:
    prompt = _build_prompt(prompt_tokens)
    rates: list[float] = []
    elapsed: list[float] = []
    counts: list[int] = []
    for _ in range(repeats):
        # cache_prompt=False so every run does a real full prefill.
        wall, obj = await _complete(client, model, prompt, max_tokens=1, cache_prompt=False)
        t = obj.get("timings") or {}
        exact_tokens = _measured_tokens(obj, "prompt_n", "prompt_tokens", prompt_tokens)
        if t.get("prompt_per_second"):
            tok_s = float(t["prompt_per_second"])
            ms = float(t.get("prompt_ms", wall * 1000))
        else:
            tok_s = exact_tokens / wall if wall > 0 else 0.0
            ms = wall * 1000
        rates.append(tok_s)
        elapsed.append(ms)
        counts.append(exact_tokens)
    return _summarise_measurements(rates, elapsed, counts)


async def _measure_generation(
    client: httpx.AsyncClient,
    model: str,
    gen_tokens: int,
    repeats: int = REPEAT_GENERATION,
) -> tuple[float, float]:
    """Return median (generation tok/s, elapsed_ms) across *repeats* runs.

    ``ignore_eos`` forces the model to emit the full ``gen_tokens`` instead
    of stopping early on a short reply. Prefers llama-server's
    ``timings.predicted_per_second`` (pure decode rate, excludes prefill);
    falls back to ``completion_tokens / wall`` for upstreams without timings.
    """
    summary = await _measure_generation_summary(client, model, gen_tokens, repeats)
    return summary.tok_s, summary.elapsed_ms


async def _measure_generation_summary(
    client: httpx.AsyncClient,
    model: str,
    gen_tokens: int,
    repeats: int = REPEAT_GENERATION,
) -> MeasurementSummary:
    prompt = "Hello"
    rates: list[float] = []
    elapsed: list[float] = []
    counts: list[int] = []
    for _ in range(repeats):
        # Use the raw completions endpoint: ignore_eos past a chat template's
        # EOS can produce output that /v1/chat/completions refuses to parse.
        wall, obj = await _complete(
            client, model, prompt, max_tokens=gen_tokens, ignore_eos=True, is_chat=False
        )
        t = obj.get("timings") or {}
        exact_tokens = _measured_tokens(obj, "predicted_n", "completion_tokens", gen_tokens)
        if t.get("predicted_per_second"):
            tok_s = float(t["predicted_per_second"])
            ms = float(t.get("predicted_ms", wall * 1000))
        else:
            tok_s = exact_tokens / wall if wall > 0 else 0.0
            ms = wall * 1000
        rates.append(tok_s)
        elapsed.append(ms)
        counts.append(exact_tokens)
    return _summarise_measurements(rates, elapsed, counts)


async def _warmup(
    client: httpx.AsyncClient,
    model: str,
    prompt_tokens: int = 32,
    gen_tokens: int = 8,
) -> float:
    """Run a tiny completion to absorb SYCL JIT. Returns elapsed seconds."""
    t0 = time.perf_counter()
    prompt = _build_prompt(prompt_tokens)
    try:
        await _complete(client, model, prompt, max_tokens=gen_tokens)
    except Exception as e:
        log.warning("warmup failed: %s", e)
    return time.perf_counter() - t0


# ------------------------------------------------------------------
# Single-shot benchmark
# ------------------------------------------------------------------


async def benchmark_model(
    server_url: str,
    model_name: str,
    *,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    gen_tokens: int = DEFAULT_GEN_TOKENS,
    load: bool = True,
    cfg: Config | None = None,
) -> BenchmarkResult:
    """Run a full benchmark cycle for one model.

    Args:
        server_url: Base URL of arc-llama serve (e.g. http://127.0.0.1:11437)
        model_name: Registered model name
        prompt_tokens: Approximate prompt length to benchmark
        gen_tokens: Number of tokens to generate
        load: Whether to preload the model via /admin/load
        cfg: Optional pre-loaded Config (used to read GPU info for VRAM)
    """
    if cfg is None:
        cfg = load_config()

    model = cfg.find_model(model_name)
    if model is None:
        return BenchmarkResult(
            model=model_name,
            ctx=0,
            cache_type_k="?",
            cache_type_v="?",
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            error=f"Model '{model_name}' not found in config",
        )

    recipe = model.launch_recipe()
    result = BenchmarkResult(
        model=model_name,
        ctx=recipe.ctx,
        cache_type_k=recipe.cache_type_k.value,
        cache_type_v=recipe.cache_type_v.value,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
    )

    gpu = cfg.find_gpu(model.gpu_pci_slot)
    card_path = _find_drm_card(model.gpu_pci_slot) if gpu else None
    vram_before = _read_vram_used(card_path) if card_path else None
    vram_total = _read_vram_total(card_path) if card_path else None
    if vram_total is None and gpu is not None:
        vram_total = gpu.vram_mb  # detected at init time; good enough for %
    result.vram_total_mb = vram_total

    headers = (
        {"Authorization": f"Bearer {cfg.server.admin_token}"} if cfg.server.admin_token else {}
    )
    async with httpx.AsyncClient(base_url=server_url, timeout=300.0, headers=headers) as client:
        # Ensure model is loaded
        if load:
            log.info("loading %s ...", model_name)
            r = await client.post(f"/admin/load/{model_name}")
            if r.status_code != 200:
                result.error = f"Failed to load model: {r.status_code} {r.text}"
                return result

        # Warm-up (SYCL JIT)
        log.info("warming up %s ...", model_name)
        result.jit_warmup_s = await _warmup(client, model_name)

        # Prompt eval
        log.info("benchmarking prompt-eval (%d tokens) ...", prompt_tokens)
        prompt_summary = await _measure_prompt_eval_summary(client, model_name, prompt_tokens)
        result.prompt_eval_tok_s = prompt_summary.tok_s
        result.prompt_eval_ms = prompt_summary.elapsed_ms
        result.measured_prompt_tokens = prompt_summary.token_count
        result.prompt_eval_samples = prompt_summary.samples
        result.prompt_eval_min_tok_s = prompt_summary.minimum
        result.prompt_eval_max_tok_s = prompt_summary.maximum

        # Generation
        log.info("benchmarking generation (%d tokens) ...", gen_tokens)
        generation_summary = await _measure_generation_summary(client, model_name, gen_tokens)
        result.generation_tok_s = generation_summary.tok_s
        result.generation_ms = generation_summary.elapsed_ms
        result.measured_gen_tokens = generation_summary.token_count
        result.generation_samples = generation_summary.samples
        result.generation_min_tok_s = generation_summary.minimum
        result.generation_max_tok_s = generation_summary.maximum

        # VRAM after. amdgpu has a card-global counter; on Intel (xe/i915)
        # there is none, so fall back to the backend process's DRM fdinfo.
        vram_after = _read_vram_used(card_path) if card_path else None
        if vram_before is not None and vram_after is not None:
            result.vram_used_mb = vram_after - vram_before
        elif vram_after is not None:
            result.vram_used_mb = vram_after
        else:
            pid = await _backend_pid(client, model.name)
            if pid is not None:
                result.vram_used_mb = _read_pid_vram_mb(pid)

    return result


async def _backend_pid(client: httpx.AsyncClient, model_name: str) -> int | None:
    """Ask the running arc-llama serve for the llama-server pid of a model.

    Purely best-effort (feeds the VRAM readout, never the measurements), so
    any failure — including a client without /admin/status — returns None.
    """
    try:
        r = await client.get("/admin/status")
        if r.status_code != 200:
            return None
        for m in r.json().get("models", []):
            if m.get("name") == model_name:
                pid = m.get("pid")
                return int(pid) if pid else None
    except Exception:
        return None
    return None


# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------


async def benchmark_sweep(
    server_url: str,
    model_name: str,
    *,
    ctx_values: list[int],
    kv_types: list[str],
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    gen_tokens: int = DEFAULT_GEN_TOKENS,
    cfg: Config | None = None,
) -> list[BenchmarkResult]:
    """Run benchmark across multiple ctx + KV configurations.

    Each config is applied via /admin/models/{name}/edit, benchmarked,
    then reverted to the original if desired.
    """
    if cfg is None:
        cfg = load_config()

    model = cfg.find_model(model_name)
    if model is None:
        return [
            BenchmarkResult(
                model=model_name,
                ctx=0,
                cache_type_k="?",
                cache_type_v="?",
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                error=f"Model '{model_name}' not found in config",
            )
        ]

    original_recipe = dict(model.recipe or {})
    results: list[BenchmarkResult] = []

    headers = (
        {"Authorization": f"Bearer {cfg.server.admin_token}"} if cfg.server.admin_token else {}
    )
    async with httpx.AsyncClient(base_url=server_url, timeout=300.0, headers=headers) as client:
        for ctx in ctx_values:
            for kv in kv_types:
                # Apply config
                edit_body = {
                    "ctx": ctx,
                    "cache_type_k": kv,
                    "cache_type_v": kv,
                }
                r = await client.post(f"/admin/models/{model_name}/edit", json=edit_body)
                if r.status_code != 200:
                    results.append(
                        BenchmarkResult(
                            model=model_name,
                            ctx=ctx,
                            cache_type_k=kv,
                            cache_type_v=kv,
                            prompt_tokens=prompt_tokens,
                            gen_tokens=gen_tokens,
                            error=f"Edit failed: {r.status_code} {r.text}",
                        )
                    )
                    continue

                # Benchmark
                res = await benchmark_model(
                    server_url,
                    model_name,
                    prompt_tokens=prompt_tokens,
                    gen_tokens=gen_tokens,
                    load=True,
                    cfg=cfg,
                )
                # Override result fields to reflect the sweep config, not the original recipe.
                res.ctx = ctx
                res.cache_type_k = kv
                res.cache_type_v = kv
                results.append(res)

        # Restore original recipe
        restore_body = {
            k: v
            for k, v in original_recipe.items()
            if k in ("ctx", "cache_type_k", "cache_type_v", "parallel", "n_gpu_layers")
        }
        if restore_body:
            await client.post(f"/admin/models/{model_name}/edit", json=restore_body)

    return results


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------


def _fmt_speed(tok_s: float | None) -> str:
    if tok_s is None:
        return "—"
    return f"{tok_s:>6.1f} tok/s"


def _fmt_time(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:>5.0f} ms"
    return f"{ms / 1000:>5.2f} s"


def _fmt_vram(used_mb: int | None, total_mb: int | None) -> str:
    if used_mb is None or total_mb is None:
        return "—"
    pct = round(used_mb / total_mb * 100, 1)
    return f"{used_mb / 1024:.1f} GB / {total_mb / 1024:.1f} GB  ({pct}%)"


def print_result(result: BenchmarkResult) -> None:
    """Pretty-print a single BenchmarkResult to the console."""
    from rich.console import Console

    console = Console()
    if result.error:
        console.print(f"\n[red]Benchmark failed for {result.model}: {result.error}[/red]")
        return

    console.print(f"\n[bold]Benchmark: {result.model}[/bold]")
    console.print(f"  Recipe:   ctx={result.ctx}, KV={result.cache_type_k}/{result.cache_type_v}")
    prompt_count = result.measured_prompt_tokens or result.prompt_tokens
    gen_count = result.measured_gen_tokens or result.gen_tokens
    console.print(f"  Prompt:   {prompt_count} measured tokens")
    console.print(f"  Generate: {gen_count} measured tokens")
    if result.jit_warmup_s is not None:
        console.print(f"  Warm-up:  {result.jit_warmup_s:.1f}s (SYCL JIT)")
    console.print(
        f"  Prompt-eval:  {_fmt_speed(result.prompt_eval_tok_s)}  |  {_fmt_time(result.prompt_eval_ms)}"
    )
    console.print(
        f"  Generation:   {_fmt_speed(result.generation_tok_s)}  |  {_fmt_time(result.generation_ms)}"
    )
    if result.prompt_eval_samples:
        console.print(
            f"  Prompt range: {_fmt_speed(result.prompt_eval_min_tok_s)} .. "
            f"{_fmt_speed(result.prompt_eval_max_tok_s)} "
            f"({len(result.prompt_eval_samples)} runs)"
        )
    if result.generation_samples:
        console.print(
            f"  Gen range:    {_fmt_speed(result.generation_min_tok_s)} .. "
            f"{_fmt_speed(result.generation_max_tok_s)} "
            f"({len(result.generation_samples)} runs)"
        )
    console.print(f"  VRAM:         {_fmt_vram(result.vram_used_mb, result.vram_total_mb)}")


def print_sweep_table(results: list[BenchmarkResult]) -> None:
    """Print a markdown-style table for a sweep run."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Benchmark Sweep")
    table.add_column("ctx")
    table.add_column("KV")
    table.add_column("Prompt-eval")
    table.add_column("Generation")
    table.add_column("VRAM")
    table.add_column("Fit")

    for r in results:
        if r.error:
            table.add_row(
                str(r.ctx),
                f"{r.cache_type_k}/{r.cache_type_v}",
                "[red]error[/red]",
                "",
                "",
                "✗",
            )
            continue
        fit = "✓" if (r.vram_pct is None or r.vram_pct < 95) else "⚠"
        table.add_row(
            str(r.ctx),
            f"{r.cache_type_k}/{r.cache_type_v}",
            _fmt_speed(r.prompt_eval_tok_s),
            _fmt_speed(r.generation_tok_s),
            _fmt_vram(r.vram_used_mb, r.vram_total_mb),
            fit,
        )
    console.print(table)
