"""Benchmark harness for arc-llama models.

Measures prompt-eval throughput, generation throughput, and VRAM usage
for a registered model. Can run a single shot or sweep ctx/KV configs.

All measurements go through the running arc-llama serve instance so the
benchmark inherits the correct SYCL env, arch profile, and router policy.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from arc_llama.config import Config, load_config

log = logging.getLogger("arc_llama.benchmark")

WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog. "
# We repeat this to build arbitrary-length prompts without tokenisation.

DEFAULT_PROMPT_TOKENS = 512
DEFAULT_GEN_TOKENS = 128

# How many repeat runs to average for stable numbers.
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
        device / "mem_info_vram_total",              # amdgpu
        device / "tile0" / "physical_vram_size_bytes",  # xe (Battlemage, DG2 on xe)
        device / "lmem_total_bytes",                 # i915 dGPU
        card_path / "lmem_total_bytes",              # i915 (older layout)
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

async def _measure_once(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, float]:
    """Send one completion and return (time_to_first_token_s, total_time_s).

    For generation measurement we use stream=True and time the arrival
    of the first chunk (TFT) and the final chunk (total).
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    first_chunk_at: float | None = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        async for _chunk in resp.aiter_text():
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
    t1 = time.perf_counter()
    tft = first_chunk_at if first_chunk_at is not None else t1
    return tft - t0, t1 - t0


async def _measure_prompt_eval(
    client: httpx.AsyncClient,
    model: str,
    prompt_tokens: int,
    repeats: int = REPEAT_PROMPT_EVAL,
) -> tuple[float, float]:
    """Return (tok/s, elapsed_ms) averaged over *repeats* runs.

    We use max_tokens=1 so almost all time is prompt eval.
    """
    prompt = _build_prompt(prompt_tokens)
    times: list[float] = []
    for _ in range(repeats):
        _, total = await _measure_once(client, model, prompt, max_tokens=1)
        times.append(total)
    best = min(times)  # use best-of-N to discard scheduler jitter
    tok_s = prompt_tokens / best if best > 0 else 0.0
    return tok_s, best * 1000


async def _measure_generation(
    client: httpx.AsyncClient,
    model: str,
    gen_tokens: int,
    repeats: int = REPEAT_GENERATION,
) -> tuple[float, float]:
    """Return (tok/s, elapsed_ms) averaged over *repeats* runs.

    We use a very short prompt so generation dominates.
    """
    prompt = "Hello"
    times: list[float] = []
    for _ in range(repeats):
        _, total = await _measure_once(client, model, prompt, max_tokens=gen_tokens)
        times.append(total)
    best = min(times)
    tok_s = gen_tokens / best if best > 0 else 0.0
    return tok_s, best * 1000


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
        await _measure_once(client, model, prompt, max_tokens=gen_tokens)
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
            model=model_name, ctx=0, cache_type_k="?", cache_type_v="?",
            prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
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

    async with httpx.AsyncClient(base_url=server_url, timeout=300.0) as client:
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
        result.prompt_eval_tok_s, result.prompt_eval_ms = await _measure_prompt_eval(
            client, model_name, prompt_tokens
        )

        # Generation
        log.info("benchmarking generation (%d tokens) ...", gen_tokens)
        result.generation_tok_s, result.generation_ms = await _measure_generation(
            client, model_name, gen_tokens
        )

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
        return [BenchmarkResult(
            model=model_name, ctx=0, cache_type_k="?", cache_type_v="?",
            prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
            error=f"Model '{model_name}' not found in config",
        )]

    original_recipe = dict(model.recipe or {})
    results: list[BenchmarkResult] = []

    async with httpx.AsyncClient(base_url=server_url, timeout=300.0) as client:
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
                    results.append(BenchmarkResult(
                        model=model_name, ctx=ctx, cache_type_k=kv, cache_type_v=kv,
                        prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                        error=f"Edit failed: {r.status_code} {r.text}",
                    ))
                    continue

                # Benchmark
                res = await benchmark_model(
                    server_url, model_name,
                    prompt_tokens=prompt_tokens, gen_tokens=gen_tokens,
                    load=True, cfg=cfg,
                )
                # Override result fields to reflect the sweep config, not the original recipe.
                res.ctx = ctx
                res.cache_type_k = kv
                res.cache_type_v = kv
                results.append(res)

        # Restore original recipe
        restore_body = {
            k: v for k, v in original_recipe.items()
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
    console.print(f"  Prompt:   {result.prompt_tokens} tokens")
    console.print(f"  Generate: {result.gen_tokens} tokens")
    if result.jit_warmup_s is not None:
        console.print(f"  Warm-up:  {result.jit_warmup_s:.1f}s (SYCL JIT)")
    console.print(f"  Prompt-eval:  {_fmt_speed(result.prompt_eval_tok_s)}  |  {_fmt_time(result.prompt_eval_ms)}")
    console.print(f"  Generation:   {_fmt_speed(result.generation_tok_s)}  |  {_fmt_time(result.generation_ms)}")
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
                str(r.ctx), f"{r.cache_type_k}/{r.cache_type_v}",
                "[red]error[/red]", "", "", "✗",
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
