"""Model swap policy.

The router owns the lifecycle of every llama-server subprocess and decides which
one is currently allowed to hold its GPU's VRAM. Two policies are supported:

  * **single_resident** (default): only one model is loaded across *all* GPUs
    at any time — switching models stops the previous one before starting the
    next. This matches conservative thermal/power use.

  * **multi_resident**: models on *different* GPUs can coexist; only models on
    the *same* GPU contend. Models still get loaded on demand and stay up for
    follow-up requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from arc_llama.config import (
    AudioModelConfig,
    Config,
    GPUConfig,
    ModelConfig,
)
from arc_llama.gguf_meta import (
    estimate_weight_vram_bytes,
    override_tensor_saved_bytes,
    scan_weight_tensors,
    weight_tensor_table,
)
from arc_llama.launcher import LlamaServer, build_audio_plan, build_plan
from arc_llama.recipes import KVCacheType, estimate_kv_bytes
from arc_llama.tts import get_engine as get_tts_engine

log = logging.getLogger("arc_llama.router")

# Rough overhead budgets for VRAM estimation (MiB).
_VRAM_COMPUTE_BUFFER_MB = 768
_VRAM_SAFETY_MARGIN_MB = 256


def _estimate_model_vram_mb(
    model: ModelConfig,
    *,
    ctx: int | None = None,
    kv_type: KVCacheType | None = None,
    n_cpu_moe: int | None = None,
    override_tensor: list[str] | None = None,
    compute_buffer_mb: int | None = None,
) -> int | None:
    """Rough VRAM footprint for one model instance, or None when it cannot
    be estimated.

    Uses GGUF tensor metadata to estimate the decompressed weight footprint,
    which is much closer to reality for heavily quantized files than the raw
    file size. Falls back to file size if the GGUF cannot be read.

    ``ctx`` / ``kv_type`` / ``n_cpu_moe`` / ``override_tensor`` override the
    recipe's values, letting callers ask "would this model fit at context N
    with KV type T and this offload?" — the tuner uses this to prune KV
    candidates that cannot hold the declared workload context and to find
    the minimum feasible expert offload.

    The ``n_cpu_moe`` accounting subtracts the routed-expert tensor bytes of
    the first N layers — exactly what ``--n-cpu-moe N`` keeps on the host —
    so a model that only fits *with* expert offload is no longer refused.
    ``override_tensor`` does the same for the regex patterns it matches.
    When offload is in force but the expert tensor bytes cannot be
    determined, the estimate is None and callers must skip the fit guard
    rather than fall back to counting full weights: that fallback is the bug
    that made offload-configured models unloadable. A wrongly-permitted load
    fails loudly at llama-server startup with a real OOM; a wrongly-refused
    one silently disables the feature.
    """
    path = Path(model.path)
    recipe = model.recipe or {}
    weight_bytes: int | None = None
    # -ot and --n-cpu-moe are alternatives, never both: when patterns are in
    # force the layer count stays 0 so the n_cpu_moe branch below is skipped.
    eff_moe = 0
    eff_ot = override_tensor if override_tensor is not None else recipe.get("override_tensor")
    if eff_ot:
        table = weight_tensor_table(path)
        if table is None:
            log.warning(
                "VRAM estimate for %s unavailable: cannot read tensor table "
                "for override_tensor; skipping the fit guard",
                model.name,
            )
            return None
        try:
            weight_bytes = estimate_weight_vram_bytes(path)
            if weight_bytes is None:
                weight_bytes = path.stat().st_size
            weight_bytes -= override_tensor_saved_bytes(table, eff_ot)
        except ValueError as exc:
            log.warning("VRAM estimate for %s unavailable: %s", model.name, exc)
            return None
    elif n_cpu_moe is not None:
        eff_moe = n_cpu_moe
    else:
        eff_moe = int(recipe.get("n_cpu_moe") or 0)
    if eff_moe > 0:
        weight_bytes = estimate_weight_vram_bytes(path, n_cpu_moe=eff_moe)
        if weight_bytes is None:
            log.warning(
                "VRAM estimate for %s unavailable: expert tensor bytes for "
                "--n-cpu-moe %d could not be determined; skipping the fit "
                "guard rather than counting full weights",
                model.name,
                eff_moe,
            )
            return None
    if weight_bytes is None:
        weight_bytes = estimate_weight_vram_bytes(path)
        if weight_bytes is None:
            try:
                weight_bytes = path.stat().st_size
            except OSError:
                weight_bytes = 0
            log.debug(
                "VRAM estimate for %s falling back to file size: %.0f MiB",
                model.name,
                weight_bytes / (1_048_576),
            )
    weight_mb = weight_bytes // (1_048_576)
    eff_ctx = ctx if ctx is not None else int(recipe.get("ctx", 8192))
    eff_kv = kv_type if kv_type is not None else KVCacheType(recipe.get("cache_type_k", "f16"))
    kv_mb = estimate_kv_bytes(eff_ctx, eff_kv, model.kv_class) // (1_048_576)
    buffer_mb = compute_buffer_mb if compute_buffer_mb is not None else _VRAM_COMPUTE_BUFFER_MB
    return weight_mb + kv_mb + buffer_mb + _VRAM_SAFETY_MARGIN_MB


def _estimate_audio_vram_mb(model: AudioModelConfig) -> int | None:
    """Rough VRAM footprint for one audio model instance.

    A declared ``vram_mb`` always wins: the user watching `arc-llama gpus`
    knows the real number better than we can derive it.

    For a transcription model the estimate is weights + projector + a KV cache
    sized to the recipe's ctx, because that KV is the dominant term and the
    reason an unconfigured ASR model can occupy more VRAM than the 27B it
    sits next to. A TTS model is measured by its engine, which knows where its
    weights actually are — OmniVoice's usually live in the Hugging Face cache
    rather than at ``model.path``.

    Returns None when the path cannot be measured, which makes the caller
    skip this model rather than guess; the fit guard treats an unmeasurable
    co-resident the same way it treats an unmeasurable offloaded LLM.
    """
    if model.vram_mb:
        return int(model.vram_mb)
    if model.task == "tts":
        engine = get_tts_engine(model.engine)
        return engine.estimate_vram_mb(model) if engine is not None else None
    path = Path(model.path).expanduser()
    recipe = model.audio_recipe()
    try:
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            return None
    except OSError:
        return None
    if not size:
        return None
    total_mb = size // (1_048_576)
    if recipe.mmproj:
        try:
            total_mb += Path(recipe.mmproj).expanduser().stat().st_size // (1_048_576)
        except OSError:
            pass
    try:
        kv_type = KVCacheType(recipe.cache_type_k)
    except ValueError:
        kv_type = KVCacheType.F16
    total_mb += estimate_kv_bytes(recipe.ctx, kv_type, "default") // (1_048_576)
    return total_mb + _VRAM_COMPUTE_BUFFER_MB + _VRAM_SAFETY_MARGIN_MB


def min_moe_offload_layers(
    model: ModelConfig,
    vram_mb: int | None,
    *,
    ctx: int | None = None,
    kv_type: KVCacheType | None = None,
) -> int | None:
    """Smallest ``--n-cpu-moe`` layer count at which *model* is estimated to fit.

    Returns 0 when the model fits with no offload, the minimal feasible layer
    count otherwise, and the MoE layer count when not even full offload fits
    (the best llama.cpp can do — the fit guard remains the arbiter). Returns
    None when the VRAM budget is unknown or the expert tensor bytes cannot be
    determined, in which case no offload math is possible.

    Costs one GGUF scan: the per-layer expert bytes are read once and every
    candidate N after that is pure arithmetic, using the same weight/KV/
    buffer accounting as ``_estimate_model_vram_mb`` so the registration-time
    suggestion, the load-time guard, and the tuner all agree.
    """
    if not vram_mb:
        return None
    scan = scan_weight_tensors(model.path)
    if scan is None:
        return None
    total_bytes, expert_by_layer = scan
    if not expert_by_layer:
        return None
    recipe = model.recipe or {}
    eff_ctx = ctx if ctx is not None else int(recipe.get("ctx", 8192))
    eff_kv = kv_type if kv_type is not None else KVCacheType(recipe.get("cache_type_k", "f16"))
    kv_mb = estimate_kv_bytes(eff_ctx, eff_kv, model.kv_class) // (1_048_576)
    fixed_mb = kv_mb + _VRAM_COMPUTE_BUFFER_MB + _VRAM_SAFETY_MARGIN_MB
    n_layers = max(expert_by_layer) + 1
    # Saved bytes grow monotonically with N, so a linear scan from 0 finds
    # the minimum; MoE layer counts are at most ~100 and each step here is
    # arithmetic only (no re-reads).
    saved_bytes = 0
    for n in range(0, n_layers + 1):
        weight_mb = (total_bytes - saved_bytes) // (1_048_576)
        if weight_mb + fixed_mb <= vram_mb:
            return n
        saved_bytes += expert_by_layer.get(n, 0)
    return n_layers


def _startup_failure_hint(log_tail: str) -> str:
    """Translate a known backend startup failure into the fix for it.

    The backend's own message is accurate but assumes context the reader does
    not have. A TTS sidecar that dies on `ModuleNotFoundError: omnivoice` is
    reporting a perfectly ordinary fact about the interpreter it was started
    with — but the user never chose that interpreter explicitly, so the fix
    (point `tts_python` at the environment that has OmniVoice) is not visible
    from the traceback.
    """
    if "No module named 'omnivoice'" in log_tail:
        return (
            "The interpreter running the TTS backend cannot import `omnivoice`. "
            "arc-llama does not depend on torch, so OmniVoice lives in its own "
            "virtualenv: point at it with `arc-llama audio set-python "
            "/path/to/OmniVoice/.venv/bin/python`."
        )
    if "No module named 'torch'" in log_tail:
        return (
            "The TTS interpreter has no torch. Install OmniVoice and a torch "
            "build for your GPU into that environment (XPU for Arc), then "
            "re-run `arc-llama audio set-python`."
        )
    if "AssertionError: Torch not compiled with XPU" in log_tail or (
        "Torch not compiled with XPU enabled" in log_tail
    ):
        return (
            "The TTS interpreter's torch has no XPU support, so it cannot use "
            "the Arc card. Install an XPU torch build, or set "
            "`recipe.device = \"cpu\"` on the model to run it on the CPU."
        )
    return ""


class Router:
    """Owns one LlamaServer per registered model and serialises swaps."""

    def __init__(self, cfg: Config, log_dir: Path | None = None):
        self.cfg = cfg
        self.log_dir = log_dir
        self._servers: dict[str, LlamaServer] = {}  # keyed by model.name
        self._lock = asyncio.Lock()
        self._loading_futures: dict[
            str, asyncio.Future[tuple[ModelConfig | AudioModelConfig, LlamaServer]]
        ] = {}
        self.metrics: dict[str, Any] = {
            "loads": 0,
            "stops": 0,
            "load_errors": 0,
            "last_load_at": None,
            "last_error": None,
        }
        self.last_activity: float = time.time()
        # Requests holding the GPU right now. Owned by server.py's _proxy_post:
        # incremented on request entry, decremented only when the forwarded
        # response (streaming included) has been fully produced.
        self.inflight: int = 0
        # Same window, attributed per model once the request has resolved one.
        # The global counter cannot answer "is THIS model still serving?": the
        # evicting request itself holds it above zero, so waiting on it before
        # an eviction would deadlock. Keyed by name so it survives rebuilds.
        self.model_inflight: dict[str, int] = {}
        # Models whose llama-server is being torn down right now. Set
        # synchronously before astop() begins, while the deciding read of
        # model_inflight is still in the same event-loop segment, so the
        # lock-free fast path in ensure_active can never hand out a server
        # that is already on its way down.
        self._stopping: set[str] = set()
        # Why an audio model has no backend, keyed by name. Populated by
        # _build_servers and surfaced through /admin/status.
        self.audio_launch_errors: dict[str, str] = {}
        self._build_servers()

    def acquire_model(self, name: str) -> None:
        """Count a request as actively using *name*. Called by _proxy_post
        once the request has resolved to a local model."""
        self.model_inflight[name] = self.model_inflight.get(name, 0) + 1

    def release_model(self, name: str) -> None:
        current = self.model_inflight.get(name, 0)
        if current <= 1:
            self.model_inflight.pop(name, None)
            if current < 1:
                log.warning("release_model(%s) with no matching acquire", name)
        else:
            self.model_inflight[name] = current - 1

    def _build_servers(self) -> None:
        """(Re)build the per-model backend registry from cfg.

        Idempotent — existing servers (running or not) are preserved by name,
        only new model entries get fresh LlamaServer instances. Use after a
        runtime config mutation (e.g. an admin scan).

        Both registries land in the same ``_servers`` map: an audio backend
        differs only in the argv its plan carries, so every lifecycle path
        below (swap, drain, stop, shutdown) treats them alike.
        """
        for m in self.cfg.models:
            if m.name in self._servers:
                continue
            gpu = self.cfg.find_gpu(m.gpu_pci_slot)
            if gpu is None:
                log.warning(
                    "model %s references unknown GPU %s; skipping",
                    m.name,
                    m.gpu_pci_slot,
                )
                continue
            plan = build_plan(self.cfg, m, gpu, host=self.cfg.server.host)
            self._servers[m.name] = LlamaServer(plan, name=m.name)
        for am in self.cfg.audio_models:
            if am.name in self._servers:
                continue
            gpu = self.cfg.find_gpu(am.gpu_pci_slot)
            if gpu is None:
                log.warning(
                    "audio model %s references unknown GPU %s; skipping",
                    am.name,
                    am.gpu_pci_slot,
                )
                continue
            try:
                plan = build_audio_plan(self.cfg, am, gpu, host=self.cfg.server.host)
            except RuntimeError as exc:
                # Missing binary, missing projector, a build without mtmd.
                # Registering nothing keeps the rest of the router working,
                # but the reason has to survive: "not launchable" with no
                # explanation is the least useful thing a UI can say.
                log.warning("audio model %s is not launchable: %s", am.name, exc)
                self.audio_launch_errors[am.name] = str(exc)
                continue
            self.audio_launch_errors.pop(am.name, None)
            self._servers[am.name] = LlamaServer(plan, name=am.name)

    def _entry_for(self, name: str) -> ModelConfig | AudioModelConfig | None:
        """The registry entry backing *name*, from either table."""
        for m in self.cfg.models:
            if m.name == name:
                return m
        for am in self.cfg.audio_models:
            if am.name == name:
                return am
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(
        self, query: str
    ) -> tuple[ModelConfig | AudioModelConfig, GPUConfig, LlamaServer] | None:
        m = self.cfg.find_any_model(query)
        if m is None:
            return None
        gpu = self.cfg.find_gpu(m.gpu_pci_slot)
        if gpu is None:
            return None
        srv = self._servers.get(m.name)
        if srv is None:
            return None
        return m, gpu, srv

    def all_models(self) -> list[ModelConfig]:
        return list(self.cfg.models)

    def all_audio_models(self) -> list[AudioModelConfig]:
        return list(self.cfg.audio_models)

    def running_models(self) -> list[str]:
        """Names of models whose llama-server process is alive (snapshot).

        Safe to call without the swap lock: the comprehension contains no
        await and the event loop is single-threaded, so _servers cannot
        change shape while it runs, and is_running is a live subprocess
        probe rather than a cached flag. Anything that changes immediately
        after the return is a policy question for the caller (drain/abort
        hooks), not a stale read.
        """
        return [n for n, s in self._servers.items() if s is not None and s.is_running]

    def backend_url_for(self, model_name: str) -> str | None:
        srv = self._servers.get(model_name)
        return srv.plan.backend_url if srv else None

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    async def ensure_active(
        self, query: str, *, acquire: bool = False
    ) -> tuple[ModelConfig | AudioModelConfig, LlamaServer]:
        """Make sure the requested model is the resident one (per policy) and
        return its (config, LlamaServer). Caller forwards the request to
        `srv.plan.backend_url`.

        With ``acquire=True`` the returned model is also counted in
        ``model_inflight`` — atomically with the readiness check, because the
        check and the counter bump happen in one synchronous event-loop
        segment. Callers that forward a real request must pass acquire=True
        and later call ``release_model``; without the atomic bump, an
        eviction drain could read a zero count for a model whose request was
        resolved but not yet counted, and stop it out from under the forward.

        Fast-path: if the model is already running *and ready* (its cached
        health state, set by wait_ready — no per-request probing), not being
        torn down, and no eviction is needed, return immediately without
        acquiring the swap lock.

        A request that finds the subprocess alive but not yet ready (a cold
        start takes tens of seconds to bind the port) waits on the shared load
        future instead of forwarding into a closed port, and concurrent
        requests for the same loading model all wait on that one future rather
        than each trying to start a new process.

        This method deliberately does NOT touch ``self.inflight``: the counter
        is owned by the request lifecycle in server.py (`_proxy_post`), which
        increments it on entry and decrements only after the forwarded response
        has been fully produced. Counting here would cover just the (often
        millisecond) swap decision and leave generation — the part that
        actually holds the GPU — invisible to the auto-tuner's abort hook.
        """
        self.last_activity = time.time()
        # Fast-path: already running AND ready → no lock, no eviction, no start.
        # is_running alone is not sufficient: the subprocess is alive for the
        # whole cold start, but the port only accepts connections once
        # wait_ready has passed, which is what `ready` caches. The _stopping
        # check closes the remaining window: an evictor marks the model before
        # the first await of its teardown, so a server on its way down is
        # never handed out from here.
        fast = self.resolve(query)
        if fast is not None:
            target_model, target_gpu, target_srv = fast
            if target_srv.is_running:
                if target_srv.ready and target_model.name not in self._stopping:
                    # Verify policy: if single-resident, we are the one; if multi,
                    # same-GPU contention would have been resolved when we started.
                    if acquire:
                        self.acquire_model(target_model.name)
                    return target_model, target_srv
                # Alive but not healthy yet: a load is in progress. Wait on the
                # starter's shared future (bounded by the starter's wait_ready
                # budget) instead of forwarding into a port that is not
                # listening. Shielded so a cancelled waiter cannot poison the
                # future the starter and other waiters still rely on.
                loading = self._loading_futures.get(target_model.name)
                if loading is not None:
                    loaded_model, loaded_srv = await asyncio.shield(loading)
                    # Re-validate after the await: the model may have been
                    # evicted between the load completing and this waiter
                    # being scheduled. The re-check plus the acquire is again
                    # one synchronous segment, so the result cannot race a
                    # drain's counter read.
                    if (
                        loaded_srv.is_running
                        and loaded_srv.ready
                        and loaded_model.name not in self._stopping
                    ):
                        if acquire:
                            self.acquire_model(loaded_model.name)
                        return loaded_model, loaded_srv
                    # Stale — fall through to the slow path and re-resolve.
                # Running-but-not-ready with no load we can join (should not
                # happen — every start registers a future before spawning).
                # Fall through to the slow path and let it re-wait or restart.

        # Slow path: may need to swap / start. Serialize with the lock.
        async with self._lock:
            # Another task may have finished loading while we waited.
            resolved = self.resolve(query)
            if resolved is None:
                raise KeyError(f"Unknown model: {query!r}")
            target_model, target_gpu, target_srv = resolved

            # If someone else is already loading this model, wait on them.
            existing_future = self._loading_futures.get(target_model.name)
            if existing_future is not None:
                loaded_model, loaded_srv = await asyncio.shield(existing_future)
                if loaded_srv.is_running and loaded_srv.ready:
                    if acquire:
                        self.acquire_model(loaded_model.name)
                    return loaded_model, loaded_srv

            await self._evict_for(target_model, target_gpu)

            if (
                target_srv.is_running
                and target_srv.ready
                and target_model.name not in self._stopping
            ):
                if acquire:
                    self.acquire_model(target_model.name)
                return target_model, target_srv

            # estimate_weight_vram_bytes reads GGUF metadata synchronously;
            # run the whole fit check in a thread so a multi-second disk read
            # cannot stall the event loop (and /admin/tune/status with it).
            await asyncio.to_thread(self._check_vram_fit, target_model, target_gpu)

            # We are the one responsible for starting.
            log.info("loading model %s on GPU %s ...", target_model.name, target_gpu.pci_slot)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[tuple[ModelConfig | AudioModelConfig, LlamaServer]] = (
                loop.create_future()
            )
            self._loading_futures[target_model.name] = future
            try:
                target_srv.start(log_dir=self.log_dir)
                ready = await target_srv.wait_ready()
                if not ready:
                    tail = target_srv.tail_log(lines=40)
                    hint = _startup_failure_hint(tail)
                    log.error(
                        "model %s failed health-check; stopping it",
                        target_model.name,
                    )
                    target_srv.stop()
                    self.metrics["last_error"] = f"{target_model.name} did not become healthy"
                    detail = f"backend for {target_model.name} did not become healthy"
                    if hint:
                        detail += f"\n\n{hint}"
                    if tail:
                        detail += "\n\n--- last log lines ---\n" + tail
                    raise RuntimeError(detail)
                self.metrics["loads"] += 1
                self.metrics["last_load_at"] = time.time()
                self.metrics["last_error"] = None
                result = (target_model, target_srv)
                future.set_result(result)
                if acquire:
                    self.acquire_model(target_model.name)
                return result
            except Exception as exc:
                if not future.done():
                    self.metrics["load_errors"] += 1
                    self.metrics["last_error"] = str(exc)
                    # Give waiters the same detailed error the starter raises
                    # (including the llama-server log tail), so _proxy_post can
                    # surface a 503 with real diagnostics rather than a bare
                    # "did not become healthy".
                    future.set_exception(exc)
                raise
            finally:
                self._loading_futures.pop(target_model.name, None)

    def _check_vram_fit(
        self, target: ModelConfig | AudioModelConfig, target_gpu: GPUConfig
    ) -> None:
        """Refuse to load *target* if its estimated VRAM won't fit on target_gpu.

        In multi-resident mode this also accounts for other loaded models that
        share the same GPU. Pinned audio models survive eviction, so they are
        counted here whatever the swap policy says: they are the co-residents
        an LLM load actually has to fit alongside.
        """
        if not target_gpu.vram_mb:
            return
        target_mb = (
            _estimate_audio_vram_mb(target)
            if isinstance(target, AudioModelConfig)
            else _estimate_model_vram_mb(target)
        )
        if target_mb is None:
            # Expert offload is in force but its bytes cannot be accounted.
            # Refusing here would silently disable expert offload (the model
            # fits precisely *because* of it); permit instead and let
            # llama-server's own OOM be the loud failure if we're wrong.
            log.warning(
                "skipping VRAM fit guard for %s: footprint with expert "
                "offload could not be estimated",
                target.name,
            )
            return
        used_mb = target_mb
        breakdown: list[str] = []
        for name, srv in self._servers.items():
            if name == target.name or not srv.is_running:
                continue
            other = self._entry_for(name)
            if other is None or other.gpu_pci_slot != target_gpu.pci_slot:
                continue
            # _evict_for has already run, so anything still alive here is a
            # real co-resident: an unevictable pinned model, or (in
            # multi-resident mode) an LLM sharing the card.
            other_mb = (
                _estimate_audio_vram_mb(other)
                if isinstance(other, AudioModelConfig)
                else _estimate_model_vram_mb(other)
            )
            if other_mb is None:
                log.warning(
                    "VRAM estimate for co-resident %s unavailable; not "
                    "counting it against the fit budget",
                    name,
                )
                continue
            used_mb += other_mb
            breakdown.append(f"{name} ~{other_mb} MiB")
        if used_mb > target_gpu.vram_mb:
            detail = ""
            if getattr(target, "always_resident", False) and used_mb > target_mb:
                # The pinned path cannot resolve this by evicting, so say what
                # the choice actually is rather than leaving "not enough VRAM"
                # to be read as a dead end.
                detail = (
                    f". {target.name!r} is pinned, so it loads alongside what is "
                    "already resident instead of evicting it. Either stop the "
                    "other model, give this one a smaller footprint (a lower "
                    "`ctx` for ASR, a quantized checkpoint for TTS), or "
                    "re-register it with --swappable to let it evict again"
                )
            # Itemised, because the total on its own is unactionable: a
            # co-resident that is being over-estimated is invisible in a single
            # number, and the fix (a `vram_mb` override, a smaller ctx, a
            # different model) depends entirely on which one it is.
            residents = "; ".join(breakdown) if breakdown else "none"
            raise RuntimeError(
                f"model {target.name!r} needs ~{target_mb} MiB on GPU "
                f"{target_gpu.pci_slot} but only {target_gpu.vram_mb} MiB is available "
                f"(estimated total {used_mb} MiB = {target.name} ~{target_mb} MiB "
                f"+ co-residents [{residents}]){detail}. These are estimates; if one "
                "is wrong for your setup, pin it with `vram_mb` in that model's "
                "config entry"
            )

    async def _evict_for(
        self,
        target: ModelConfig | AudioModelConfig,
        target_gpu: GPUConfig,
        drain_seconds: float = 30.0,
    ) -> None:
        """Stop the right neighbours so the target can have its GPU.

        An incumbent that is still serving requests gets a bounded drain
        first: killing llama-server mid-generation errors the streaming
        client for no reason the user can see. After ``drain_seconds`` the
        eviction proceeds anyway — the new request asked for this GPU, and
        blocking it behind an arbitrarily long generation would trade one
        stall for another. New requests for the incumbent can keep arriving
        through the lockless fast path while we wait, which is exactly why
        the drain is bounded rather than a wait-for-zero.
        """
        if getattr(target, "always_resident", False):
            # A pinned model is declared to *coexist*, so loading one evicts
            # nothing. Protecting it from eviction is only half the promise:
            # if the first voice command still displaced the LLM, the utterance
            # would cost exactly the cold start pinning exists to avoid, and
            # the model would simply be pinned in the wrong place afterwards.
            # Whether it actually fits alongside is _check_vram_fit's call,
            # which runs next and now sees the incumbent as a real co-resident.
            log.debug("%s is pinned; loading it evicts nothing", target.name)
            return

        single = self.cfg.server.single_resident
        for name, srv in self._servers.items():
            if name == target.name:
                continue
            if not srv.is_running:
                continue
            other_model = self._entry_for(name)
            if other_model is None:
                self._stopping.add(name)
                try:
                    await srv.astop()
                finally:
                    self._stopping.discard(name)
                continue
            if getattr(other_model, "always_resident", False):
                # A pinned model (an ASR backend serving voice commands, say)
                # keeps its VRAM through every swap. Evicting it would make a
                # single utterance cost two cold starts: one to load the LLM
                # that displaced it, one to load it again for the next
                # utterance. Its footprint is still charged to the GPU budget
                # in _check_vram_fit, so this cannot silently overcommit.
                log.debug("not evicting pinned model %s for %s", name, target.name)
                continue
            if single or other_model.gpu_pci_slot == target_gpu.pci_slot:
                deadline = time.monotonic() + drain_seconds
                while self.model_inflight.get(name, 0) > 0 and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                # From the final counter read to the _stopping mark there is
                # no await, so a lock-free fast-path acquire either landed
                # before the read (count > 0, we keep draining) or after the
                # mark (sees _stopping and takes the slow path). It can never
                # slip between them and receive a server that is going down.
                still = self.model_inflight.get(name, 0)
                if still:
                    log.warning(
                        "evicting %s with %d request(s) still in flight after "
                        "%.0fs drain; their clients will see errors",
                        name,
                        still,
                        drain_seconds,
                    )
                log.info("evicting %s before starting %s", name, target.name)
                self._stopping.add(name)
                try:
                    await srv.astop()
                finally:
                    self._stopping.discard(name)

    async def stop_one(self, name: str) -> bool:
        """Stop a single model's llama-server. Returns True if it was running."""
        async with self._lock:
            srv = self._servers.get(name)
            if srv is None or not srv.is_running:
                return False
            await srv.astop()
            self.metrics["stops"] += 1
            return True

    async def stop_all(self) -> int:
        """Stop every running llama-server. Returns the count stopped."""
        async with self._lock:
            stopped = 0
            for srv in self._servers.values():
                if srv.is_running:
                    await srv.astop()
                    stopped += 1
            self.metrics["stops"] += stopped
            return stopped

    async def rebuild_model(self, name: str, drain_seconds: float = 30.0) -> tuple[bool, bool]:
        """Drop and rebuild the LlamaServer for one model after a config edit.

        If the model is currently loaded, it's stopped first — the recipe is
        consumed at process start, so an in-flight server can't pick up new
        flags. A request that acquired the model through the lock-free fast
        path an instant before we took the lock (the deferred autotune
        restore racing a real request is the case that motivated this) gets
        the same bounded drain an eviction gets, instead of having its
        generation killed mid-stream. Returns (rebuilt, was_running).
        """
        async with self._lock:
            old = self._servers.get(name)
            was_running = bool(old and old.is_running)
            if old is not None and old.is_running:
                deadline = time.monotonic() + drain_seconds
                while self.model_inflight.get(name, 0) > 0 and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                still = self.model_inflight.get(name, 0)
                if still:
                    log.warning(
                        "rebuild %s: stopping with %d request(s) still in "
                        "flight after %.0fs drain; their clients will see errors",
                        name,
                        still,
                        drain_seconds,
                    )
                # Same synchronous segment as the final counter read: a
                # fast-path acquire either preceded it (we drained) or
                # follows the mark (takes the slow path and waits on the
                # lock we hold).
                self._stopping.add(name)
                try:
                    await old.astop()
                finally:
                    self._stopping.discard(name)
            self._servers.pop(name, None)
            cfg_model = self._entry_for(name)
            if cfg_model is None:
                return False, was_running
            gpu = self.cfg.find_gpu(cfg_model.gpu_pci_slot)
            if gpu is None:
                return False, was_running
            try:
                if isinstance(cfg_model, AudioModelConfig):
                    # For TTS this also rewrites the generated voice table,
                    # which is how an edited voice reaches the subprocess.
                    plan = build_audio_plan(self.cfg, cfg_model, gpu, host=self.cfg.server.host)
                else:
                    plan = build_plan(self.cfg, cfg_model, gpu, host=self.cfg.server.host)
            except RuntimeError as exc:
                log.warning("rebuild %s: not launchable: %s", name, exc)
                return False, was_running
            self._servers[name] = LlamaServer(plan, name=name)
            return True, was_running

    async def shutdown(self) -> None:
        async with self._lock:
            for srv in self._servers.values():
                await srv.astop()
