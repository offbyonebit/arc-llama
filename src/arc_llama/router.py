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

from arc_llama.config import Config, GPUConfig, ModelConfig
from arc_llama.failures import StartupFailureError
from arc_llama.gguf_meta import (
    estimate_weight_vram_bytes,
    kv_bytes_per_token_f16,
    override_tensor_saved_bytes,
    scan_weight_tensors,
    weight_tensor_table,
)
from arc_llama.launcher import LlamaServer, build_plan
from arc_llama.preflight import preflight_launch
from arc_llama.recipes import KVCacheType, estimate_kv_bytes

log = logging.getLogger("arc_llama.router")

# Rough overhead budgets for VRAM estimation (MiB).
_VRAM_COMPUTE_BUFFER_MB = 768
_VRAM_SAFETY_MARGIN_MB = 256


def estimate_model_vram_quick_mb(model: ModelConfig) -> int | None:
    """Conservative constant-time fit estimate for a launch-plan preview.

    Dense GGUF weight storage is already packed in the representation loaded
    by llama.cpp, so file size plus KV, compute, and safety allocations is a
    useful preview without walking every tensor. MoE/regex offload changes the
    resident weight set and deliberately falls back to the exact estimator.
    The router's admission guard continues to use ``_estimate_model_vram_mb``;
    this helper only removes a multi-second scan from user-facing setup.
    """
    recipe = model.recipe or {}
    if recipe.get("n_cpu_moe") or recipe.get("override_tensor"):
        return None
    try:
        size = Path(model.path).stat().st_size
    except OSError:
        return None
    mib = 1_048_576
    weight_mb = (size + mib - 1) // mib
    ctx = int(recipe.get("ctx", 8192))
    kv_type = KVCacheType(recipe.get("cache_type_k", "f16"))
    kv_mb = (
        estimate_kv_bytes(
            ctx,
            kv_type,
            model.kv_class,
            kv_bytes_per_token_f16(model.path),
        )
        // mib
    )
    return weight_mb + kv_mb + _VRAM_COMPUTE_BUFFER_MB + _VRAM_SAFETY_MARGIN_MB


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
    kv_mb = estimate_kv_bytes(
        eff_ctx,
        eff_kv,
        model.kv_class,
        kv_bytes_per_token_f16(model.path),
    ) // (1_048_576)
    buffer_mb = compute_buffer_mb if compute_buffer_mb is not None else _VRAM_COMPUTE_BUFFER_MB
    return weight_mb + kv_mb + buffer_mb + _VRAM_SAFETY_MARGIN_MB


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
    kv_mb = estimate_kv_bytes(
        eff_ctx,
        eff_kv,
        model.kv_class,
        kv_bytes_per_token_f16(model.path),
    ) // (1_048_576)
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


class Router:
    """Owns one LlamaServer per registered model and serialises swaps."""

    def __init__(self, cfg: Config, log_dir: Path | None = None):
        self.cfg = cfg
        self.log_dir = log_dir
        self._servers: dict[str, LlamaServer] = {}  # keyed by model.name
        self._lock = asyncio.Lock()
        self._loading_futures: dict[str, asyncio.Future[tuple[ModelConfig, LlamaServer]]] = {}
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
        """(Re)build the per-model LlamaServer registry from cfg.

        Idempotent — existing servers (running or not) are preserved by name,
        only new model entries get fresh LlamaServer instances. Use after a
        runtime config mutation (e.g. an admin scan).
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

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve(self, query: str) -> tuple[ModelConfig, GPUConfig, LlamaServer] | None:
        m = self.cfg.find_model(query)
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
    ) -> tuple[ModelConfig, LlamaServer]:
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
                configured = self.cfg.find_model(query)
                if configured is not None:
                    raise StartupFailureError(
                        "gpu_unavailable",
                        f"Configured GPU is unavailable: {configured.gpu_pci_slot}.",
                        "Assign the model to an available enabled GPU and retry.",
                        details={
                            "model": configured.name,
                            "gpu": configured.gpu_pci_slot,
                        },
                    )
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

            if (
                target_srv.is_running
                and target_srv.ready
                and target_model.name not in self._stopping
            ):
                if acquire:
                    self.acquire_model(target_model.name)
                return target_model, target_srv

            # Reject predictable failures before evicting a healthy resident.
            # File and runtime checks are blocking filesystem operations, and
            # the fit estimate may scan GGUF metadata, so keep both off-loop.
            await asyncio.to_thread(
                preflight_launch, target_model, target_gpu, target_srv.plan
            )
            await asyncio.to_thread(self._check_vram_fit, target_model, target_gpu)

            await self._evict_for(target_model, target_gpu)

            # We are the one responsible for starting.
            log.info("loading model %s on GPU %s ...", target_model.name, target_gpu.pci_slot)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[tuple[ModelConfig, LlamaServer]] = loop.create_future()
            self._loading_futures[target_model.name] = future
            try:
                try:
                    target_srv.start(log_dir=self.log_dir)
                except OSError as exc:
                    category = "runtime_missing" if isinstance(exc, FileNotFoundError) else "process_exited"
                    raise StartupFailureError(
                        category,
                        f"llama-server could not start for {target_model.name}: {exc}.",
                        "Check the configured runtime and its required libraries, then retry.",
                        details={
                            "model": target_model.name,
                            "backend": target_gpu.backend,
                            "gpu": target_gpu.pci_slot,
                            "argv": target_srv.plan.argv,
                            "reason": str(exc),
                        },
                    ) from exc
                ready = await target_srv.wait_ready()
                if not ready:
                    tail = target_srv.tail_log(lines=40)
                    log.error(
                        "model %s failed health-check; stopping it",
                        target_model.name,
                    )
                    target_srv.stop()
                    self.metrics["last_error"] = f"{target_model.name} did not become healthy"
                    process = getattr(target_srv, "process", None)
                    exit_code = process.poll() if process is not None else None
                    category = "process_exited" if exit_code is not None else "startup_timeout"
                    failure = StartupFailureError(
                        category,
                        (
                            f"llama-server exited while loading {target_model.name}."
                            if exit_code is not None
                            else f"llama-server timed out while loading {target_model.name}."
                        ),
                        "Open the retained model log, correct the reported problem, and retry.",
                        details={
                            "model": target_model.name,
                            "backend": target_gpu.backend,
                            "gpu": target_gpu.pci_slot,
                            "argv": target_srv.plan.argv,
                            "exit_code": exit_code,
                            "log_path": (
                                str(getattr(target_srv, "log_path", None))
                                if getattr(target_srv, "log_path", None)
                                else None
                            ),
                            "log_tail": tail,
                        },
                    )
                    log.error(
                        "startup diagnostic %s: %s details=%r",
                        failure.diagnostics_id,
                        failure.message,
                        failure.details,
                    )
                    raise failure
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
                    # The starter raises ``exc`` directly. If no concurrent
                    # request joined this future, nobody awaits it and asyncio
                    # otherwise emits "Future exception was never retrieved".
                    # Retrieving it here only marks it observed; existing and
                    # later waiters still receive the same exception.
                    future.exception()
                raise
            finally:
                self._loading_futures.pop(target_model.name, None)

    def _check_vram_fit(self, target: ModelConfig, target_gpu: GPUConfig) -> None:
        """Refuse to load *target* if its estimated VRAM won't fit on target_gpu.

        In multi-resident mode this also accounts for other loaded models that
        share the same GPU.
        """
        if not target_gpu.vram_mb:
            return
        target_mb = _estimate_model_vram_mb(target)
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
        for name, srv in self._servers.items():
            if name == target.name or not srv.is_running:
                continue
            other = next((m for m in self.cfg.models if m.name == name), None)
            if other is None or other.gpu_pci_slot != target_gpu.pci_slot:
                continue
            other_mb = _estimate_model_vram_mb(other)
            if other_mb is None:
                log.warning(
                    "VRAM estimate for co-resident %s unavailable; not "
                    "counting it against the fit budget",
                    name,
                )
                continue
            used_mb += other_mb
        if used_mb > target_gpu.vram_mb:
            message = (
                f"model {target.name!r} needs ~{target_mb} MiB on GPU "
                f"{target_gpu.pci_slot} but only {target_gpu.vram_mb} MiB is available "
                f"(estimated total with co-residents: {used_mb} MiB)"
            )
            raise StartupFailureError(
                "out_of_memory",
                message,
                "Reduce context or GPU layers, enable expert offload, or choose a smaller model.",
                details={
                    "model": target.name,
                    "gpu": target_gpu.pci_slot,
                    "estimated_model_mb": target_mb,
                    "estimated_total_mb": used_mb,
                    "available_mb": target_gpu.vram_mb,
                },
            )

    async def _evict_for(
        self, target: ModelConfig, target_gpu: GPUConfig, drain_seconds: float = 30.0
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
        single = self.cfg.server.single_resident
        for name, srv in self._servers.items():
            if name == target.name:
                continue
            if not srv.is_running:
                continue
            other_model = next((m for m in self.cfg.models if m.name == name), None)
            if other_model is None:
                self._stopping.add(name)
                try:
                    await srv.astop()
                finally:
                    self._stopping.discard(name)
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
            cfg_model = next((m for m in self.cfg.models if m.name == name), None)
            if cfg_model is None:
                return False, was_running
            gpu = self.cfg.find_gpu(cfg_model.gpu_pci_slot)
            if gpu is None:
                return False, was_running
            plan = build_plan(self.cfg, cfg_model, gpu, host=self.cfg.server.host)
            self._servers[name] = LlamaServer(plan, name=name)
            return True, was_running

    async def shutdown(self) -> None:
        async with self._lock:
            for srv in self._servers.values():
                await srv.astop()
