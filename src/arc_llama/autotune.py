"""Background auto-tuner.

The router serves the user's first request immediately with the static,
arch-derived recipe. When the router is idle, this module picks one used,
local model whose tuned fingerprint is missing or stale and drives
tune_model over the loopback admin HTTP surface. If a real request arrives
mid-sweep, the should_abort hook stops the loop; tune.py's try/finally
restores the winning recipe so far.

Keeping the *when* decision here and the *what* search in tune.py means we
do not duplicate the staged sweep logic. A fingerprint helper here ties the
validity of a saved recipe to the exact binary, GPU, and arc-llama version
that measured it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from arc_llama.config import Config, GPUConfig, ModelConfig
    from arc_llama.router import Router
    from arc_llama.tune import TuneReport

log = logging.getLogger("arc_llama.autotune")

# Bump whenever build_stages changes its axes, so previously tuned recipes are
# reconsidered under the new search space.
# 2: added the MoE expert-offload axis (n_cpu_moe) between the KV and ubatch
#    stages, and ubatch candidates are now pruned against the chosen offload.
TUNE_SCHEMA_VERSION = 3

LOOP_INTERVAL_SECONDS = 15


def _file_fingerprint(path: str) -> str:
    """Stable identifier for a file: size + mtime, or empty if missing."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{st.st_mtime}"
    except OSError:
        return ""


def compute_fingerprint(
    model: ModelConfig,
    llama_server_path: str,
    gpu: GPUConfig | None,
    arc_llama_version: str,
    workload_key: str = "",
) -> str:
    """SHA256 over everything that invalidates a tuned recipe.

    A recipe measured with one llama-server binary is not valid for another,
    so the llama-server path and mtime are mixed in. The GPU slot, arch and
    backend catch card swaps and SYCL/Vulkan switches. The arc-llama version
    and schema version make package upgrades and search-space changes retune
    automatically. The workload profile key makes changing any workload answer
    retune too: the profile changes what the sweep measures, so a recipe
    tuned under the old answers was never measured for the new workload.
    """
    h = hashlib.sha256()
    h.update(b"model:")
    h.update(model.path.encode())
    h.update(b"|")
    h.update(_file_fingerprint(model.path).encode())
    h.update(b"\nbinary:")
    h.update(llama_server_path.encode())
    h.update(b"|")
    h.update(_file_fingerprint(llama_server_path).encode())
    h.update(b"\ngpu:")
    h.update((gpu.pci_slot if gpu else "").encode())
    h.update(b"|")
    h.update((gpu.arch if gpu else "").encode())
    h.update(b"|")
    h.update((gpu.backend if gpu else "").encode())
    h.update(b"\nversion:")
    h.update(arc_llama_version.encode())
    h.update(b"\nworkload:")
    h.update(workload_key.encode())
    h.update(b"\nschema:")
    h.update(str(TUNE_SCHEMA_VERSION).encode())
    return h.hexdigest()


def fingerprint_matches(model: ModelConfig, fingerprint: str) -> bool:
    """True when the model already carries the given fingerprint."""
    return bool(model.tune_fingerprint) and model.tune_fingerprint == fingerprint


def set_tuned_state(
    cfg: Config,
    model: ModelConfig,
    fingerprint: str,
) -> None:
    """Record the model as tuned with the current fingerprint and timestamp."""
    model.tune_state = "tuned"
    model.tuned_at = time.time()
    model.tune_fingerprint = fingerprint
    model.tune_error = ""


def set_tune_failed_state(
    cfg: Config,
    model: ModelConfig,
    error: str,
) -> None:
    """Record a failed sweep so we do not retry in a tight loop."""
    model.tune_state = "failed"
    model.tune_error = error[:2048]


def reset_tuned_state_if_stale(
    cfg: Config,
    model: ModelConfig,
    arc_llama_version: str,
) -> None:
    """If the model is tuned but the fingerprint changed, mark it untuned.

    Called once per loop iteration before picking a candidate. An upstream
    model is always left untouched.
    """
    if not cfg.tune.retune_on_fingerprint_change:
        return
    from arc_llama import workload

    gpu = cfg.find_gpu(model.gpu_pci_slot)
    fp = compute_fingerprint(
        model,
        cfg.paths.llama_server,
        gpu,
        arc_llama_version,
        workload.fingerprint_key(cfg.workload),
    )
    if model.tune_state == "tuned" and not fingerprint_matches(model, fp):
        log.info(
            "autotune: fingerprint changed for %s; treating as untuned",
            model.name,
        )
        model.tune_state = "untuned"
        model.tune_fingerprint = ""
        model.tuned_at = None
        model.tune_error = ""


def _now() -> float:
    return time.time()


class Autotuner:
    """Async background loop that queues and runs idle-time sweeps."""

    def __init__(
        self,
        cfg: Config,
        router: Router,
        *,
        version: str,
        on_save: Callable[[], None] | None = None,
        loop_interval: float = LOOP_INTERVAL_SECONDS,
    ) -> None:
        self.cfg = cfg
        self.router = router
        self.version = version
        self.on_save = on_save
        self.loop_interval = loop_interval
        self._task: asyncio.Task[None] | None = None
        self._abort_event = asyncio.Event()
        self._stopping = False
        self.running_model: str | None = None
        self.running_stage: str | None = None
        self._lock = asyncio.Lock()
        self._use_counts: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self.last_results: dict[str, dict[str, Any]] = {}

    def bump_use(self, model_name: str) -> None:
        """Call when a model is used by a real request.

        Records both how many times the model has been used and the last time
        it was touched. Both are in-memory only; a server restart simply delays
        tuning by one more use.
        """
        self._use_counts[model_name] = self._use_counts.get(model_name, 0) + 1
        self._last_used[model_name] = time.time()

    @property
    def is_running(self) -> bool:
        """True when the background task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def is_sweep_running(self) -> bool:
        return self.running_model is not None

    async def start(self) -> None:
        """Start the background loop.

        Serialized with stop() through the same lock: a start() that races an
        in-flight stop() used to see the old (about-to-be-reaped) task, no-op,
        and could leave _stopping set with no loop running, or orphan a new
        task when stop() then set _task = None. Now start() waits for stop()
        to finish reaping before deciding, so exactly one of them wins.
        """
        if not self.cfg.tune.auto:
            log.debug("autotune: disabled by config")
            return
        async with self._lock:
            if self._task is not None and not self._task.done():
                return
            self._abort_event.clear()
            self._stopping = False
            self._task = asyncio.create_task(self._loop())
            log.info("autotune: started background loop")

    async def stop(self) -> None:
        async with self._lock:
            if self._task is None:
                return
            # Set _stopping before the abort event: _loop checks it at the top
            # of every iteration, so the loop exits even if the cancellation is
            # lost (see _wait_for_abort).
            self._stopping = True
            self._abort_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            log.info("autotune: stopped background loop")

    def queue_now(self, model_name: str) -> bool:
        """Mark a model as eligible for immediate tuning.

        The actual ordering is still decided by _pick_candidate, but a queued
        model is treated as if its use_count were already high enough.
        """
        m = self.cfg.find_model(model_name)
        if m is None or m.tune_state == "failed":
            return False
        self._use_counts[model_name] = max(
            self._use_counts.get(model_name, 0),
            self.cfg.tune.min_uses,
        )
        return True

    def abort_sweep(self) -> bool:
        """Request abort of the currently running sweep, if any.

        Returns True if a sweep was running and the abort signal was set.
        """
        if not self.is_sweep_running:
            return False
        self._abort_event.set()
        return True

    async def _wait_for_abort(self, timeout: float) -> None:
        """Wait up to *timeout* seconds for the abort event to be set.

        Deliberately not `asyncio.wait_for`. On Python < 3.12 `wait_for`
        swallows a CancelledError that lands after the inner future has already
        resolved, and `stop()` hits that window every time: it sets the abort
        event and cancels in the same tick, so the event is ready before the
        cancellation is delivered. The loop then never exits and shutdown hangs
        forever awaiting the task. `asyncio.wait` has no such window.
        """
        waiter = asyncio.ensure_future(self._abort_event.wait())
        try:
            await asyncio.wait({waiter}, timeout=timeout)
        finally:
            waiter.cancel()
            # Awaited, not just cancelled: a bare cancel() leaves the task
            # pending until the loop next runs it, which on a closing loop
            # means "Task was destroyed but it is pending!".
            try:
                await waiter
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        try:
            while not self._stopping:
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.exception("autotune: tick failed: %s", exc)
                if self._stopping:
                    break
                await self._wait_for_abort(self.loop_interval)
                # Only clear if this iteration set it; a new abort signal
                # should be visible on the next iteration.
                if self._abort_event.is_set() and not self._stopping:
                    self._abort_event.clear()
        except asyncio.CancelledError:
            log.info("autotune: loop cancelled")
            raise

    async def _tick(self) -> None:
        if not self.cfg.tune.auto:
            return

        idle = self.cfg.tune.idle_seconds
        now = _now()
        if self.router.inflight > 0 or now - self.router.last_activity < idle:
            return

        for m in self.cfg.models:
            reset_tuned_state_if_stale(self.cfg, m, self.version)

        candidate = self._pick_candidate()
        if candidate is None:
            return

        if not self.cfg.server.single_resident and self._other_model_running(candidate.name):
            log.info(
                "autotune: %s skipped; another model is running in multi-resident mode",
                candidate.name,
            )
            self._mark_skipped(candidate.name)
            return

        await self._run_sweep(candidate)

    def _pick_candidate(self) -> ModelConfig | None:
        """Select the most recently used eligible model.

        Upstream models are skipped by definition — they have no local
        recipe. Candidates must be untuned/skipped, used enough times, and
        have no failed state. Among multiple eligible models, prefer the one
        that was used most recently.
        """
        eligible: list[tuple[ModelConfig, float]] = []
        for m in self.cfg.models:
            if self.cfg.find_gpu(m.gpu_pci_slot) is None:
                continue
            if m.tune_state == "failed":
                continue
            if self._use_count(m.name) < self.cfg.tune.min_uses:
                continue
            if not self._fingerprint_outdated(m):
                continue
            # Order by most recent use. Never-used models share 0.0, but
            # min_uses >= 1 guarantees they only appear after at least one use.
            last_used = self._last_used.get(m.name, 0.0)
            eligible.append((m, last_used))
        if not eligible:
            return None
        # Prefer most recently used. Use name as a stable tie-breaker.
        eligible.sort(key=lambda x: (x[1], x[0].name), reverse=True)
        return eligible[0][0]

    def _use_count(self, name: str) -> int:
        return self._use_counts.get(name, 0)

    def _last_used_at(self, name: str) -> float:
        return self._last_used.get(name, 0.0)

    def _fingerprint_outdated(self, model: ModelConfig) -> bool:
        from arc_llama import workload

        gpu = self.cfg.find_gpu(model.gpu_pci_slot)
        fp = compute_fingerprint(
            model,
            self.cfg.paths.llama_server,
            gpu,
            self.version,
            workload.fingerprint_key(self.cfg.workload),
        )
        return not fingerprint_matches(model, fp)

    def _other_model_running(self, candidate_name: str) -> bool:
        # Router.running_models() is a lock-free snapshot (no await inside,
        # live subprocess probes); the decision it feeds is re-validated by
        # the sweep's should_abort hook if a load starts right after.
        return any(n != candidate_name for n in self.router.running_models())

    def _mark_skipped(self, name: str) -> None:
        m = self.cfg.find_model(name)
        if m is not None:
            m.tune_state = "skipped"
            if self.on_save:
                try:
                    self.on_save()
                except Exception as exc:  # noqa: BLE001
                    log.warning("autotune: save after skip failed: %s", exc)

    async def _run_sweep(self, model: ModelConfig) -> None:
        from arc_llama import workload
        from arc_llama.tune import tune_model as _tune_model

        self.running_model = model.name
        self.running_stage = "baseline"
        log.info("autotune: starting sweep of %s", model.name)

        # Synchronous on purpose: tune.py invokes on_stage like the on_start /
        # on_done callbacks in tune_all — synchronously. An async def here was
        # never awaited, so sweep_stage stayed "baseline" for the whole sweep.
        def _stage_callback(name: str, stage: int, total: int) -> None:
            self.running_stage = f"{name} ({stage}/{total})"
            log.info(
                "autotune: %s stage %d/%d: %s",
                model.name,
                stage,
                total,
                name,
            )

        def _should_abort() -> bool:
            return self.router.inflight > 0 or self._abort_event.is_set()

        restore_bodies: list[dict[str, Any]] = []

        async def _deferred_restore(final_state: dict[str, Any]) -> None:
            """Wait for real requests to drain, then POST the restore recipe.

            tune_model calls this instead of applying the restore itself when
            the sweep was aborted. The wait happens here, in the policy layer,
            because autotune owns the decision about when it is safe to write
            a recipe (i.e., when no user request is holding the backend).
            """
            deadline = _now() + 30.0
            # Plain sleep, deliberately not _wait_for_abort. The old wait
            # cleared _abort_event each lap to stop the set event turning the
            # 0.2s waits into a spin — but that consumed abort signals that
            # belong to the outer loop and abort_sweep() callers, so an abort
            # posted during this drain silently vanished. The event is not
            # this coroutine's to clear. Responsiveness is unharmed: stop()
            # cancels the enclosing task, which interrupts the sleep, and
            # _stopping is checked every lap.
            while self.router.inflight > 0 and _now() < deadline and not self._stopping:
                await asyncio.sleep(0.2)
            if self._stopping:
                # Don't start a recipe write we cannot finish. The enclosing
                # task is already cancelled, so the POST below would be torn
                # down at its first await anyway; returning here makes that
                # explicit instead of leaving it to cancellation timing.
                log.info("autotune: shutting down; skipping deferred restore")
                return
            if self.router.inflight > 0:
                log.warning(
                    "autotune: inflight still %d after timeout; applying restore anyway",
                    self.router.inflight,
                )
            headers: dict[str, str] = {}
            if self.cfg.server.admin_token:
                headers["Authorization"] = f"Bearer {self.cfg.server.admin_token}"
            async with httpx.AsyncClient(
                base_url=f"http://{self.cfg.server.host}:{self.cfg.server.port}",
                timeout=600.0,
                headers=headers,
            ) as client:
                from arc_llama.tune import _apply_edits

                err = await _apply_edits(client, model.name, final_state)
                restore_bodies.append(dict(final_state))
                if err:
                    log.warning("autotune: deferred restore failed: %s", err)

        report: TuneReport | None = None
        error: str | None = None
        try:
            report = await _tune_model(
                f"http://{self.cfg.server.host}:{self.cfg.server.port}",
                model.name,
                target=workload.tune_target(self.cfg),
                prompt_tokens=self.cfg.tune.prompt_tokens,
                gen_tokens=self.cfg.tune.gen_tokens,
                apply=True,
                cfg=self.cfg,
                should_abort=_should_abort,
                on_stage=_stage_callback,
                on_deferred_restore=_deferred_restore,
            )
            if report.aborted:
                log.info("autotune: sweep of %s aborted; leaving untuned", model.name)
                model.tune_state = "untuned"
                model.tuned_at = None
                model.tune_error = ""
            elif report.error:
                error = report.error
                set_tune_failed_state(self.cfg, model, error)
            else:
                gpu = self.cfg.find_gpu(model.gpu_pci_slot)
                fp = compute_fingerprint(
                    model,
                    self.cfg.paths.llama_server,
                    gpu,
                    self.version,
                    workload.fingerprint_key(self.cfg.workload),
                )
                set_tuned_state(self.cfg, model, fp)
                if report.baseline is not None and report.best is not None:
                    self.last_results.pop(model.name, None)
                    self.last_results[model.name] = {
                        "measured_at": time.time(),
                        "applied": report.applied,
                        "before": {
                            "prompt_tok_s": report.baseline.prompt_eval_tok_s,
                            "generation_tok_s": report.baseline.generation_tok_s,
                        },
                        "after": {
                            "prompt_tok_s": report.best.prompt_eval_tok_s,
                            "generation_tok_s": report.best.generation_tok_s,
                        },
                        "improvement_pct": report.improvement_pct,
                    }
                    while len(self.last_results) > 128:
                        self.last_results.pop(next(iter(self.last_results)))
        except asyncio.CancelledError:
            log.info("autotune: sweep of %s cancelled", model.name)
            model.tune_state = "untuned"
            model.tuned_at = None
            model.tune_error = ""
            raise
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            log.exception("autotune: sweep of %s failed: %s", model.name, error)
            set_tune_failed_state(self.cfg, model, error)
        finally:
            self.running_model = None
            self.running_stage = None
            if self.on_save:
                try:
                    self.on_save()
                except Exception as exc:  # noqa: BLE001
                    log.warning("autotune: save after sweep failed: %s", exc)


async def start_autotuner(
    cfg: Config,
    router: Router,
    *,
    version: str,
    on_save: Callable[[], None] | None = None,
) -> Autotuner:
    """Create and start the autotuner for this server instance."""
    tuner = Autotuner(cfg, router, version=version, on_save=on_save)
    await tuner.start()
    return tuner
