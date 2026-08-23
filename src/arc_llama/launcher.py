"""Manage llama-server subprocesses.

A `LlamaServer` owns one llama-server process bound to one model on one GPU.
It builds the command line from an arch profile + recipe + model config so the
SYCL gotchas are applied uniformly.

Subprocesses are launched with `setsid` (so they own a fresh process group we
can `killpg`) AND `PR_SET_PDEATHSIG=SIGTERM` so the kernel reaps the child if
the arc-llama parent dies hard — no orphan llama-servers holding VRAM.
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from arc_llama.arch import Arch, ArchProfile, Backend, profile_for
from arc_llama.binary import list_vulkan_devices, resolve_vulkan_index
from arc_llama.config import (
    AudioModelConfig,
    Config,
    GPUConfig,
    ModelConfig,
)
from arc_llama.gguf_meta import has_mtp_heads
from arc_llama.platform_checks import (
    oneapi_runtime_env_needed,
    oneapi_setvars_path,
    source_setvars,
)
from arc_llama.policy import apply_launch_policy
from arc_llama.server_caps import probe_server_caps

log = logging.getLogger("arc_llama.launcher")

DEFAULT_HEALTH_TIMEOUT = 120  # seconds — generous for cold-start SYCL JIT
HEALTH_POLL_INTERVAL = 1.5

# Log rotation for llama-server subprocess logs.
_MAX_LOG_BYTES = 50 * 1024 * 1024
_LOG_BACKUPS = 3

_IS_WINDOWS = sys.platform == "win32"
# Not defined on POSIX. Fallback lets the Windows code path stay import-safe
# when exercised under tests that monkeypatch _IS_WINDOWS on a Linux runner.
_CTRL_BREAK_EVENT = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)

# Linux prctl(2) constant. We don't import a real binding — one syscall.
_PR_SET_PDEATHSIG = 1


_libc: ctypes.CDLL | None = None


def _load_libc() -> ctypes.CDLL | None:
    global _libc
    if _libc is not None:
        return _libc
    try:
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        # int prctl(int option, unsigned long arg2, ...arg5)
        lib.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ]
        lib.prctl.restype = ctypes.c_int
        _libc = lib
    except OSError:
        _libc = None
    return _libc


def _preexec_isolate_and_pdeathsig() -> None:
    """preexec_fn: detach into a new session and tie our lifetime to the parent's.

    Runs in the child between fork and exec. setsid() makes the child a new
    process-group/session leader (so we can `killpg` it cleanly), prctl with
    PR_SET_PDEATHSIG ensures the kernel sends us SIGTERM the moment the parent
    arc-llama process exits — covers crashes, SIGKILL, oom-killer, etc.

    NOTE: PDEATHSIG tracks the *thread* that did the fork, not the whole parent
    process. If Python's main thread dies but a worker thread is what spawned us,
    we wouldn't get the signal. arc-llama spawns from the asyncio loop running
    in the main thread, so this is a non-issue today, but a future move to a
    thread-pool launcher would need rework.
    """
    os.setsid()  # type: ignore[attr-defined]
    libc = _load_libc()
    if libc is None:
        return
    rc = libc.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(int(signal.SIGTERM)), 0, 0, 0)
    if rc != 0:
        # Can't really log here — preexec_fn runs in a fragile post-fork state.
        # The child will simply not receive PDEATHSIG; not fatal.
        pass


def _rotate_log(log_path: Path) -> None:
    """Rotate an existing log file so it doesn't grow unbounded.

    Keeps up to ``_LOG_BACKUPS`` historic files (``.log.1``, ``.log.2``, ...).
    """
    if not log_path.exists():
        return
    try:
        if log_path.stat().st_size < _MAX_LOG_BYTES:
            return
    except OSError:
        return
    for i in range(_LOG_BACKUPS, 0, -1):
        src = log_path.parent / f"{log_path.name}.{i}"
        dst = log_path.parent / f"{log_path.name}.{i + 1}"
        if src.exists():
            try:
                src.replace(dst)
            except OSError:
                pass
    try:
        log_path.replace(log_path.parent / f"{log_path.name}.1")
    except OSError:
        pass


@dataclass
class LaunchPlan:
    """Everything needed to invoke a backend for one model."""
    argv: list[str]
    env: dict[str, str]
    cwd: str | None = None
    health_url: str = ""
    backend_url: str = ""
    health_timeout: float | None = None
    """Seconds to wait for /health, or None for ``DEFAULT_HEALTH_TIMEOUT``.

    The default is sized for llama-server's cold-start SYCL JIT. A TTS backend
    that imports torch and may download several GB of weights on first use
    needs a far larger budget, and a timeout that fires mid-download looks
    exactly like a backend that never starts.
    """


# Environment variables that only make sense for the SYCL backend; they can
# confuse the Vulkan loader or cause unnecessary backend initialization delays.
_SYCL_ONLY_ENVS: frozenset[str] = frozenset({
    "ONEAPI_DEVICE_SELECTOR",
    "ZES_ENABLE_SYSMAN",
    "SYCL_CACHE_PERSISTENT",
    "SYCL_CACHE_DIR",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "SYCL_DEVICE_FILTER",
    "SYCL_DEVICE_ALLOWLIST",
    "GGML_SYCL_DISABLE_OPT",
})


@lru_cache(maxsize=8)
def _vulkan_devices_cached(binary_path: str) -> tuple[tuple[int, str], ...]:
    return tuple(list_vulkan_devices(binary_path))


def _vulkan_index_for(gpu: GPUConfig, llama_server: str | Path | None) -> int | None:
    """Resolve the Vulkan device index for *gpu*, or None to leave it unset.

    Returning None means "don't set GGML_VK_VISIBLE_DEVICES at all", which lets
    llama.cpp apply its own default. That is deliberately preferred over
    guessing: a wrong index runs the model on another vendor's GPU and looks
    like success.
    """
    if gpu.vulkan_index is not None:
        return gpu.vulkan_index

    if llama_server:
        devices = list(_vulkan_devices_cached(str(llama_server)))
        index = resolve_vulkan_index(devices, gpu_name=gpu.name)
        if index is not None:
            return index
        if len(devices) > 1:
            log.warning(
                "Vulkan: could not identify which of %d devices is %s (%s). "
                "Not setting GGML_VK_VISIBLE_DEVICES; llama.cpp will pick its "
                "own default, which may be a different vendor's GPU. Set "
                "vulkan_index for this GPU in the config to be certain. "
                "Devices: %s",
                len(devices),
                gpu.name or "the configured GPU",
                gpu.pci_slot,
                ", ".join(f"{i}={n}" for i, n in devices),
            )
            return None
        if len(devices) == 1:
            return devices[0][0]

    # No binary to ask, so no way to map. Only safe when there is nothing to
    # confuse: a single-vendor box makes index 0 the right answer anyway.
    return None


def build_env(
    profile: ArchProfile,
    gpu: GPUConfig,
    llama_server: str | Path | None = None,
    oneapi_setvars: str | None = None,
    backend_override: Backend | None = None,
) -> dict[str, str]:
    """Compose the environment for llama-server based on backend and arch.

    ``backend_override`` ignores the GPU's configured backend, for a runtime
    whose backend support differs from llama.cpp's — a torch TTS engine reaches
    an Arc card through Level Zero and so wants the SYCL device selector
    whatever the GPU is configured for.
    """
    backend = backend_override or (Backend(gpu.backend) if gpu.backend else Backend.SYCL)
    env = os.environ.copy()

    if backend == Backend.VULKAN:
        # Vulkan path: keep the environment clean of SYCL/oneAPI selectors.
        for k in _SYCL_ONLY_ENVS:
            env.pop(k, None)
        # Restrict visible Vulkan devices to the one this model is bound to.
        # GGML_VK_VISIBLE_DEVICES accepts a comma-separated list; we expose
        # exactly one device so the model cannot accidentally land elsewhere.
        #
        # This must NOT be sycl_index. SYCL enumerates Intel devices only, so
        # sycl_index 0 is the first Arc card; Vulkan enumerates every vendor,
        # so on a box with a discrete NVIDIA/AMD card the Arc can be Vulkan1
        # while sycl_index is still 0. Using sycl_index there silently ran
        # models on the other vendor's GPU.
        index = _vulkan_index_for(gpu, llama_server)
        if index is not None:
            env["GGML_VK_VISIBLE_DEVICES"] = str(index)
        return env

    # SYCL path: apply arch-specific env, stripping known-bad inherited vars.
    for k in profile.sycl_env_remove:
        env.pop(k, None)
    env.update(profile.sycl_env)
    env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{gpu.sycl_index}"

    # If the current environment is missing the oneAPI runtime libraries, try to
    # source a setvars.sh automatically. This helps tarball/custom-prefix installs
    # that the system loader doesn't know about.
    if oneapi_runtime_env_needed():
        setvars: Path | None = None
        if oneapi_setvars:
            candidate = Path(oneapi_setvars).expanduser()
            if candidate.exists():
                setvars = candidate
        if setvars is None:
            setvars = oneapi_setvars_path()
        if setvars is not None:
            log.info(
                "SYCL environment missing oneAPI runtime libs; sourcing %s",
                setvars,
            )
            sourced = source_setvars(setvars)
            # Merge sourced env, but never overwrite the device selector we just
            # set or any user-provided SYCL-only overrides.
            for k, v in sourced.items():
                if k == "ONEAPI_DEVICE_SELECTOR":
                    continue
                if k in _SYCL_ONLY_ENVS and k in env:
                    continue
                env[k] = v
        else:
            log.warning(
                "SYCL environment missing oneAPI runtime libs and no setvars.sh "
                "found. If llama-server fails to start, source your oneAPI "
                "setvars.sh before running arc-llama."
            )
    return env


def build_plan(
    cfg: Config, model: ModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
) -> LaunchPlan:
    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    profile = profile_for(arch)
    backend = Backend(gpu.backend) if gpu.backend else Backend.SYCL
    env = build_env(
        profile,
        gpu,
        llama_server=cfg.paths.llama_server,
        oneapi_setvars=getattr(cfg.paths, "oneapi_setvars", None),
    )
    recipe = model.launch_recipe()

    # --- MTP head detection & safety wiring ---
    mtp_present = has_mtp_heads(model.path)

    # 1. We no longer force -ub 8 for MTP models. Empirically it destroys
    #    prompt-eval throughput (~9.5x slower) and upstream's auto-fit handles
    #    memory sizing better than a blanket micro-ubatch.

    # 2. Validate the draft-mtp wiring. A model may carry MTP heads either
    #    embedded in its own GGUF, or in a sidecar draft (--spec-draft-model).
    #    Warn only when neither source actually provides them.
    if recipe.spec_type == "draft-mtp":
        if recipe.spec_draft_model:
            draft_path = Path(recipe.spec_draft_model)
            if not draft_path.exists():
                log.warning(
                    "[%s] spec_draft_model %s does not exist; speculative "
                    "decoding will fail.",
                    model.name, draft_path,
                )
            elif not has_mtp_heads(draft_path):
                log.warning(
                    "[%s] spec_draft_model %s has no MTP heads; draft-mtp will "
                    "likely degenerate or crash.",
                    model.name, draft_path,
                )
        elif not mtp_present:
            log.warning(
                "[%s] recipe.spec_type='draft-mtp' but neither the GGUF nor a "
                "spec_draft_model provides MTP heads. Speculative decoding will "
                "likely degenerate or crash.",
                model.name,
            )

    # 3. Launch policy: only verified adjustments (e.g. Vulkan q8 needs
    #    --flash-attn). Do not strip draft-mtp or auto-switch backends based
    #    on unbenchmarked arch heuristics.
    recipe = apply_launch_policy(
        recipe,
        arch=arch,
        backend=backend,
        model_path=model.path,
        model_name=model.name,
    )

    caps = probe_server_caps(cfg.paths.llama_server)
    if recipe.flash_attn is not None and not caps.supports_flash_attn:
        log.info(
            "[%s] recipe requests flash_attn=%s but %s has no --flash-attn; omitting",
            model.name, recipe.flash_attn, cfg.paths.llama_server,
        )
        recipe.flash_attn = None

    argv: list[str] = [
        cfg.paths.llama_server,
        "-m", model.path,
        "--host", host,
        "--port", str(model.port),
    ]
    argv.extend(recipe.to_argv(fa_takes_value=caps.flash_attn_takes_value))
    backend_url = f"http://{host}:{model.port}"
    return LaunchPlan(
        argv=argv,
        env=env,
        backend_url=backend_url,
        health_url=f"{backend_url}/health",
    )


def resolve_binary(path_or_name: str) -> str | None:
    """Resolve a configured binary to a runnable path, or None if absent.

    A bare name (`llama-server`) goes through PATH the way the shell would;
    anything with a separator is taken literally. Returning None rather than
    the unresolved string lets callers say "not installed" instead of leaving
    the user to decode an ENOENT from a subprocess that never started.
    """
    if not path_or_name:
        return None
    candidate = Path(path_or_name).expanduser()
    if os.sep in path_or_name or (os.altsep and os.altsep in path_or_name):
        return str(candidate) if candidate.exists() else None
    found = shutil.which(str(candidate))
    return found or None


def build_audio_plan(
    cfg: Config, model: AudioModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
) -> LaunchPlan:
    """Everything needed to launch the backend serving one audio model.

    Dispatches on the model's task: transcription is llama-server, speech is
    whichever TTS engine the entry names. Every backend answers `/health` with
    `{"status": "ok"}`, so everything downstream — the readiness gate, the
    router lifecycle, the proxy — is identical once the plan is built.

    The TTS registry is imported here rather than at module scope because the
    engines import this module for ``LaunchPlan`` and ``build_env``. Keeping
    the edge inside the function is what lets an engine reuse the launcher
    instead of reimplementing the environment it needs.
    """
    if model.task == "tts":
        from arc_llama.tts import require_engine

        engine = require_engine(model.engine)
        plan = engine.build_plan(cfg, model, gpu, host=host)
        if plan.health_timeout is None:
            plan.health_timeout = engine.health_timeout
        return plan
    return build_llamacpp_audio_plan(cfg, model, gpu, host=host)


def build_llamacpp_audio_plan(
    cfg: Config, model: AudioModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
) -> LaunchPlan:
    """Launch plan for an ASR model served by llama-server.

    This is an ordinary llama-server invocation with an audio projector
    attached, so it goes through the same `build_env` the LLMs use: the arch
    SYCL profile, the stripped known-bad vars, the device selector. That is
    the whole point of transcribing on llama.cpp here — it is the only ASR
    runtime with a SYCL build, so it is the only one that reaches an Arc card
    the same way the LLMs do.

    Deliberately launched without a `--config`/router setup: llama.cpp's
    multi-model router mode is reported to 500 on this endpoint
    (ggml-org/llama.cpp#21906), and arc-llama is the router anyway.
    """
    recipe = model.audio_recipe()
    if not recipe.mmproj:
        raise RuntimeError(
            f"audio model {model.name!r} has no 'mmproj' in its recipe. "
            "llama.cpp keeps the audio projector in a separate GGUF "
            "(mmproj-*.gguf, published beside the weights); without it "
            "llama-server loads the model as a plain text LLM and "
            "transcription returns confident nonsense."
        )
    mmproj_path = Path(recipe.mmproj).expanduser()
    if not mmproj_path.exists():
        raise RuntimeError(
            f"audio model {model.name!r}: mmproj not found at {mmproj_path}"
        )
    binary = resolve_binary(cfg.paths.llama_server)
    if binary is None:
        raise RuntimeError(
            f"llama-server not found at {cfg.paths.llama_server!r}. Run "
            "`arc-llama install-runtime` or set paths.llama_server."
        )

    arch = Arch(gpu.arch) if gpu.arch else Arch.UNKNOWN
    profile = profile_for(arch)
    env = build_env(
        profile,
        gpu,
        llama_server=binary,
        oneapi_setvars=getattr(cfg.paths, "oneapi_setvars", None),
    )

    caps = probe_server_caps(binary)
    if caps.probed and not caps.supports_mmproj:
        raise RuntimeError(
            f"{binary} has no --mmproj, so it was built without multimodal "
            "(mtmd) support and cannot serve ASR. Install a newer build with "
            "`arc-llama install-runtime`."
        )

    # -c is not optional here. llama-server's default is 0, meaning "whatever
    # the GGUF was trained for", and Qwen3-ASR advertises 65536 — a ~7 GB KV
    # cache in front of 2 GB of weights, allocated on the first request and
    # held for the life of the process. -np 1 keeps that budget from being
    # split into auto-chosen slots as well.
    argv: list[str] = [
        binary,
        "-m", str(Path(model.path).expanduser()),
        "--mmproj", str(mmproj_path),
        "--host", host,
        "--port", str(model.port),
        "-ngl", str(recipe.n_gpu_layers),
        "-c", str(recipe.ctx),
        "-np", "1",
        "-ctk", recipe.cache_type_k,
        "-ctv", recipe.cache_type_v,
    ]
    argv.extend(recipe.extra_flags)

    backend_url = f"http://{host}:{model.port}"
    return LaunchPlan(
        argv=argv,
        env=env,
        backend_url=backend_url,
        health_url=f"{backend_url}/health",
    )


class LlamaServer:
    """One llama-server subprocess. Lifecycle: start → wait_ready → stop.

    Despite the name, nothing here is llama.cpp-specific: it drives whatever
    ``plan.argv`` says and gates on ``plan.health_url`` returning
    ``{"status": "ok"}``. Every audio backend answers /health in that shape --
    it is the one thing a TTS engine must implement -- so they reuse this class
    as-is, including the process-group kill that gets the VRAM back.
    """

    def __init__(self, plan: LaunchPlan, name: str = "llama-server"):
        self.plan = plan
        self.name = name
        self.process: subprocess.Popen[bytes] | None = None
        self.started_at: float | None = None
        # Cached health state: True only after wait_ready() has seen a healthy
        # /health response. is_running means "subprocess alive"; a cold start
        # leaves the port unbound for tens of seconds, so callers that forward
        # traffic must check this, not is_running. Cleared on start and stop.
        self.ready: bool = False
        self._log_file: Any = None  # file handle opened in start(), closed in stop()
        self._log_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, log_dir: Path | None = None) -> None:
        if self.is_running:
            log.debug("[%s] already running, pid=%s", self.name, self.process.pid)  # type: ignore[union-attr]
            return
        self.ready = False
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        self._log_path = None
        log_file: Any = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{self.name}.log"
            _rotate_log(log_path)
            self._log_path = log_path
            log_file = open(log_path, "ab")
            stdout = log_file
            stderr = subprocess.STDOUT
        log.info("[%s] starting: %s", self.name, " ".join(self.plan.argv))
        popen_kwargs: dict[str, Any] = {}
        if _IS_WINDOWS:
            # A new process group lets us terminate the whole subtree cleanly
            # without Unix-specific killpg. The constant is only defined on
            # Windows; the getattr guard keeps tests on Linux valid.
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["preexec_fn"] = _preexec_isolate_and_pdeathsig
        try:
            self.process = subprocess.Popen(
                self.plan.argv,
                env=self.plan.env,
                stdout=stdout,
                stderr=stderr,
                **popen_kwargs,
            )
        except Exception:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            self._log_path = None
            raise
        self._log_file = log_file
        self.started_at = time.time()

    async def wait_ready(self, timeout: float | None = None) -> bool:
        timeout = timeout if timeout is not None else (
            self.plan.health_timeout
            if self.plan.health_timeout is not None
            else DEFAULT_HEALTH_TIMEOUT
        )
        deadline = time.time() + timeout
        last_progress = time.time()
        progress_interval = 15.0  # log every 15 s so the terminal isn't silent
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                while time.time() < deadline:
                    if not self.is_running:
                        log.warning("[%s] process exited before becoming healthy", self.name)
                        return False
                    try:
                        r = await client.get(self.plan.health_url)
                        if r.status_code == 200 and r.json().get("status") == "ok":
                            elapsed = time.time() - self.started_at if self.started_at else 0
                            log.info("[%s] ready after %.1fs", self.name, elapsed)
                            self.ready = True
                            return True
                    except Exception:
                        pass
                    now = time.time()
                    if now - last_progress >= progress_interval:
                        remaining = max(0, deadline - now)
                        log.info(
                            "[%s] still loading... %.0fs elapsed, %.0fs budget remaining",
                            self.name, timeout - remaining, remaining,
                        )
                        last_progress = now
                    await asyncio.sleep(HEALTH_POLL_INTERVAL)
        except asyncio.CancelledError:
            # If the waiter is cancelled (router timeout, client disconnect,
            # shutdown) the llama-server child is still holding GPU VRAM.
            # Stop it before re-raising so we don't leak a process that blocks
            # every subsequent model load on a single-GPU box.
            log.info("[%s] wait_ready cancelled; stopping subprocess", self.name)
            try:
                # Shielded: CancelledError is a BaseException, so an unshielded
                # `await self.astop()` that is cancelled again (loop shutdown)
                # would propagate straight past the `except Exception` below and
                # leave the child alive — precisely the leak this handler exists
                # to prevent.
                await asyncio.shield(self.astop())
            except asyncio.CancelledError:
                # Cancelled while cleaning up. Fall back to the blocking stop:
                # briefly stalling the loop during teardown is much cheaper than
                # orphaning a process that holds the GPU.
                self.stop()
            except Exception:
                log.exception("[%s] failed to stop subprocess during cancellation", self.name)
            raise
        log.warning("[%s] health-check timed out after %.0fs", self.name, timeout)
        return False

    def tail_log(self, lines: int = 50) -> str:
        """Return the last *lines* of the llama-server log, if any."""
        if self._log_path is None:
            return ""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        all_lines = text.splitlines()
        return "\n".join(all_lines[-lines:])

    def stop(self, drain_seconds: float = 3.0) -> None:
        self.ready = False
        if not self.is_running:
            return
        proc = self.process
        assert proc is not None
        log.info("[%s] stopping pid=%s", self.name, proc.pid)
        if _IS_WINDOWS:
            # proc.terminate()/kill() both just call TerminateProcess on Windows —
            # there's no graceful/forceful distinction, and neither touches child
            # processes. CTRL_BREAK_EVENT goes to the whole CREATE_NEW_PROCESS_GROUP
            # (the closest equivalent to SIGTERM here); taskkill /T kills the whole
            # subtree, mirroring killpg on the Linux side below.
            try:
                proc.send_signal(_CTRL_BREAK_EVENT)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                log.warning(
                    "[%s] CTRL_BREAK timed out, force-killing process tree", self.name
                )
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=drain_seconds,
                )
                try:
                    proc.wait(timeout=drain_seconds)
                except subprocess.TimeoutExpired:
                    pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)  # type: ignore[attr-defined]
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=drain_seconds)
            except subprocess.TimeoutExpired:
                log.warning("[%s] SIGTERM timed out, sending SIGKILL", self.name)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=drain_seconds)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        self.started_at = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    async def astop(self, drain_seconds: float = 3.0) -> None:
        """Async version of stop() for callers running on the event loop.

        stop() performs blocking waits and subprocess calls; running it
        directly from async code stalls the loop while a stuck child drains.
        This wrapper offloads the blocking work to a thread.
        """
        await asyncio.to_thread(self.stop, drain_seconds)
