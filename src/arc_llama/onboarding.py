"""A first-model setup, reusing the normal installer and downloader.

Nothing downloads on a read. A single explicit start authorizes runtime
preparation, the named starter download and loading it; no prompt is sent.
Pausing finishes the current step rather than abandoning a publishing thread.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from arc_llama.arch import Backend
from arc_llama.binary import detect_backends
from arc_llama.config import Config, GPUConfig
from arc_llama.failures import StartupFailureError
from arc_llama.model_library import (
    DownloadJob,
    DownloadManager,
    fit_badge,
)
from arc_llama.starter_discovery import (
    Starter,
    check_runtime,
    discover,
    source,
    validate,
)

ACTIVE = {"checking", "runtime", "downloading", "registering", "loading"}


def enabled_gpu(cfg: Config) -> GPUConfig | None:
    return next((g for g in cfg.gpus if g.enabled), None)


def runtime_path(cfg: Config) -> Path | None:
    path = Path(cfg.paths.llama_server).expanduser()
    if path.is_file():
        return path
    found = shutil.which(cfg.paths.llama_server)
    return Path(found) if found else None


class FirstAnswerSetup:
    def __init__(self, cfg: Config, config_path: Path, router: Any) -> None:
        self.cfg, self.config_path, self.router = cfg, config_path, router
        self.phase = "idle"
        self.error: str | None = None
        self.model: str | None = None
        self.gpu_slot: str | None = None
        self.starter: Starter | None = None
        self.suggestion: Starter | None = None
        self.discovery_status = "idle"
        self.discovery_error: str | None = None
        self.discovery_task: asyncio.Task[None] | None = None
        self.discovery_context: tuple[Any, ...] | None = None
        self.discovery_at = 0.0
        self.download: DownloadJob | None = None
        self.runtime_done = 0
        self.runtime_total = 0
        self.pause_requested = False
        self.task: asyncio.Task[None] | None = None
        self.downloads = DownloadManager(cfg, self._register)

    async def _register(self, path: Path) -> list[str]:
        from arc_llama.models import register_discovered

        gpu = self._check_gpu()
        previous = list(self.cfg.models)
        added = register_discovered(self.cfg, [path], gpu_pci_slot=gpu.pci_slot)
        # Registration runs against the same enabled GPU used for the estimate.
        # A modest context keeps the first load quick; users can change it later.
        for model in added:
            model.recipe["ctx"] = min(model.recipe.get("ctx", 4096), 4096)
            model.recipe["parallel"] = 1
        if added:
            try:
                self.cfg.save(self.config_path)
            except OSError:
                self.cfg.models = previous
                raise
            self.router._build_servers()
        names = [m.name for m in added]
        if not names:
            names = [
                m.name
                for m in self.cfg.models
                if Path(m.path).resolve() == path.resolve()
            ]
        return names

    def _context(self) -> tuple[Any, ...]:
        gpu = enabled_gpu(self.cfg)
        return (gpu.pci_slot, gpu.vram_mb, gpu.backend) if gpu else (None,)

    def refresh(self, *, force: bool = False) -> None:
        # Never replace an accepted choice during an active/paused setup.
        if self.starter or self.cfg.models:
            return
        context = self._context()
        changed = context != self.discovery_context
        if not changed and self.discovery_task and not self.discovery_task.done():
            return
        if not changed and not force and time.monotonic() - self.discovery_at < 300:
            return
        if self.discovery_task and not self.discovery_task.done():
            self.discovery_task.cancel()
        self.discovery_context = context
        self.suggestion = None
        self.discovery_error = None
        gpu = enabled_gpu(self.cfg)
        if not gpu or not gpu.vram_mb:
            self.discovery_status = "hardware"
            self.discovery_at = time.monotonic()
            return
        self.discovery_status = "searching"
        self.discovery_task = asyncio.create_task(self._discover(context, gpu.vram_mb))

    async def _discover(self, context: tuple[Any, ...], vram_mb: int) -> None:
        try:
            suggestion = await discover(vram_mb)
            if context != self._context() or self.starter:
                return
            self.suggestion = suggestion
            self.discovery_status = "ready" if suggestion else "no_match"
        except Exception:
            if context == self._context():
                self.discovery_status = "unavailable"
                self.discovery_error = "The public model catalog could not be checked. Check your connection and retry, or use Discover."
        finally:
            if context == self._context():
                self.discovery_at = time.monotonic()

    def public(self) -> dict[str, Any]:
        gpu = enabled_gpu(self.cfg)
        suggestion = self.starter or (
            self.suggestion if self.discovery_context == self._context() else None
        )
        return {
            "phase": self.phase,
            "discovery": {
                "status": self.discovery_status,
                "error": self.discovery_error,
            },
            "error": self.error,
            "model": self.model,
            "pause_requested": self.pause_requested,
            "gpu": {
                "name": gpu.name,
                "vram_mb": gpu.vram_mb,
                "pci_slot": gpu.pci_slot,
                "backend": gpu.backend,
            }
            if gpu
            else None,
            "starter": {
                **asdict(suggestion),
                "source": source(suggestion),
                "fit": "estimated",
                "compatibility": "Architecture checked before weights download; GPU execution remains unverified.",
            }
            if suggestion
            else None,
            "runtime_ready": runtime_path(self.cfg) is not None,
            "runtime_bytes_done": self.runtime_done,
            "runtime_bytes_total": self.runtime_total,
            "download": self.download.public() if self.download else None,
            "chat_url": f"/chat?model={quote(self.model, safe='')}&welcome=1"
            if self.model
            else None,
        }

    def start(self, starter_id: str) -> None:
        if self.task is not None and not self.task.done():
            if self.starter and self.starter.id == starter_id:
                return  # A double click or second tab joins this setup.
            raise ValueError("A setup is already running. Wait for it to finish.")
        suggestion = self.starter or (
            self.suggestion if self.discovery_context == self._context() else None
        )
        if (
            suggestion is None
            or suggestion.id != starter_id
            or (not self.starter and time.monotonic() - self.discovery_at >= 300)
        ):
            raise ValueError(
                "This recommendation changed. Refresh and choose a model that fits."
            )
        if (
            self.cfg.models
            and not self.model
            and not (self.download and self.download.registered)
        ):
            raise ValueError(
                "Models are already registered. Choose one from Your models instead."
            )
        if any(name != self.model for name in self.router.running_models()):
            raise ValueError(
                "Another model is running. Open Chat with it, or stop it before setup."
            )
        self.starter = suggestion
        gpu = enabled_gpu(self.cfg)
        if self.gpu_slot is None and gpu is not None:
            self.gpu_slot = gpu.pci_slot
        self._check_gpu()
        self.pause_requested = False
        self.error = None
        self.phase = "checking"
        self.task = asyncio.create_task(self._run())

    def _check_gpu(self) -> GPUConfig:
        gpu = enabled_gpu(self.cfg)
        if (
            gpu is None
            or gpu.pci_slot != self.gpu_slot
            or self.starter is None
            or fit_badge(self.starter.size_mb, gpu.vram_mb) != "fits"
        ):
            raise ValueError(
                "The selected GPU or its available capacity changed. Review your hardware and choose another model."
            )
        return gpu

    def pause(self) -> None:
        if self.phase in ACTIVE:
            self.pause_requested = True

    def _paused(self) -> bool:
        if self.pause_requested:
            self.phase = "paused"
            return True
        return False

    async def _run(self) -> None:
        from arc_llama.runtime import install_runtime

        try:
            assert self.starter is not None
            starter = self.starter
            if not self.model:
                if self.download and self.download.status == "done":
                    self.model = next(iter(self.download.registered), None)
                else:
                    gpu = enabled_gpu(self.cfg)
                    if gpu is None:
                        raise ValueError(
                            "No enabled GPU is configured. Check your hardware and retry."
                        )
                    await validate(starter, gpu.vram_mb or 0)
                    if self._paused():
                        return
                    self._check_gpu()
                    current = runtime_path(self.cfg)
                    if current is not None:
                        available = await asyncio.to_thread(detect_backends, current)
                        if not available:
                            raise ValueError(
                                "The installed runtime has no recognized Vulkan or SYCL backend. Check System or install a supported runtime before continuing."
                            )
                        if Backend(gpu.backend) not in available:
                            previous_backend = gpu.backend
                            gpu.backend = (
                                Backend.SYCL.value
                                if Backend.SYCL in available
                                else Backend.VULKAN.value
                            )
                            try:
                                self.cfg.save(self.config_path)
                            except OSError:
                                gpu.backend = previous_backend
                                raise
                    else:
                        if self.cfg.models:
                            raise ValueError(
                                "Models were registered during setup. Review them before changing the runtime."
                            )
                        self.phase = "runtime"

                        def progress(done: int, total: int) -> None:
                            self.runtime_done, self.runtime_total = done, total

                        await asyncio.to_thread(
                            install_runtime,
                            backend="vulkan",
                            cfg=self.cfg,
                            config_path=self.config_path,
                            on_progress=progress,
                        )
                    if self._paused():
                        return
                    self._check_gpu()
                    self.phase = "checking"
                    await check_runtime(starter, self.cfg.paths.llama_server)
                    if self._paused():
                        return
                    self._check_gpu()
                    if self.download is None or self.download.status == "error":
                        self.download = self.downloads.submit(
                            starter.repo,
                            starter.file,
                            starter.size_bytes,
                            revision=starter.revision,
                        )
                    self.phase = "downloading"
                    while self.download.status not in {"done", "error"}:
                        if self.download.status == "registering":
                            self.phase = "registering"
                        await asyncio.sleep(0.2)
                    if self.download.status == "error":
                        raise ValueError(
                            self.download.error
                            or "The model download did not finish. Check your connection and retry."
                        )
                    self.model = next(iter(self.download.registered), None)
            if self._paused():
                return
            if not self.model:
                raise ValueError(
                    "The model was downloaded but could not be registered. Scan for models and retry."
                )
            # Never evict another person's active model as a side effect of setup.
            if any(name != self.model for name in self.router.running_models()):
                raise ValueError(
                    "Another model is now running. Stop it before continuing setup, or use it in Chat."
                )
            self._check_gpu()
            self.phase = "loading"
            _, server = await self.router.ensure_active(
                self.model, allow_eviction=False
            )
            if not server.is_running or not server.ready:
                raise ValueError(
                    "The runtime did not become ready. Check System and retry."
                )
            self.phase = "paused" if self._paused() else "ready"
        except Exception as exc:
            self.error = (
                f"{exc.message} {exc.action}"
                if isinstance(exc, StartupFailureError)
                else str(exc)
            )[:500]
            self.phase = "error"

    async def shutdown(self) -> None:
        # Let the normal downloader finish its current integrity/publication
        # step; cancelling a thread does not stop its filesystem writes.
        self.pause_requested = True
        if self.discovery_task:
            self.discovery_task.cancel()
            await asyncio.gather(self.discovery_task, return_exceptions=True)
        if self.task is not None:
            await self.task
        await self.downloads.shutdown()
