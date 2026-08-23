"""The TTS engine interface and its registry.

A TTS engine is the pairing of *how to launch a backend* with *how to talk to
it*, and nothing else — everything around it (the router lifecycle, eviction,
the VRAM fit guard, the OpenAI endpoint, in-flight accounting) is engine
agnostic and already exists. So plugging in a new one means implementing the
four methods below and calling :func:`register_engine`; no other module needs
to learn its name.

The two ends of that contract are deliberately narrow:

* ``build_plan`` returns the same :class:`~arc_llama.launcher.LaunchPlan` the
  LLM path uses, so a TTS backend is started, health-gated, drained and evicted
  by exactly the code that runs llama-server.
* ``build_payload`` translates an OpenAI ``/v1/audio/speech`` body into whatever
  the backend wants on ``speech_path``. An engine that already speaks OpenAI
  returns it nearly unchanged; one that does not (a `llama-tts` binary, say)
  does its mapping here instead of leaking a second request shape into the
  server.
"""
from __future__ import annotations

import logging
import stat
from pathlib import Path
from typing import Any

from arc_llama.config import AudioModelConfig, Config, GPUConfig
from arc_llama.launcher import LaunchPlan

log = logging.getLogger("arc_llama.tts")

# A speech backend is not llama-server: it may import torch, resolve a HF repo
# and download several GB before it answers /health. The LLM budget (120 s) is
# sized for SYCL JIT and is not enough for a first run, and a health timeout
# that expires mid-download leaves the user with a model that never starts and
# no indication that it was nearly there.
DEFAULT_TTS_HEALTH_TIMEOUT = 900.0

_VRAM_OVERHEAD_MB = 1024
"""Slack added to a measured-on-disk footprint.

A torch runtime holds more than the weights — CUDA/SYCL context, the audio
tokenizer, activation buffers — and the fit guard's job is to refuse a load
that would OOM, so the estimate errs high.
"""


class TTSEngine:
    """One way of serving ``/v1/audio/speech``.

    Subclasses are instantiated once and registered by name; the instance holds
    no per-model state, so it is safe to share across every model that names it.
    """

    name: str = ""
    description: str = ""
    speech_path: str = "/v1/audio/speech"
    """Path on the backend that answers a speech request."""
    health_timeout: float = DEFAULT_TTS_HEALTH_TIMEOUT
    accepts_remote_path: bool = False
    """Whether ``model.path`` may name something that is not on disk yet.

    True for an engine that resolves a Hugging Face repo id itself, which is
    how OmniVoice is normally addressed. Registration skips its
    path-must-exist check for those.
    """

    # -- registration -------------------------------------------------

    def validate(self, model: AudioModelConfig) -> None:
        """Raise ValueError/FileNotFoundError if this entry cannot be served.

        Called at `arc-llama audio add` time, where the user is standing right
        there and can fix it, rather than at first request.
        """

    # -- lifecycle ----------------------------------------------------

    def build_plan(
        self, cfg: Config, model: AudioModelConfig, gpu: GPUConfig, host: str = "127.0.0.1"
    ) -> LaunchPlan:
        raise NotImplementedError

    def estimate_vram_mb(self, model: AudioModelConfig) -> int | None:
        """Rough resident footprint, or None when it cannot be measured.

        None makes the fit guard skip this model rather than guess, which is
        how it already treats an unmeasurable co-resident.
        """
        if model.vram_mb:
            return int(model.vram_mb)
        size = _path_size(Path(model.path).expanduser())
        if not size:
            return None
        return size // (1024 * 1024) + _VRAM_OVERHEAD_MB

    # -- requests -----------------------------------------------------

    def build_payload(self, model: AudioModelConfig, body: dict[str, Any]) -> dict[str, Any]:
        """Map an OpenAI speech body onto the backend's request body."""
        return dict(body)

    # -- diagnostics --------------------------------------------------

    def preflight(self, cfg: Config, model: AudioModelConfig) -> list[str]:
        """Problems `arc-llama doctor` should report, as plain sentences."""
        return []


def _path_size(path: Path) -> int:
    """Total bytes at *path*, counting each underlying file once.

    The de-duplication is not defensive tidiness, it is the whole point. A
    Hugging Face cache stores every weight once in `blobs/` and exposes it
    through `snapshots/<rev>/` as a symlink; `Path.stat()` follows symlinks, so
    the naive walk counts every byte of the model twice and the fit guard
    refuses loads that would comfortably fit. Keying on (device, inode) also
    collapses revisions that share an unchanged blob, and any hardlinked
    layout, for the same reason.
    """
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    seen: set[tuple[int, int]] = set()
    for entry in path.rglob("*"):
        try:
            info = entry.stat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        key = (info.st_dev, info.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += info.st_size
    return total


_ENGINES: dict[str, TTSEngine] = {}


def register_engine(engine: TTSEngine) -> TTSEngine:
    """Add *engine* to the registry, keyed by its name."""
    if not engine.name:
        raise ValueError("a TTS engine must have a name")
    _ENGINES[engine.name] = engine
    return engine


def get_engine(name: str) -> TTSEngine | None:
    return _ENGINES.get(name)


def require_engine(name: str) -> TTSEngine:
    """Look up *name*, or raise with the list of engines that do exist."""
    engine = _ENGINES.get(name)
    if engine is None:
        known = ", ".join(engine_names()) or "(none)"
        raise ValueError(f"unknown TTS engine {name!r}; registered engines: {known}")
    return engine


def engine_names() -> list[str]:
    return sorted(_ENGINES)


def engines() -> list[TTSEngine]:
    return [_ENGINES[name] for name in engine_names()]
