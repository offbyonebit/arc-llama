"""Backend-neutral adapter seam for image generation backends.

The service never talks to a concrete backend directly: ``POST
/v1/images/generations`` resolves the requested model to an adapter through
this seam and asks it to render the request. Two adapters ship today:

- ``fake`` — deterministic, dependency-free output for end-to-end tests and
  wiring checks (the only adapter enabled by default);
- ``unavailable`` — always raises :class:`BackendUnavailableError`, to verify
  and demonstrate the failure surface without starting any process;
- ``comfyui`` — a real adapter rendering through a local ComfyUI server
  (``arc_llama_vision.comfyui``; selected with ``ARV_BACKEND=comfyui``).
  It is resolved lazily by :func:`create_backend` so importing the seam
  stays cheap; the adapter module itself imports only the stdlib.

Plugging a real backend adapter in
---------------------------------

``FakeImageBackend`` is the reference implementation. A real adapter:

1. **Subclasses**: inherit from :class:`ImageBackend` (or construct a class
   exposing the same members, like an ``arc_llama`` plugin).
2. **Declares names**: assign ``backend_name`` (configuration key) and
   ``models`` (stable model identifiers whose listings will include image
   modality metadata).
3. **Renders**: implement :meth:`ImageBackend.generate` producing
   :class:`GeneratedImage` records for each call. Accept the parsed request
   fields (size, count, format options), which the seam parses centrally so
   adapters need no HTTP-aware code.
4. **Registers**: add the module path to :data:`BACKEND_REGISTRY` (or its own
   entry-point group), then launch with::

       ARV_BACKEND=<name> ARV_BACKEND_OPTIONS='{...}' \\
           python -m arc_llama_vision --host 127.0.0.1 --port 11440

Heavy dependencies (e.g. ``torch``, ``diffusers``) live inside real adapters'
own optional-dependency groups and are imported lazily at :meth:`startup` /
first use, never at seam import time — mirroring the doc's plugin guidance.
"""

from __future__ import annotations

import re
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_SERVICE_NAME = "vision-companion"


class CapabilityError(Exception):
    """The configured backend cannot satisfy a request.

    Raised for unknown backends and unsupported sizes/formats/models. The
    endpoint maps it to HTTP 400 with the stable JSON error schema.
    """


class BackendUnavailableError(Exception):
    """The selected backend is not usable right now (process down, no GPU...).

    The endpoint maps this to HTTP 503, mirroring the audio companion doc.
    """


@dataclass
class ModelInfo:
    """A stable image model identity, surfaced in GET /v1/models."""

    id: str
    backend: str  # adapter name, shown as metadata
    owned_by: str = _SERVICE_NAME
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedImage:
    """One rendered image in an OpenAI-compatible generations response."""

    # OpenAI-compatible clients accept both; PNG and JPEG are also standard.
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    # Non-standard but useful for diagnostics; omitted when None so the JSON
    # response keeps an OpenAI-identical shape for the standard fields.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_response_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.b64_json is not None:
            out["b64_json"] = self.b64_json
        if self.url is not None:
            out["url"] = self.url
        if self.revised_prompt is not None:
            out["revised_prompt"] = self.revised_prompt
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class ImageBackend(ABC):
    """The adapter seam. One adapter instance serves one backend process.

    Concrete adapters must define ``backend_name`` and ``models`` and
    implement :meth:`generate`. ``startup``/``shutdown`` may claim or release
    heavy resources (model processes, GPU memory, HTTP pools).

    The endpoint layer validates request fields (size syntax and bounds,
    batch limits, prompt length) before calling :meth:`generate`, so adapters
    translate options, not schemas. Adapters may still call
    :func:`validate_size` defensively for direct (non-HTTP) use.
    """

    backend_name: str = "abstract"
    models: list[ModelInfo] = []

    def __init__(self, options: dict[str, Any] | None = None):
        """Store the per-adapter options dict from ARV_BACKEND_OPTIONS.

        Built-in adapters ignore options; real adapters read their own
        documented keys here or in ``startup``.
        """
        self.options: dict[str, Any] = dict(options or {})

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: ModelInfo,
        *,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        """Render ``n`` images for ``prompt`` using ``model``.

        ``size`` has already passed endpoint validation but adapters should
        still reject sizes *unsupported by the backend itself* (e.g. only
        square outputs) with :class:`CapabilityError`.
        """

    async def startup(self) -> None:  # noqa: B027 - optional hook, like Plugin.startup
        """Claim heavy resources. Optional; failures should raise."""

    async def shutdown(self) -> None:  # noqa: B027 - optional hook, like Plugin.shutdown
        """Release heavy resources. Optional."""


# Valid OpenAI-style sizes: "WIDTHxHEIGHT" with each dimension in [64, 4096].
_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
MIN_DIM = 64
MAX_DIM = 4096


def validate_size(size: str | None) -> tuple[int, int] | None:
    """Return (width, height) for a valid size string, else raise CapabilityError."""
    if size is None:
        return None
    m = _SIZE_RE.match(size.strip())
    if not m:
        raise CapabilityError(
            f"Invalid size {size!r}: expected 'WIDTHxHEIGHT' with integer pixels."
        )
    w, h = int(m.group(1)), int(m.group(2))
    if not (MIN_DIM <= w <= MAX_DIM) or not (MIN_DIM <= h <= MAX_DIM):
        raise CapabilityError(
            f"Invalid size {size!r}: dimensions must be between {MIN_DIM} and {MAX_DIM}."
        )
    return (w, h)


def _fake_png_bytes(width: int, height: int, seed: int) -> bytes:
    """Build a tiny valid PNG whose pixel data encodes the seed deterministically.

    Uses only stdlib (struct-free) PNG chunking plus zlib, so the scaffold
    needs no PIL/numpy. The decoded image is a single-color 8-bit RGB raster
    whose RGB channels are bytes of the 32-bit CRC of the prompt — visually
    deterministic output for end-to-end tests.
    """
    import struct

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    r = (seed >> 16) & 0xFF
    g = (seed >> 8) & 0xFF
    b = seed & 0xFF
    row = bytes((r, g, b)) * width
    raw = b"\x00" + row  # filter byte 0 per scanline
    scanlines = raw * height

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolor
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


class FakeImageBackend(ImageBackend):
    """Deterministic, dependency-free backend for end-to-end tests.

    Output is a small valid PNG whose color derives from the prompt CRC, so
    repeated calls with identical input produce byte-identical images while
    distinct prompts remain distinguishable. No model weights are used and
    nothing leaves the process.
    """

    backend_name = "fake"
    models = [
        ModelInfo(
            id="arc-vision-diffusion",
            backend="fake",
            metadata={"modality": "image->image", "description": "deterministic test backend"},
        ),
        ModelInfo(
            id="arc-vision-thumbnail",
            backend="fake",
            metadata={"modality": "image->image", "description": "deterministic test backend"},
        ),
    ]

    async def generate(
        self,
        prompt: str,
        model: ModelInfo,
        *,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        dims = validate_size(size) or (64, 64)
        prompt_crc = zlib.crc32(prompt.encode("utf-8")) & 0xFFFFFFFF

        images: list[GeneratedImage] = []
        for i in range(n):
            # Fold the per-image index into the seed so batch entries differ.
            seed = (prompt_crc ^ (i * 0x9E3779B1)) & 0xFFFFFFFF
            png = _fake_png_bytes(dims[0], dims[1], seed)
            images.append(
                GeneratedImage(
                    b64_json=_b64encode(png),
                    metadata={
                        "backend": self.backend_name,
                        "format": "png",
                        "size": f"{dims[0]}x{dims[1]}",
                        "seed": seed,
                    },
                )
            )
        return images


def _b64encode(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


class UnavailableBackend(ImageBackend):
    """Always raises BackendUnavailableError; used to exercise the 503 path.

    Declares the standard fake model ids so requests resolve to a model and
    then fail at render time — the same shape as a backend whose model list is
    configured but whose process is down.
    """

    backend_name = "unavailable"
    models = [
        ModelInfo(
            id="arc-vision-diffusion",
            backend="unavailable",
            metadata={"modality": "image->image", "description": "outage simulator"},
        ),
    ]

    async def generate(
        self,
        prompt: str,
        model: ModelInfo,
        *,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        raise BackendUnavailableError(
            "backend 'unavailable' is unable to accept generation requests"
        )


# Registry maps canonical name -> factory. Real adapters can be added to
# BACKEND_REGISTRY via a matching entry point in their package metadata, then
# enabled through ARV_BACKEND=<name>. The comfyui adapter is resolved lazily:
# keep it as a name-only marker here so importing the module never pays for
# (or requires) anything beyond the stdlib seam, per the plugin guidance.
BACKEND_REGISTRY: dict[str, type[ImageBackend]] = {
    "fake": FakeImageBackend,
    "unavailable": UnavailableBackend,
}


def create_backend(name: str, options: dict[str, Any] | None = None) -> ImageBackend:
    """Instantiate the named adapter, raising CapabilityError for unknown names.

    ``options`` is the per-adapter settings bag from ARV_BACKEND_OPTIONS;
    each adapter documents its own keys. No silent fallback to "fake" on
    unknown names — that would mask configuration mistakes and route real
    traffic to fake responses. The ``comfyui`` adapter is a real backend
    and is loaded lazily from :mod:`arc_llama_vision.comfyui` on first use.
    """
    if name == "comfyui":
        from arc_llama_vision.comfyui import ComfyUIBackend

        cls: type[ImageBackend] = ComfyUIBackend
    else:
        cls = BACKEND_REGISTRY.get(name)  # type: ignore[assignment]
        if cls is None:
            known = ", ".join(sorted([*BACKEND_REGISTRY, "comfyui"]))
            raise CapabilityError(f"Unknown image backend {name!r}. Known backends: {known}")
    return cls(options)
