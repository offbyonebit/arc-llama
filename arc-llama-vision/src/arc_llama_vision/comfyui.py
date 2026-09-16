"""ComfyUI adapter: render images through a local ComfyUI server over HTTP.

Selected with ``ARV_BACKEND=comfyui`` and configured through the
``ARV_BACKEND_OPTIONS`` JSON bag::

    ARV_BACKEND=comfyui \\
    ARV_BACKEND_OPTIONS='{
        "base_url": "http://127.0.0.1:8190",
        "model": "arc-vision-flux2-klein-9b-uncensored",
        "unet_gguf": "flux2-klein-9b-Q4_K_M.gguf",
        "clip_gguf": "qwen3-8b-Q2_K-uncensored.gguf",
        "vae": "flux2-vae.safetensors",
        "steps": 20,
        "filename_prefix": "arc-vision"
    }' \\
        python -m arc_llama_vision

Only stdlib HTTP/JSON/base64 is used (``urllib.request`` through
``asyncio.to_thread``), so the companion stays free of heavyweight ML
dependencies — ComfyUI itself owns torch and the model weights.

The default graph is a minimal API-format FLUX.2 Klein text-to-image
workflow built in code (no template files are read from disk):

- ``UnetLoaderGGUF``         → MODEL (ComfyUI-GGUF custom node)
- ``CLIPLoaderGGUF``         → CLIP  (ComfyUI-GGUF custom node, type ``flux2``)
- ``VAELoader``              → VAE
- ``CLIPTextEncode`` x2      → CONDITIONING (positive / empty negative)
- ``EmptyFlux2LatentImage``  → LATENT (blank latent at width x height)
- ``KSamplerSelect``         → SAMPLER
- ``Flux2Scheduler``         → SIGMAS (Flux2 noise schedule)
- ``CFGGuider``              → GUIDER
- ``RandomNoise``            → NOISE
- ``SamplerCustomAdvanced``  → LATENT (advanced sampling)
- ``VAEDecode``              → IMAGE
- ``SaveImage``              → written into ComfyUI's ``output/`` directory

Rendering submits the workflow to ``POST /prompt``, polls
``GET /history/{prompt_id}`` until completion (bounded by ``poll_timeout``),
fetches the first image through ``GET /view``, and returns it as a
``b64_json`` :class:`GeneratedImage`. The companion keeps no bytes on disk;
ComfyUI owns all persistence.

Error mapping follows the seam contract: request-shaped problems (workflow
rejected by ComfyUI validation, non-16-multiple sizes, executions errors
for this prompt, non-image payloads) raise :class:`CapabilityError` (→400),
while connectivity problems (server down, poll timeout, HTTP 5xx) raise
:class:`BackendUnavailableError` (→503). A render whose prompt vanishes
from the queue without a history entry (a crashed server mid-render, e.g.
a GPU runtime fault) is an outage too and raises
:class:`BackendUnavailableError` quickly instead of stalling until
``poll_timeout``.

Tested setup note: this adapter was exercised against a ComfyUI instance
running FLUX.2 Klein 9B Q4 (GGUF diffusion model) with a public Q2_K
quantized uncensored Qwen3 text-encoder GGUF, CPU-only, where a 512x512
render at the default 20 steps took about 7 minutes. No GPU support is
claimed or tested here — hardware dispatch is entirely ComfyUI's business.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from arc_llama_vision.backend import (
    BackendUnavailableError,
    CapabilityError,
    GeneratedImage,
    ImageBackend,
    ModelInfo,
    validate_size,
)

# ComfyUI's own default port is 8188; the adapter defaults to 8190 so a
# long-running local ComfyUI can keep 8188 without colliding with the
# companion's tests and examples.
DEFAULT_BASE_URL = "http://127.0.0.1:8190"

DEFAULT_MODEL_ID = "arc-vision-flux2-klein-9b-uncensored"
# The *_gguf / vae defaults are deliberately concrete but must be edited to
# match the filenames actually present in the ComfyUI models directories
# (models/unet_gguf, models/clip_gguf, models/vae).
DEFAULT_UNET_GGUF = "flux2-klein-9b-Q4_K_M.gguf"
DEFAULT_CLIP_GGUF = "qwen3-8b-Q2_K-uncensored.gguf"
DEFAULT_VAE = "flux2-vae.safetensors"
# Defaults mirror upstream ComfyUI's official Flux.2 Klein template:
# KSamplerSelect "euler" + CFGGuider cfg=5 + Flux2Scheduler steps=20.
DEFAULT_SAMPLER = "euler"
DEFAULT_GUIDANCE = 5.0
DEFAULT_STEPS = 20
DEFAULT_FILENAME_PREFIX = "arc-vision"

# Bounded, generous timeouts: submitting is cheap (validation only) while
# the tested CPU-only setup needed ~7 minutes per 512x512 render at 20
# steps, so the poll window defaults to 20 minutes.
DEFAULT_SUBMIT_TIMEOUT = 60.0
DEFAULT_POLL_TIMEOUT = 1200.0
DEFAULT_POLL_INTERVAL = 1.0
_HTTP_TIMEOUT = 15.0

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ARV_BACKEND_OPTIONS keys -> ComfyUIConfig fields (string options).
_STRING_FIELDS = {
    "base_url": "base_url",
    "model": "model_id",
    "unet_gguf": "unet_gguf",
    "clip_gguf": "clip_gguf",
    "vae": "vae",
    "sampler": "sampler",
    "filename_prefix": "filename_prefix",
}


@dataclass
class ComfyUIConfig:
    """Validated per-adapter configuration, merged from ARV_BACKEND_OPTIONS.

    Unknown keys are ignored so one options bag can carry future knobs
    without breaking older revisions; known keys are strictly validated —
    a typo fails fast at construction with the key named in the error.
    """

    base_url: str = DEFAULT_BASE_URL
    model_id: str = DEFAULT_MODEL_ID
    unet_gguf: str = DEFAULT_UNET_GGUF
    clip_gguf: str = DEFAULT_CLIP_GGUF
    vae: str = DEFAULT_VAE
    sampler: str = DEFAULT_SAMPLER
    guidance: float = DEFAULT_GUIDANCE
    steps: int = DEFAULT_STEPS
    filename_prefix: str = DEFAULT_FILENAME_PREFIX
    submit_timeout: float = DEFAULT_SUBMIT_TIMEOUT
    poll_timeout: float = DEFAULT_POLL_TIMEOUT
    poll_interval: float = DEFAULT_POLL_INTERVAL

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> ComfyUIConfig:
        cfg = cls()
        for key, value in options.items():
            if key in _STRING_FIELDS:
                if not isinstance(value, str) or not value.strip():
                    raise CapabilityError(f"comfyui option {key!r} must be a non-empty string")
                setattr(cfg, _STRING_FIELDS[key], value.strip())
            elif key == "guidance":
                cfg.guidance = _num(key, value, lo=0.0, hi=100.0)
            elif key == "steps":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise CapabilityError(f"comfyui option {key!r} must be an integer")
                if not 1 <= value <= 10000:
                    raise CapabilityError(f"comfyui option {key!r} must be between 1 and 10000")
                cfg.steps = value
            elif key in ("submit_timeout", "poll_timeout", "poll_interval"):
                setattr(cfg, key, _num(key, value, lo=0.01, hi=86400.0))
            # Unknown keys are ignored on purpose (forward compatibility).
        cfg.base_url = cfg.base_url.rstrip("/")
        if not cfg.base_url.startswith(("http://", "https://")):
            raise CapabilityError("comfyui option 'base_url' must start with http:// or https://")
        return cfg


def _num(key: str, value: Any, *, lo: float, hi: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CapabilityError(f"comfyui option {key!r} must be a number")
    if not lo <= float(value) <= hi:
        raise CapabilityError(f"comfyui option {key!r} must be between {lo} and {hi}")
    return float(value)


class ComfyUIBackend(ImageBackend):
    """Real adapter rendering through a local ComfyUI server.

    The adapter talks ComfyUI's HTTP API directly (prompt submission,
    history polling, image retrieval) without websockets or template
    files. All blocking I/O runs on worker threads via ``asyncio.to_thread``
    so the companion's event loop stays responsive during multi-minute
    CPU-only renders, per the seam's no-HTTP-in-the-event-loop rule.
    """

    backend_name = "comfyui"

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(options)
        self.config = ComfyUIConfig.from_options(self.options)
        # Stable per-instance client id, mirroring the ComfyUI UI's behavior.
        self._client_id = f"arc-llama-vision-{id(self):016x}"
        # Advertise the configured model id, not a hardcoded one, so the
        # companion's model list matches what its clients ask for.
        self.models = [
            ModelInfo(
                id=self.config.model_id,
                backend=self.backend_name,
                metadata={
                    "modality": "image->image",
                    "description": (
                        "FLUX.2 Klein text-to-image GGUF workflow via a local ComfyUI server"
                    ),
                },
            ),
        ]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Probe the ComfyUI server so an unreachable one is visible early.

        Failures raise :class:`BackendUnavailableError`; the app logs them,
        marks itself degraded (``/health`` reports it, generation answers
        503), and keeps serving. This is a probe, not a claim on resources
        — the server may appear or disappear later, so every generate
        re-checks connectivity too.

        Runtime-binding diagnostics live in
        :mod:`arc_llama_vision.xpu_runtime` for launcher-side tooling: the
        mixed SYCL/UR binding that crashes GPU renders belongs to the
        ComfyUI process, whose pid is not exposed over HTTP, so it cannot
        be probed from here. The poll loop below detects the *symptom*
        (a prompt that vanishes when the backend crashes) and fails fast.
        """
        await self._probe()

    async def shutdown(self) -> None:
        """Nothing to release — every HTTP request is self-contained."""

    async def _probe(self) -> dict[str, Any]:
        """GET ``/system_stats`` (falling back to ``/prompt``) for health.

        Returns the parsed body on success; raises BackendUnavailableError
        when neither endpoint answers. Connection-level failures (refused,
        DNS, firewall timeouts) all raise :class:`urllib` errors caught here.
        """
        body = await self._get_json(
            f"{self.config.base_url}/system_stats",
            fallback=f"{self.config.base_url}/prompt",
            timeout=self.config.submit_timeout,
        )
        if body is None:
            raise BackendUnavailableError(f"cannot reach ComfyUI server at {self.config.base_url}")
        return body

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model: ModelInfo,
        *,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> list[GeneratedImage]:
        if n != 1:
            # One image per render keeps per-render work bounded; the app
            # layer's default batch cap is 1 anyway. Batch rendering is a
            # capability matter, not availability.
            raise CapabilityError("comfyui backend renders one image per request (n must be 1)")
        dims = validate_size(size) or (1024, 1024)
        width, height = dims
        if width % 16 or height % 16:
            # EmptyFlux2LatentImage and Flux2Scheduler accept multiples of
            # 16 only (the FLUX.2 patch size); reject before submitting.
            raise CapabilityError(
                f"size {size!r} is not supported: the comfyui backend requires "
                f"width and height divisible by 16"
            )
        seed = random.SystemRandom().randrange(0, 2**63)

        workflow = self.build_workflow(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            filename_prefix=self.config.filename_prefix,
        )
        prompt_id = await self._submit(workflow)
        outputs = await self._poll(prompt_id)
        return await self._collect_images(outputs, prompt_id, seed, width, height)

    # ------------------------------------------------------------------
    # workflow graph construction
    # ------------------------------------------------------------------

    def build_workflow(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        filename_prefix: str,
    ) -> dict[str, dict[str, Any]]:
        """Build the minimal API-format FLUX.2 Klein text-to-image workflow.

        ComfyUI's API format maps node ids to ``{"class_type", "inputs"}``
        objects, where an input value of ``[node_id, slot]`` is a link to
        another node's output. All node classes and input names mirror the
        upstream ComfyUI signatures (plus the city96/ComfyUI-GGUF loaders
        for the GGUF diffusion model and text encoder). The wiring follows
        ComfyUI's official Flux.2 Klein template: KSamplerSelect →
        SamplerCustomAdvanced driven by CFGGuider, RandomNoise,
        Flux2Scheduler sigmas, and the EmptyFlux2LatentImage latent.
        """
        return {
            "10": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": self.config.unet_gguf},
            },
            "11": {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {"clip_name": self.config.clip_gguf, "type": "flux2"},
            },
            "12": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": self.config.vae},
            },
            "20": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["11", 0]},
            },
            "21": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["11", 0]},
            },
            "30": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "31": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
            },
            "32": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": self.config.sampler},
            },
            "33": {
                "class_type": "Flux2Scheduler",
                "inputs": {
                    "steps": self.config.steps,
                    "width": width,
                    "height": height,
                },
            },
            "34": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["10", 0],
                    "positive": ["20", 0],
                    "negative": ["21", 0],
                    "cfg": self.config.guidance,
                },
            },
            "35": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["31", 0],
                    "guider": ["34", 0],
                    "sampler": ["32", 0],
                    "sigmas": ["33", 0],
                    "latent_image": ["30", 0],
                },
            },
            "36": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["35", 0], "vae": ["12", 0]},
            },
            "37": {
                "class_type": "SaveImage",
                "inputs": {"images": ["36", 0], "filename_prefix": filename_prefix},
            },
        }

    # ------------------------------------------------------------------
    # HTTP plumbing (stdlib urllib via asyncio.to_thread)
    # ------------------------------------------------------------------

    async def _get_json(
        self, url: str, *, fallback: str | None = None, timeout: float = _HTTP_TIMEOUT
    ) -> Any:
        """GET and parse JSON on a worker thread; None means unreachable.

        Connection-level failures are normalized here so callers translate
        them into :class:`BackendUnavailableError` with precise context.
        """
        try:
            return await asyncio.to_thread(_http_json, url, timeout=timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            if fallback is not None:
                return await self._get_json(fallback, timeout=timeout)
            return None

    async def _submit(self, workflow: dict[str, dict[str, Any]]) -> str:
        """POST the workflow to ``/prompt``; return the queued prompt_id."""
        payload = json.dumps({"prompt": workflow, "client_id": self._client_id}).encode()
        url = f"{self.config.base_url}/prompt"
        try:
            body = await asyncio.to_thread(
                _http_json_bytes, url, data=payload, timeout=self.config.submit_timeout
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                # 400 = ComfyUI validated the workflow and rejected it
                # (missing model file, unknown node, bad input). A request
                # problem, not an outage.
                raise CapabilityError(
                    f"ComfyUI rejected the workflow (HTTP 400): {_error_body(exc)}"
                ) from exc
            raise BackendUnavailableError(f"ComfyUI /prompt answered HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(
                f"cannot submit to ComfyUI at {self.config.base_url}: {exc}"
            ) from exc
        prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise CapabilityError(f"ComfyUI /prompt returned no prompt_id: {body!r}")
        return prompt_id

    async def _poll(self, prompt_id: str) -> dict[str, Any]:
        """Poll ``/history/{prompt_id}`` until the workflow finishes.

        ComfyUI writes a prompt's history entry when it finishes executing,
        so the loop looks for the prompt_id key, checks its status, and is
        bounded by ``poll_timeout``. The sleep between polls leaves the
        shared event loop free for the HTTP surface.

        A submitted prompt that is absent from both the running and pending
        queue *and* has no history entry means the backend lost it: the
        server crashed or restarted mid-render (a GPU runtime segfault
        looks exactly like this from HTTP). Waiting for ``poll_timeout``
        would stall the caller for the full window against a dead render,
        so the loop fails fast with :class:`BackendUnavailableError` after
        a short confirmation (the poll below already re-checks history, so
        a completed prompt is never mistaken for a lost one).
        """
        url = f"{self.config.base_url}/history/{urllib.parse.quote(prompt_id)}"
        # ComfyUI runs one prompt at a time; a queue position is a plain
        # prompt id string in both lists. Sampled sparsely (see below)
        # because the authoritative completion signal is history.
        queue_url = f"{self.config.base_url}/queue"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.poll_timeout
        # The backend may restart between consecutive polls; require the
        # prompt to be missing twice in a row before declaring it lost.
        lost_streak = 0
        while True:
            try:
                body = await asyncio.to_thread(_http_json, url, timeout=_HTTP_TIMEOUT)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                raise BackendUnavailableError(
                    f"cannot poll history from ComfyUI at {self.config.base_url}: {exc}"
                ) from exc
            entry = body.get(prompt_id) if isinstance(body, dict) else None
            if isinstance(entry, dict):
                status = entry.get("status")
                status_str = status.get("status_str") if isinstance(status, dict) else None
                if status_str == "error":
                    raise CapabilityError("ComfyUI reported an execution error for this workflow")
                completed = status.get("completed", True) if isinstance(status, dict) else True
                if completed:
                    outputs = entry.get("outputs")
                    if isinstance(outputs, dict):
                        return outputs
            else:
                # No history entry yet: still running or lost. Distinguish
                # by asking the queue. Both checks are cheap JSON GETs and
                # history remains the completion authority, so at worst this
                # adds one request per poll interval.
                queued = await self._prompt_in_queue(prompt_id, queue_url)
                if queued:
                    lost_streak = 0
                else:
                    lost_streak += 1
                    if lost_streak >= 2:
                        raise BackendUnavailableError(
                            "ComfyUI lost the render: the queued prompt "
                            f"{prompt_id} is neither running, pending, nor in "
                            "history (the backend likely crashed or restarted "
                            "mid-render)"
                        )
            if loop.time() >= deadline:
                raise BackendUnavailableError(
                    f"ComfyUI did not finish rendering within "
                    f"{int(self.config.poll_timeout)} seconds"
                )
            await asyncio.sleep(self.config.poll_interval)

    async def _prompt_in_queue(self, prompt_id: str, queue_url: str) -> bool:
        """Return whether ``prompt_id`` is queued (running or pending).

        Unreachable states raise :class:`BackendUnavailableError`: a dead
        server is an outage, same as a submitted render never finishing.
        ComfyUI's queue entries are prompt-id strings; malformed shapes
        (older/newer versions, proxies) count as not-queued rather than
        crashing the poll loop.
        """
        try:
            body = await asyncio.to_thread(_http_json, queue_url, timeout=_HTTP_TIMEOUT)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(
                f"cannot read ComfyUI queue at {self.config.base_url}: {exc}"
            ) from exc
        if not isinstance(body, dict):
            return True  # unknown shape: never claim the prompt is lost
        for key in ("queue_running", "queue_pending"):
            seq = body.get(key)
            if isinstance(seq, list) and any(
                _queue_item_contains_prompt(item, prompt_id) for item in seq
            ):
                return True
        return False

    async def _collect_images(
        self,
        outputs: dict[str, Any],
        prompt_id: str,
        seed: int,
        width: int,
        height: int,
    ) -> list[GeneratedImage]:
        """Take the first output image, fetch it via ``/view``, return one
        :class:`GeneratedImage` with base64 PNG bytes and render metadata."""
        for node_id in sorted(outputs):
            node_out = outputs[node_id]
            if not isinstance(node_out, dict):
                continue
            for image in node_out.get("images") or []:
                if not isinstance(image, dict) or image.get("type") != "output":
                    continue
                filename = image.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                subfolder = image.get("subfolder") or ""
                data = await self._fetch_view(filename, subfolder)
                return [
                    GeneratedImage(
                        b64_json=base64.b64encode(data).decode("ascii"),
                        metadata={
                            "backend": self.backend_name,
                            "format": "png",
                            "size": f"{width}x{height}",
                            "seed": seed,
                            "comfyui": {
                                "prompt_id": prompt_id,
                                "filename": filename,
                                "subfolder": subfolder,
                            },
                        },
                    )
                ]
        raise CapabilityError("ComfyUI prompt completed without any image outputs")

    async def _fetch_view(self, filename: str, subfolder: str) -> bytes:
        """GET ``/view`` for the saved image bytes (SaveImage writes PNG)."""
        url = f"{self.config.base_url}/view?" + urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": "output"}
        )
        try:
            data = await asyncio.to_thread(_http_bytes, url, timeout=_HTTP_TIMEOUT)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(
                f"cannot fetch image from ComfyUI at {self.config.base_url} (/view): {exc}"
            ) from exc
        if not data.startswith(_PNG_MAGIC):
            raise CapabilityError("ComfyUI /view returned non-PNG data")
        return data


# ----------------------------------------------------------------------
# stdlib HTTP helpers (module-private thread targets)
# ----------------------------------------------------------------------
# Tests monkeypatch these three functions to simulate a ComfyUI server.


def _http_json(url: str, *, timeout: float) -> Any:
    """GET a URL and return its parsed JSON body."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _queue_item_contains_prompt(item: Any, prompt_id: str) -> bool:
    """Match ComfyUI queue entries across its string and structured shapes.

    Older ComfyUI versions exposed prompt IDs directly in ``queue_running`` /
    ``queue_pending``. Current versions return queue records containing the ID
    alongside execution metadata, so comparing the whole record to the ID
    incorrectly reports an active GPU render as lost.
    """
    if isinstance(item, str):
        return item == prompt_id
    if isinstance(item, dict):
        return any(_queue_item_contains_prompt(value, prompt_id) for value in item.values())
    if isinstance(item, (list, tuple)):
        return any(_queue_item_contains_prompt(value, prompt_id) for value in item)
    return False


def _http_bytes(url: str, *, timeout: float) -> bytes:
    """GET a URL and return its raw body."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _http_json_bytes(url: str, *, data: bytes, timeout: float) -> Any:
    """POST JSON bytes and return the response's parsed JSON body.

    A non-2xx answer raises :class:`urllib.error.HTTPError` for the caller
    to classify (400 → workflow rejected; else → server problem).
    """
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "arc-llama-vision"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _error_body(exc: urllib.error.HTTPError) -> str:
    """Best-effort extraction of an HTTPError body for diagnostics."""
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # diagnostics must never raise in the error path
        return "<no error body>"
