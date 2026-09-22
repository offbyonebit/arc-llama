# Vision companion integration (image generation)

This document defines the boundary for a separate image-generation companion
to arc-llama, following the same companion pattern as
[integrations/audio-companion.md](audio-companion.md) and the
[plugin contract](../plugins.md). arc-llama remains focused on local LLM
inference and model lifecycle management; image generation lives outside the
core.

**Status: bounded first pass.** A scaffold (`arc-llama-vision/`) implements
the full HTTP surface and adapter seam with a deterministic fake backend,
an outage-testing unavailable backend, and one real optional backend: a
ComfyUI adapter for FLUX.2 Klein GGUF text-to-image workflows. No model
weights or heavyweight ML dependencies are involved in this package —
the ComfyUI adapter talks plain stdlib HTTP to a ComfyUI server that owns
torch and the weights.

The integration boundary is HTTP. The companion must not import private
arc-llama modules such as the router, launcher, or server implementation.

## Scope

The companion project owns:

- image-generation backends and their optional dependencies;
- prompt/size/format handling for image requests;
- its own process lifecycle, logging, and resource management; and
- the image HTTP endpoints described below.

arc-llama owns:

- local GGUF model discovery and `llama-server` lifecycle;
- the main OpenAI-compatible LLM API; and
- model-list merging from registered upstreams.

The companion should remain useful when run by itself. A missing vision
dependency or failed backend startup must not prevent arc-llama from
starting or serving text inference — they are separate processes with
separate dependency sets.

## Companion API

The companion exposes an OpenAI-compatible base URL, for example:

```text
http://127.0.0.1:11440
```

### `GET /health`

```json
{"status": "ok", "backend": "fake"}
```

`status` is `"degraded"` when backend startup failed, in which case
generation requests return `503`.

### `GET /v1/models`

Standard OpenAI model list; every entry carries image modality metadata:

```json
{
  "object": "list",
  "data": [
    {
      "id": "arc-vision-diffusion",
      "object": "model",
      "owned_by": "vision-companion",
      "created": 0,
      "metadata": {"modality": "image->image", "backend": "fake"}
    }
  ]
}
```

Model ids should be distinct from local LLM names (the scaffold uses
`arc-vision-*` ids) so they cannot collide.

### `POST /v1/images/generations`

Accepts a JSON request compatible with OpenAI clients:

```json
{
  "model": "arc-vision-diffusion",
  "prompt": "A red cube on a table.",
  "n": 1,
  "size": "512x512"
}
```

Response (`b64_json` is the only guaranteed format):

```json
{
  "created": 1710000000,
  "data": [{"b64_json": "<base64 image bytes>"}]
}
```

Errors use the stable schema `{"detail": {"error": {"type", "message"}}}`:

- `400` malformed requests (including body-validation failures remapped
  from FastAPI's 422), over-limit `n`/prompt/size, and `response_format:
  "url"`;
- `404` unknown model; and
- `503` backend unavailable (process down or startup failed).

## Backend adapter seam

All rendering crosses the `arc_llama_vision.backend.ImageBackend` seam. See
the scaffold README (`arc-llama-vision/README.md`) for a copy-paste
adapter example and rules (lazy heavy imports, `BackendUnavailableError`
vs `CapabilityError`, no HTTP parsing inside adapters). Three adapters
ship:

- `fake` — deterministic PNG output derived from the prompt, for tests;
- `unavailable` — always fails, to exercise the 503 path; and
- `comfyui` — real rendering through a local ComfyUI server (below).

A real adapter (e.g. for a local diffusion server) is registered in
`BACKEND_REGISTRY` and selected with `ARV_BACKEND` plus per-adapter
`ARV_BACKEND_OPTIONS`. Real adapters must import heavy dependencies lazily
inside `startup()`/`generate()`, never at module import time.

### The `comfyui` adapter

Selected with `ARV_BACKEND=comfyui`, entirely configured through the
`ARV_BACKEND_OPTIONS` JSON bag, and using only stdlib HTTP/JSON/base64
(`urllib.request` through `asyncio.to_thread` — no websockets, no ML
packages, no template files):

```bash
ARV_BACKEND=comfyui \
ARV_BACKEND_OPTIONS='{
  "base_url": "http://127.0.0.1:8190",
  "model": "arc-vision-flux2-klein-9b-uncensored",
  "unet_gguf": "flux2-klein-9b-Q4_K_M.gguf",
  "clip_gguf": "qwen3-8b-Q2_K-uncensored.gguf",
  "vae": "flux2-vae.safetensors",
  "steps": 20,
  "guidance": 5.0,
  "sampler": "euler",
  "filename_prefix": "arc-vision",
  "poll_timeout": 1200
}' \
.venv/bin/python -m arc_llama_vision
```

Options (default): `base_url` (`http://127.0.0.1:8190`), `model`
(`arc-vision-flux2-klein-9b-uncensored` — the id served at `/v1/models`),
`unet_gguf` / `clip_gguf` / `vae` (the filenames ComfyUI lists in
`models/unet_gguf`, `models/clip_gguf`, `models/vae`), `steps` (`20`),
`guidance` (`5.0`), `sampler` (`euler`), `filename_prefix`
(`arc-vision`), and the `submit_timeout` / `poll_timeout` /
`poll_interval` bounds (`60` / `1200` / `1` seconds). Unknown keys are
ignored; malformed values fail startup with the key named in the error.

Requires a ComfyUI install with the city96/ComfyUI-GGUF custom nodes
(`UnetLoaderGGUF`, `CLIPLoaderGGUF`) and the Flux.2 core nodes
(`EmptyFlux2LatentImage`, `Flux2Scheduler`). Each request submits a
minimal API-format FLUX.2 Klein text-to-image graph — `UnetLoaderGGUF`,
`CLIPLoaderGGUF` (type `flux2`), `VAELoader`, `CLIPTextEncode` (positive
plus empty negative), `EmptyFlux2LatentImage`, `KSamplerSelect`,
`Flux2Scheduler`, `CFGGuider`, `RandomNoise`, `SamplerCustomAdvanced`,
`VAEDecode`, `SaveImage` — polls `GET /history/{prompt_id}` with a
bounded timeout, fetches the first image via `GET /view`, and returns an
OpenAI-shaped response whose first `data` entry carries `b64_json` plus a
`comfyui` metadata object (`prompt_id`, `filename`, `subfolder`).
`width`/`height` must be multiples of 16 (FLUX.2 patch size), else 400.
Startup probes `/system_stats` (falling back to `/prompt`); an
unreachable server degrades `/health` and generation answers 503.

**Tested setup note:** this adapter was run against ComfyUI with
**FLUX.2 Klein 9B (Q4)** plus a **public Q2 uncensored Qwen3 text
encoder**, **CPU-only**, where **512x512** at the default 20 steps took
about **7 minutes** per image. `poll_timeout` defaults to 20 minutes
accordingly, and `n` is fixed at 1. GPU support is neither claimed nor
tested here — device placement is ComfyUI's business.

## Registering with arc-llama

```bash
arc-llama upstream add vision-companion http://127.0.0.1:11440
```

This causes the companion's models to appear in arc-llama's `/v1/models`
response.

### Important current limitation

**Core upstream routing does not yet proxy image endpoints.** The current
`main` branch implements transparent upstream routing for text endpoints
(`chat/completions`, `completions`, `embeddings`) — not
`/v1/images/generations`. Registering the companion makes its models
*visible* in the merged model list, but requests for them through the
core server are not forwarded. Do not advertise `arc-llama/v1/images/*`
passthrough as supported until the core adds and tests those routes.
Clients should call the companion directly at its own base URL today.

If transparent image routing is added later, it should preserve the JSON
request body and response unchanged and select the upstream by the
request's `model` field, just as text requests are selected today.

## Configuration and lifecycle

The companion provides:

- a configurable bind host and port (default `127.0.0.1:11440`);
- environment-variable (`ARV_*`) and command-line configuration, CLI winning;
- bounded request limits (batch size, prompt length, size bounds) that are
  enforced before any backend work; and
- graceful shutdown through uvicorn's lifecycle.

Do not assume arc-llama starts or stops the companion; it is an independent
process with an independent dependency set.

## Windows requirements

Windows is a supported target, per the same rules as the audio companion:
`pathlib`-only path handling, no hard-coded POSIX paths or signals, no
`fork`/`setsid` assumptions, OS-assigned or configured TCP ports, and
thread-based test harnesses. The companion's own test suite runs on
Windows.

## Compatibility and testing

The companion declares its compatible arc-llama API version and tests
against its own in-process app plus one real-socket server boot. Minimum
integration checks (all implemented in `arc-llama-vision/tests/`):

1. Start the companion alone; verify `/health` (ok and degraded) and model
   listing with unique ids and modality metadata.
2. Verify generation returns the documented response schema and `b64_json`
   bytes, deterministically.
3. Verify unknown models, invalid input, and over-limit requests return
   documented errors (404/400).
4. Verify backend-unavailable behavior returns `503`.
5. Verify a standalone server boot serves every endpoint over real TCP.
6. Verify no heavyweight ML dependencies are imported by the package.
7. For the `comfyui` adapter, verify configuration/option validation,
   workflow construction, and the submit → history → view lifecycle against
   a simulated ComfyUI server (see `arc-llama-vision/tests/`), including
   poll-timeout and connection-failure paths.