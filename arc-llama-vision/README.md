# arc-llama-vision

A bounded, first-pass **image-generation companion** for
[arc-llama](https://github.com/offbyonebit/arc-llama). It follows the
repo's established companion design (see `docs/plugins.md` and
`docs/integrations/audio-companion.md`): a standalone, local-first HTTP
service that lives *outside* the arc-llama core, with no imports of private
core modules, no model weights, and no heavyweight ML dependencies.

The scaffold ships a **backend-neutral adapter seam** with a real optional
backend. Three adapters are included:

| Adapter | Behavior |
| --- | --- |
| `fake` (default) | Deterministic, dependency-free PNG output for tests and wiring checks |
| `unavailable` | Always fails; exercises the backend-unavailable (503) path |
| `comfyui` | Real text-to-image rendering through a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server (see below) |

## Contents

- [Quick start](#quick-start)
- [HTTP API](#http-api)
- [OpenAI-compatible response shape](#openai-compatible-response-shape)
- [Error contract](#error-contract)
- [Configuration](#configuration)
- [ComfyUI backend](#comfyui-backend)
- [Plugging in a real backend adapter](#plugging-in-a-real-backend-adapter)
- [Integration with arc-llama](#integration-with-arc-llama)
- [Current limitations](#current-limitations)
- [Development and tests](#development-and-tests)

## Quick start

From the repo root, using the project venv:

```bash
.venv/bin/python -m arc_llama_vision
# or with options:
.venv/bin/python -m arc_llama_vision --host 127.0.0.1 --port 11440 --backend fake
```

Then verify:

```bash
curl http://127.0.0.1:11440/health
# {"status":"ok","backend":"fake"}

curl http://127.0.0.1:11440/v1/models
# {"object":"list","data":[{"id":"arc-vision-diffusion",...},...]}

curl -X POST http://127.0.0.1:11440/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "arc-vision-diffusion", "prompt": "a red cube", "size": "512x512"}'
```

## HTTP API

### `GET /health`

Liveness probe. `200` with `{"status": "ok", "backend": "<name>"}` when the
app and its backend started successfully. `status` becomes `"degraded"`
when backend startup failed — e.g. its process did not launch — in which
case generation requests return `503` (see the [error contract](#error-contract)).

### `GET /v1/models`

Standard OpenAI model list. Every entry carries image modality metadata so
clients can tell image models apart from text models:

```json
{
  "object": "list",
  "data": [
    {
      "id": "arc-vision-diffusion",
      "object": "model",
      "owned_by": "vision-companion",
      "created": 0,
      "metadata": {
        "modality": "image->image",
        "backend": "fake",
        "description": "deterministic test backend"
      }
    },
    {
      "id": "arc-vision-thumbnail",
      "object": "model",
      "owned_by": "vision-companion",
      "created": 0,
      "metadata": { "modality": "image->image", "backend": "fake", "description": "deterministic test backend" }
    }
  ]
}
```

Model ids are distinct from arc-llama text-model ids (which come from GGUF
filenames and may appear as `*.gguf`), so they cannot collide.

### `POST /v1/images/generations`

Accepts OpenAI-compatible image-generation parameters:

| Field | Required | Notes |
| --- | --- | --- |
| `model` | yes | Must be one of the ids from `GET /v1/models` |
| `prompt` | yes | Non-empty, ≤ `max_prompt_chars` (default 8192) |
| `n` | no | Default 1; capped at `max_batch_images` (default 1) |
| `size` | no | `WIDTHxHEIGHT`, each dimension 64–4096 |
| `response_format` | no | `b64_json` (default). `url` is rejected — see response shape |
| `quality`, `style`, `user` | no | Accepted and forwarded to the adapter |

Unknown fields are ignored, so newer OpenAI client options do not break the
endpoint.

## OpenAI-compatible response shape

```json
{
  "created": 1710000000,
  "data": [
    {
      "b64_json": "<base64 PNG bytes>",
      "metadata": {
        "backend": "fake",
        "format": "png",
        "size": "512x512",
        "seed": 4026604612
      }
    }
  ]
}
```

- Standard fields are OpenAI-identical: integer `created`, `data` array with
  per-image `b64_json` (or `url` when a backend supplies it).
- Only `b64_json` is guaranteed. Persistent `url` output would require an
  image store, which conflicts with the companion being stateless and
  local-first, so `response_format: "url"` returns `400` unless a backend
  adapter documents and provides real URLs.
- Each image may carry a non-standard `metadata` object for diagnostics
  (`backend`, `format`, `size`, `seed`); it is omitted entirely when the
  adapter supplies none, keeping the response OpenAI-identical.

With the default `fake` backend, images are tiny valid PNGs whose color is
derived deterministically from the prompt's CRC32, so identical requests
produce byte-identical responses — suitable for end-to-end assertions
without any ML dependency.

## Error contract

All failures use one stable JSON schema:

```json
{
  "detail": {
    "error": {
      "type": "invalid_request_error" | "server_error",
      "message": "human-readable description"
    }
  }
}
```

| Condition | Status |
| --- | --- |
| Malformed body / missing or invalid fields (incl. FastAPI validation) | `400` |
| `n` over the batch limit, prompt over the length limit, bad `size`, `response_format: "url"` | `400` |
| Unknown model | `404` |
| Backend process unavailable or backend startup failed | `503` |
| Adapter raised an unexpected error | `500` |

## Configuration

Precedence (low → high): built-in defaults → `ARV_*` environment variables →
CLI options. The companion keeps no state on disk.

| Env var | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `ARV_HOST` | `--host` | `127.0.0.1` | Bind host |
| `ARV_PORT` | `--port` | `11440` | Bind port (audio companion doc uses 11438; core uses 11437) |
| `ARV_BACKEND` | `--backend` | `fake` | Adapter name; must exist in the backend registry |
| `ARV_MAX_BATCH_IMAGES` | – | `1` | Max images returned per request |
| `ARV_MAX_PROMPT_CHARS` | – | `8192` | Max prompt length in characters |
| `ARV_BACKEND_OPTIONS` | – | `{}` | JSON object of per-adapter options, e.g. `{"api_base": "http://127.0.0.1:7860"}` |
| `ARV_CORS_ORIGINS` | – | localhost:3000 | Comma-separated allowed browser origins (same policy idea as the core) |

## ComfyUI backend

`ARV_BACKEND=comfyui` selects a real backend adapter that renders through
a local ComfyUI server (any ComfyUI install that has the
[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom nodes
and the Flux.2 core nodes). All configuration flows through
`ARV_BACKEND_OPTIONS` (JSON):

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
  "filename_prefix": "arc-vision"
}' \
.venv/bin/python -m arc_llama_vision
```

| Option | Default | Meaning |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8190` | ComfyUI server base URL |
| `model` | `arc-vision-flux2-klein-9b-uncensored` | Model id advertised at `GET /v1/models` |
| `unet_gguf` | `flux2-klein-9b-Q4_K_M.gguf` | GGUF diffusion-model filename in ComfyUI's `models/unet_gguf` |
| `clip_gguf` | `qwen3-8b-Q2_K-uncensored.gguf` | GGUF text-encoder filename in `models/clip_gguf` |
| `vae` | `flux2-vae.safetensors` | VAE filename in `models/vae` |
| `steps` | `20` | Step count passed to `Flux2Scheduler` |
| `guidance` | `5.0` | CFG value passed to `CFGGuider` |
| `sampler` | `euler` | Sampler name passed to `KSamplerSelect` |
| `filename_prefix` | `arc-vision` | SaveImage filename prefix in ComfyUI's `output/` |
| `submit_timeout` | `60` | Seconds allowed for `POST /prompt` |
| `poll_timeout` | `1200` | Seconds allowed for the full render (see note below) |
| `poll_interval` | `1` | Seconds between `/history/{id}` polls |

**File names must match your install.** The `unet_gguf`, `clip_gguf`, and
`vae` values are the filenames as ComfyUI lists them — adjust them to the
files actually present in ComfyUI's `models/unet_gguf`, `models/clip_gguf`,
and `models/vae` directories.

### Intel Arc XPU launch requirements

For Intel Arc GPU inference, run ComfyUI with a PyTorch XPU build and the
`ComfyUI-GGUF` custom node installed. Keep ComfyUI's PyTorch/SYCL/Unified
Runtime libraries together. In particular, a system oneAPI
`LD_LIBRARY_PATH` can cause PyTorch's bundled SYCL runtime to load a
different oneAPI Unified Runtime; that combination can crash during the
Flux text-encoder embedding step.

Use a launcher that removes inherited compiler/library paths while retaining
the device-selection variables:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /path/to/ComfyUI

exec env \
  -u LD_LIBRARY_PATH \
  -u LIBRARY_PATH \
  -u CPATH \
  -u CPLUS_INCLUDE_PATH \
  -u C_INCLUDE_PATH \
  -u PKG_CONFIG_PATH \
  -u CMAKE_PREFIX_PATH \
  ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
  ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}" \
  SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-0}" \
  /path/to/comfyui-venv/bin/python main.py \
  --listen 127.0.0.1 --port 8188 \
  --extra-model-paths-config /path/to/extra_model_paths.yaml \
  --use-pytorch-cross-attention
```

Before using the companion, verify that ComfyUI reports an XPU device in
`/system_stats`, for example `xpu:0 Intel(R) Arc(TM) Pro B60 Graphics`.
Do not silently fall back to CPU for production image generation: it can
make a valid request appear hung and can take many minutes per image.

How it works, end to end:

1. Each request builds a minimal **API-format FLUX.2 Klein text-to-image
   workflow** in code — `UnetLoaderGGUF`, `CLIPLoaderGGUF` (`type=flux2`),
   `VAELoader`, `CLIPTextEncode` (positive + empty negative),
   `EmptyFlux2LatentImage`, `KSamplerSelect`, `Flux2Scheduler`,
   `CFGGuider`, `RandomNoise`, `SamplerCustomAdvanced`, `VAEDecode`, and
   `SaveImage` — mirroring the official ComfyUI Flux.2 Klein template's
   wiring.
2. The workflow is submitted to `POST /prompt`; the returned `prompt_id`
   is polled at `GET /history/{prompt_id}` with a bounded timeout.
3. The first output image is fetched via `GET /view` and returned as
   `b64_json` with metadata (`backend`, `format`, `size`, `seed`, and a
   `comfyui` object with `prompt_id`/`filename`/`subfolder`).

Startup probes the ComfyUI server (`/system_stats`, falling back to
`/prompt`); an unreachable server marks the companion `degraded` (503 on
generation) without crashing it.

Size rule: `width` and `height` must be **multiples of 16** (the FLUX.2
patch size enforced by `EmptyFlux2LatentImage`/`Flux2Scheduler`); other
sizes return `400`.

**Performance note (tested setup):** this adapter has been exercised
against ComfyUI 0.20.1 with PyTorch 2.11.0+xpu on an **Intel Arc Pro B60**,
running **FLUX.2 Klein 9B Q4** with a public **Q2 uncensored Qwen3 text
encoder**. A real **512x512 GPU render completed in about 19 seconds at one
step** through the Arc Llama route. The default 20-step configuration will
take longer; `n` remains fixed at 1 to keep per-request VRAM bounded.
Device placement is entirely ComfyUI's business, so verify `/system_stats`
before generating. Set `poll_timeout` generously for cold starts and higher
step counts.

## Plugging in a real backend adapter

The seam is `arc_llama_vision.backend.ImageBackend` (an abstract base, but
any class exposing the same members works, mirroring the plugin contract in
`docs/plugins.md`). A real adapter for e.g. an SDXL- or FLUX-style backend:

```python
# my_vision_backend.py (own package; heavy deps stay optional and lazy)
from arc_llama_vision.backend import ImageBackend, ModelInfo, GeneratedImage


class SdxlGrpcBackend(ImageBackend):
    backend_name = "sdxl-grpc"
    models = [
        ModelInfo(
            id="arc-vision-sdxl",
            backend="sdxl-grpc",
            metadata={"modality": "image->image", "description": "SDXL via local server"},
        ),
    ]

    def __init__(self):
        self._client = None  # nothing heavy at construction time

    async def startup(self) -> None:
        # Import torch/diffusers/http client HERE (not at module import), so
        # the scaffold and core never pay for the adapter's dependencies.
        import torch  # lazy

    async def generate(self, prompt, model, *, n=1, size=None, **kwargs):
        w, h = (size or "1024x1024").split("x")
        png_bytes = ...  # render via llama.cpp-free backend of your choice
        return [GeneratedImage(b64_json=encode(png_bytes), metadata={"backend": self.backend_name})]

    async def shutdown(self) -> None: ...  # release GPU/process resources
```

Register it either by adding the class to `BACKEND_REGISTRY` (via your
package's own entry point or a small patch) and launching:

```bash
ARV_BACKEND=sdxl-grpc \
ARV_BACKEND_OPTIONS='{"api_base": "http://127.0.0.1:7860"}' \
arc-llama-vision --port 11440
```

Guidelines the built-in adapters model:

- declare stable `ModelInfo` ids with image modality metadata;
- import heavy dependencies inside `startup()`/`generate()`, never at module
  import time, so a missing dependency never breaks the whole service;
- raise `BackendUnavailableError` for a down process (→503) and
  `CapabilityError` for requests the adapter cannot fulfill (→400). Unexpected
  exceptions are caught, logged, and returned as 500 without crashing the app;
- keep request parsing out of the adapter: the endpoint validates required
  fields, sizes, and bounds, and passes plain arguments to `generate()`;
- honor `backend_options` keys that your adapter documents.

## Integration with arc-llama

This companion is a plain independent process. arc-llama's existing
`upstream` mechanism can be pointed at it (once a real backend exists, the
models appear in the hub's unified `GET /v1/models`):

```bash
arc-llama upstream add vision-companion http://127.0.0.1:11440
```

Nothing here requires the companion to be running to keep the core text
API fully functional. The companion never imports arc-llama internals;
it depends only on FastAPI/uvicorn (already in the core's dependency tree)
and can also be embedded as an `arc_llama.plugins`-style plugin by passing
`create_app(cfg).build_routes()` to a plugin's `register()` if desired later.

## Current limitations

This is an explicitly **bounded first pass**:

1. **One real backend, narrowly scoped.** The `fake` and `unavailable`
   adapters ship for tests and outage simulation, and `comfyui` renders
   real images through a local ComfyUI server (FLUX.2 Klein GGUF
   workflows only, one image per request). Other backends plug into the
   documented seam above; quality, sizing behavior, and performance are
   theirs to define.
2. **No model weights are downloaded or bundled**, and no ML libraries
   (torch/diffusers/transformers/numpy) are imported anywhere — verified by
   test at subprocess import. The ComfyUI adapter uses only stdlib
   HTTP/JSON/base64 (urllib + `asyncio.to_thread`) and never touches the
   weights itself; ComfyUI owns every heavyweight dependency.
3. **Core upstream routing does not yet proxy image endpoints.** Registering
   this companion with `arc-llama upstream add` makes its models *visible* in
   the merged model list, but arc-llama's core server does not forward
   `POST /v1/images/generations` requests to upstreams today (only
   `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` are
   routed). Until image-endpoint proxying is added and tested in the core,
   **do not advertise `arc-llama/v1/images/*` passthrough as supported**:
   clients should call the companion directly at its own base URL.
   Adding transparent image routing later should mirror the text flow —
   select the upstream by the request's `model` field and preserve the
   request body and JSON response unchanged.
4. **URL response format is rejected** (only `b64_json` is guaranteed) since
   the companion is stateless by design.
5. Batch size is capped (default 1 image per request); `size` accepts any
   `WIDTHxHEIGHT` from 64×64 to 4096×4096 at the endpoint layer, and the
   ComfyUI backend additionally requires multiples of 16 (the fake backend
   renders small PNGs only).

## Development and tests

```bash
# from the repo root (uv-managed venv already present)
.venv/bin/python -m pytest arc-llama-vision/tests -q

# lint and type-check the new package
.venv/bin/ruff check arc-llama-vision
.venv/bin/mypy arc-llama-vision/src \
    --python-executable=.venv/bin/python 2>/dev/null || true
```

The tests are fully local, bounded, and cross-platform (no fixed ports —
one test binds an OS-assigned loopback port for the real-socket end-to-end
run — and no POSIX-only paths/signals). Coverage:
health (ready + degraded), model listing (modality metadata, unique ids),
generation (OpenAI shape, valid PNG bytes, determinism, batching, optional
fields), invalid model (404), invalid input (400 incl. validation remap,
size, limits), backend-unavailable startup/render (503), adapter seam
factories, configuration precedence, and a standalone real-server boot with
`uvicorn` plus a bounded HTTP client run against every endpoint including
the 404 path.

The ComfyUI adapter has its own focused suite
(`tests/test_comfyui_adapter.py`) that simulates the ComfyUI HTTP surface
(monkeypatched stdlib HTTP targets — no server, weights, or network):
config/option validation and defaults, exact workflow construction (node
classes, input names, and link topology mirrored from upstream
signatures), mocked submit → history → view success, poll timeout and
server-gone failures (503), workflow-rejection/execution-error mapping
(400), startup probe and `/prompt` fallback, a degraded end-to-end app
boot, and a stdlib-only import check.

Cross-platform notes (per the companion integration doc): the package uses
only stdlib path-free mechanisms, `pathlib` where paths are involved,
threads (not fork), OS-assigned sockets, and uvicorn's built-in signal
handling. Nothing assumes POSIX.
