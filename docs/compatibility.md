# Compatibility contract

This contract starts with the 0.9 release line. Changes outside the guarantees
below may still happen during 0.9, but releases will document them.

## Supported hosts

- Windows 10/11 x86-64 and Linux x86-64.
- Python 3.10 through 3.14.
- Battlemage GPUs recognized by `arc-llama gpus`; native validation is recorded
  for the Arc Pro B60 on Linux (see the README's Tested hardware table).
- Alchemist and Lunar Lake GPUs are recognized and expected to work through the
  same Vulkan and SYCL runtimes, but no native results have been recorded for
  them yet, so they are not covered by this contract. Hardware reports
  ([issue #71](https://github.com/offbyonebit/arc-llama/issues/71)) move a
  device into the validated list.
- Vulkan is the portable default. SYCL is supported when a compatible Intel
  oneAPI runtime and SYCL llama.cpp build are installed.

macOS is outside the supported hardware matrix because Intel Arc is not a
supported macOS inference target. Other GPU vendors may work through an
upstream server, but Arc Llama does not manage their local runtimes.

## OpenAI-compatible API

Arc Llama supports model discovery through `GET /v1/models` and forwards
non-streaming and streaming requests for:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

The request and response fields accepted by the active llama.cpp build are
passed through. Arc Llama guarantees routing, model selection, ordinary JSON
errors, and OpenAI-style server-sent event forwarding. A field implemented only
by a particular llama.cpp build remains conditional on that build. The `/admin`,
`/v1/agent`, and `/v1/chats` routes are Arc Llama extensions and are versioned
with this project rather than the OpenAI API.

`GET /health` returns HTTP 503 with `status: starting` until startup completes,
or `status: error` after a startup failure. Once ready it returns HTTP 200;
`status: loading` identifies a model launch, while `status: ok` means no model
launch is in progress.

### Model capabilities

Each local entry in `GET /v1/models` carries `metadata.capabilities`:
`chat`, `completion`, and `embedding` for ordinary models, plus `vision`
when a projector is attached, or only `rerank` for a reranker.

- **Vision-language models.** A model whose recipe has `mmproj` starts with
  `--mmproj` and accepts OpenAI `image_url` content parts in
  `/v1/chat/completions`. `arc-llama scan`, `add`, and `run` attach a
  projector from the model's folder automatically: one whose name shares
  the model's family (`mmproj-gemma-3-...`), or a generic `mmproj-F16.gguf`
  when every model in the folder is the same family. Hugging Face downloads
  fetch the repo's projector beside the model. Image requests to a model
  without a projector fail with HTTP 400 before any model is loaded. The
  projector's size counts toward the VRAM fit; set `mmproj_offload = false`
  to keep it on the CPU.
- **Rerankers.** `POST /v1/rerank` forwards to a model whose recipe sets
  `reranking = true`. Other models are refused with HTTP 400.
- **Split GGUFs.** Files named `<name>-00001-of-0000N.gguf` register as one
  model. Only the first shard is listed, sizes and VRAM fit cover every
  shard, Hugging Face downloads fetch all shards, and a missing shard fails
  with `shard_missing` before llama-server starts.

## Ollama-compatible API

Arc Llama also provides a small compatibility surface for Ollama-aware clients:

- `GET /api/tags` returns registered local and upstream models.
- `POST /api/chat` translates text, base64 PNG/JPEG/GIF/WebP images, function
  tools and tool-result history to the OpenAI chat path. Tool arguments return
  as JSON objects. Streamed argument fragments are assembled before a complete
  tool-call event is emitted; ordinary text streams immediately as NDJSON.
- `POST /api/generate` translates `prompt`, `system` and optional images to the
  same chat path, returning Ollama `response` text in JSON or NDJSON.
- Both accept `format: "json"` or a JSON schema object through OpenAI
  `response_format`. Sampling options supported here are `temperature`,
  `top_p`, `top_k`, `min_p`, `stop`, `seed`, `repeat_penalty` and `num_predict`.
  Actual tool, vision and schema enforcement still depends on the model and
  active llama-server build.
- `POST /api/show` returns basic local-library details, known GGUF architecture
  and context metadata, configured context parameters and capability tags.
  Unavailable templates, licenses, quantization/parameter counts and full
  tokenizer metadata are omitted; `verbose: true` is refused explicitly.
- `GET /api/ps` lists ready local models with known file sizes and basic
  details. It omits unmeasured VRAM allocation, content digests and per-request
  expiration times. Upstream-server residency is not observable here.

Common client settings `keep_alive`, `think` and server load options such as
`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`,
`use_mlock`, `numa`, `low_vram`, `f16_kv`, `logits_all`, `vocab_only` and
`num_gqa` are accepted and ignored. Configure model lifetime, context and runtime
settings in Arc Llama's controls; `think` does not change reasoning behavior.
Unmappable prompt controls (raw/templates/suffixes and legacy encoded context)
and unknown options return clear errors. Load-only empty requests are refused
rather than being treated as chat generation. Invalid upstream tool arguments
or truncated streams return errors; a failed stream does not emit a successful
completion marker.

Field names and response shapes follow the
[public Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
(accessed 2026-11-16). Tests exercise controlled upstream responses; they do not
establish native GPU inference, model tool quality or JSON-schema enforcement.

The Ollama routes translate requests into Arc Llama's existing model router;
they do not implement Ollama model storage, Modelfiles, or runtime management.

## Configuration compatibility

Existing documented keys are preserved throughout the 0.9 line. New optional
keys receive defaults when an older configuration is loaded. Removing or
changing the meaning of a documented key requires a schema-version change, an
automatic migration where feasible, and a changelog entry. Unknown keys may be
ignored and must not be used as the only copy of user data.

Recipes and community registries are treated as advisory inputs. Unsupported
recipe fields are ignored or rejected, and shared performance claims do not
become active without the command's provenance and local measurement gates.

## Experimental features

Multi-GPU scheduling, speculative strategies beyond target-only fallback,
agents, MCP tools, plugins, and the audio-companion contract are experimental
for 0.9. They may gain fields or stricter validation between minor releases.
Core single-GPU inference, runtime installation, model registration, routing,
and the three OpenAI-compatible endpoints above are the stabilization target.

### Splitting one model across GPUs (experimental)

A recipe can spread one model over several cards with `tensor_split` (one
proportion per card, for example `[1, 1]`) and `split_gpus` (the cards' PCI
slots, including the model's own GPU). Every listed GPU must be enabled and
use the same backend. The launcher exposes all of them to llama-server
(`ONEAPI_DEVICE_SELECTOR` or `--device`), and the fit check
counts their combined VRAM. An invalid combination falls back to the model's
single GPU with a warning.
