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
- `POST /api/chat` accepts Ollama `messages` requests and supports JSON or
  newline-delimited streaming responses.
- `POST /api/generate` accepts an Ollama `prompt` and supports JSON or
  newline-delimited streaming responses.

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

## Anthropic Messages compatibility

`POST /v1/messages` translates Messages requests through the same local/upstream
chat proxy as `/v1/chat/completions`. Use a registered model name or alias, not a
hosted model name unless that name exists in your registry. Remote access follows
the existing client-key rules; this route accepts either `x-api-key` or
`Authorization: Bearer <key>`. If both are supplied, Bearer takes precedence.
Beta/version headers are accepted without enabling hosted service features.

Supported request shapes include string or text-block system prompts; user and
assistant text; base64 PNG/JPEG/GIF/WebP images; client tools with `input_schema`;
assistant `tool_use` and user `tool_result` blocks; tool selection and parallel
call control; `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` and
`stream`. Tool errors are preserved as labeled tool output. Metadata, cache
hints and hosted service-tier/speed/geography hints are accepted and ignored.
They do not enable prompt caching, hosted tools or geography guarantees locally.

`thinking: {"type":"adaptive"}` enables the runtime template's thinking mode;
`disabled` disables it. `output_config.effort` maps to `reasoning_effort`, and its
JSON schema format maps to OpenAI `response_format`. These remain dependent on
the chosen runtime/template. Fixed thinking budgets, signed thinking-history
blocks, server-executed tools, PDF/documents and automatic context edits cannot
be faithfully mapped and return a descriptive 400. The adapter returns final
text and tool calls, without hosted signed-reasoning blocks. Other unsupported
answer-changing request fields are errors rather than silently discarded.

Responses contain text/tool-use blocks, JSON object tool inputs, stop reasons
and token usage. Length and tool-call finishes map to `max_tokens` and `tool_use`.
An explicitly reported matched stop string maps to `stop_sequence`; a generic
OpenAI `stop` without that information maps to `end_turn` because it cannot
identify whether a custom sequence or ordinary end-of-turn ended generation.
Usage counts are forwarded, not estimated: an unreported count is zero and does
not establish that no tokens were consumed. No hosted cache-token counts are
invented.

Streams emit `message_start`, indexed content block start/delta/stop events,
`message_delta` and `message_stop`. Text is forwarded incrementally. Tool
metadata/arguments are collected until completion so split names/IDs and
parallel calls form stable blocks; argument fragments then use
`input_json_delta`. Final usage arrives in `message_delta`. A malformed,
truncated or failed upstream stream emits `error` without a success terminator.
Disconnects close the existing proxy's response/client and release local work.

```sh
curl http://127.0.0.1:11437/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: YOUR_ARC_KEY' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"YOUR_REGISTERED_MODEL","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Contract sources, accessed 2026-11-16:
[Messages request/response reference](https://platform.claude.com/docs/en/api/messages/create),
[stream event reference](https://platform.claude.com/docs/en/build-with-claude/streaming),
and [the Python SDK request fields](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/messages/messages.py).
A [CLI report](https://github.com/anthropics/claude-code/issues/68551) describes
adaptive thinking/cache hints sent to custom endpoints; this is reported client
behavior, not a measured CLI session here. Automated compatibility checks use
controlled upstream responses; no complete external CLI workflow is claimed.
