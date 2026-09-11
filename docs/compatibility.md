# Compatibility contract

This contract starts with the 0.9 release line. Changes outside the guarantees
below may still happen during 0.9, but releases will document them.

## Supported hosts

- Windows 10/11 x86-64 and Linux x86-64.
- Python 3.10 through 3.14.
- Intel Arc Alchemist, Battlemage, and Lunar Lake GPUs recognized by
  `arc-llama gpus`.
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
