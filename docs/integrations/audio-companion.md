# Audio companion integration

This document defines the boundary for a separate audio/voice companion to
arc-llama. The companion should provide speech-to-text (ASR), text-to-speech
(TTS), or both, while arc-llama remains focused on local LLM inference and
model lifecycle management.

The integration boundary is HTTP. The companion must not import private
arc-llama modules such as the router, launcher, or server implementation.

## Scope

The companion project owns:

- ASR and TTS backends and their optional dependencies;
- audio-device or audio-file handling;
- voice selection and voice-specific configuration;
- its own process lifecycle, logging, and resource management; and
- the audio HTTP endpoints described below.

arc-llama owns:

- local GGUF model discovery and `llama-server` lifecycle;
- the main OpenAI-compatible LLM API; and
- optional registration of the companion as an upstream endpoint.

The companion should remain useful when run by itself. A failure or missing
audio dependency must not prevent arc-llama from starting or serving text
inference.

## Existing arc-llama API

The default arc-llama server is available at:

```text
http://127.0.0.1:11437
```

The host and port are configurable. On a running server, the companion may
use these endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Basic server health check |
| `GET /v1/models` | Discover local and registered upstream models |
| `POST /v1/chat/completions` | OpenAI-compatible text generation |
| `POST /v1/completions` | OpenAI-compatible completion API |
| `POST /v1/embeddings` | OpenAI-compatible embeddings API |
| `GET /admin/status` | Detailed status, GPU, model, and upstream information |

`/v1/models` returns standard OpenAI model-list data. Local models use
`owned_by: "arc-llama"`; models discovered from an upstream use
`owned_by: "upstream:NAME"`.

The administrative endpoints that change state, including model load and
stop operations, may require:

```http
Authorization: Bearer <ARC_LLAMA_ADMIN_TOKEN>
```

The companion must treat administrative endpoints as optional. It should
continue to work if no admin token is configured or if those endpoints are
unavailable.

## Companion API

The companion should expose an OpenAI-compatible base URL, for example:

```text
http://127.0.0.1:11438
```

It should implement:

### `GET /health`

Return HTTP `200` when the service is able to accept requests. A small JSON
body is recommended:

```json
{"status":"ok"}
```

Return a non-200 response while the service is starting or when its required
backend is unavailable.

### `GET /v1/models`

Return a standard OpenAI model list. Each audio-capable model must have a
stable `id` and should include useful metadata, such as its modality and
backend:

```json
{
  "object": "list",
  "data": [
    {
      "id": "audio-transcriber",
      "object": "model",
      "owned_by": "audio-companion",
      "metadata": {"modality": "audio->text", "backend": "llama.cpp"}
    }
  ]
}
```

### `POST /v1/audio/transcriptions`

Accept the standard multipart form used by OpenAI-compatible clients:

- `file`: audio file, required;
- `model`: model ID, required;
- `language`: optional language hint;
- `prompt`: optional transcription hint; and
- `response_format`: `json` by default, with any additional formats clearly
  documented by the companion.

For the default JSON format, return at least:

```json
{"text":"Transcribed text"}
```

Return `400` for malformed requests, `404` for an unknown model, and `503`
when the selected backend is unavailable.

### `POST /v1/audio/speech`

Accept a JSON request compatible with OpenAI clients:

```json
{
  "model": "audio-speaker",
  "input": "Text to speak.",
  "voice": "default",
  "response_format": "wav"
}
```

Return the generated audio bytes with an accurate `Content-Type`, such as
`audio/wav`, `audio/mpeg`, or `audio/opus`. Return a JSON error using a stable
schema for invalid input, an unknown voice/model, or backend failures.

The companion should document its supported audio formats, maximum input
size, maximum text length, and whether requests are serialized when sharing a
GPU.

## Registering with arc-llama

The current upstream mechanism is the closest existing integration point:

```bash
arc-llama upstream add audio-companion http://127.0.0.1:11438
arc-llama upstream list
```

This causes the companion's models to appear in arc-llama's `/v1/models`
response and allows supported model-targeted requests to be sent to the
companion without starting a local `llama-server` for those models.

The companion should use distinct model IDs, for example
`audio-transcriber` and `audio-speaker`, so they cannot collide with local
LLM model names.

### Important current limitation

The current `main` branch documents transparent upstream routing for the
text endpoints, not audio endpoints. Do not advertise
`arc-llama/v1/audio/*` passthrough as supported until the core adds and tests
those routes. The companion can still run independently and be called
directly by Home Assistant or another client today.

If transparent audio routing is added later, it should preserve the
multipart request and binary response unchanged and select the upstream by
the request's `model` field, just as text requests are selected today.

## Configuration and lifecycle

The companion should provide:

- a configurable bind host and port;
- a configurable arc-llama base URL;
- environment-variable and command-line configuration;
- a clear startup command and a health check;
- graceful shutdown; and
- a documented way to select CPU versus GPU/back-end execution.

Do not assume that arc-llama starts or stops the companion. A future launcher
integration may be added separately, but the initial companion must be an
independent process with an independent dependency set.

## Windows requirements

Windows is a supported target for the integration. The companion must:

- use `pathlib.Path` or equivalent platform-neutral path handling;
- avoid hard-coded `/tmp`, `/bin`, shell pipelines, and POSIX-only signals;
- avoid assuming `fork`, process groups, or `setsid` exist;
- support Windows Python environments and executable paths containing spaces;
- make temporary files with the platform's temporary-directory API; and
- include a Windows smoke test covering startup, `/health`, model listing,
  transcription, and speech generation.

If a backend is optional or unavailable on Windows, the service should fail
with a clear capability error while leaving the rest of the companion
usable.

## Compatibility and testing

The companion should pin or declare its compatible arc-llama API version and
test against a real running arc-llama instance or a small HTTP mock. Minimum
integration checks are:

1. Start the companion without arc-llama and verify `/health`.
2. Start arc-llama and register the companion as an upstream.
3. Verify the audio models appear in `/v1/models` with unique IDs.
4. Verify ASR returns the expected transcription schema.
5. Verify TTS returns audio bytes and the correct content type.
6. Verify unknown models and unavailable backends return documented errors.
7. Verify the same checks on Windows.

The companion's README should link back to this document and include a
copy-and-run example for Home Assistant.
