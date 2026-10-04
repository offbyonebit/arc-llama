# Linux end-to-end validation — 2026-10-03

Checkout: `/home/slowe/arc-llama`; branch: `maintenance/rc1-local-follow-up`.
Base HEAD: `6473f7a88c3d28b39096b83fb6ee4af9227bb015`.
Candidate: **0.9.0rc2**, including uncommitted source fixes. Nothing was committed,
pushed, or released. This report supersedes the October 1 report's Linux
coverage gaps; it does not certify the previous RC1 binary or future changes.

## Environment and containment

- Linux kernel 7.0.0-28-generic, xe driver, Intel Arc Pro B60, 24 GiB VRAM,
  32 GiB host RAM. Other physical Intel cards were not present.
- Python 3.10.20, 3.12.3, and 3.14.4.
- Portable Vulkan runtime **b11381**, actually downloaded from the official
  llama.cpp release into fresh temporary state; existing native SYCL runtime
  under `/mnt/storage/llama.cpp/build/bin/llama-server` with production oneAPI
  environment. The SYCL binary itself was not replaced by this work.
- All native GPU inference jobs ran serially and shared the production resident
  lock. Temporary Uvicorn services used distinct state and backend ports.
  The normal service remained idle during these runs.
- CPU regression jobs had no GPU devices, isolated PID/network namespaces,
  MemoryMax=2G and no swap. Temporary small-model GPU jobs had MemoryMax=4G;
  Qwen jobs used 8G. Full-context validation also used a 12 GiB host-memory
  reserve watchdog. Timeouts and cgroup cleanup applied to owned test processes.
  A cgroup cannot isolate the desktop from every GPU driver failure.

## Final automated and package checks

| Check | Result |
| --- | --- |
| Python 3.10 complete core suite | 1128 passed; live marker deselected for its separate native run |
| Python 3.12 complete core suite | 1128 passed; live marker deselected; one Starlette deprecation warning |
| Python 3.14 complete core suite | 1128 passed; live marker deselected |
| Opt-in native inference smoke | Passed separately; real CLI child, JSON, SSE, and signal shutdown |
| Separate vision companion suite | 86 passed on each of Python 3.10, 3.12, and 3.14; no skips; dependency deprecation warnings on 3.12 |
| Deterministic Chromium regression suite | 10/10 passed; these checks use controlled API responses |
| Ruff / mypy / JS syntax / lock consistency / diff whitespace | Passed; mypy checked 48 source files |
| Wheel and sdist | Built; wheel built from sdist; final archives inspected for required assets and unwanted environments/configuration |
| Fresh Python 3.10 core-only wheel | Version/help commands passed; textual, fastembed, and mcp absent |
| Published GitHub RC1 wheel → final candidate | Existing config bytes and folder chat preserved; appended history saved/read after upgrade |

No core tests were silently skipped for missing developer-specific GGUFs.
The former metadata/MTP fixture skips now use generated GGUF-format metadata
files. These test metadata and argument construction; native inference uses
real downloaded or existing models. The marked native test is a separate job,
not omitted coverage. Native Windows was not run.

## Real installed-wheel HTTP and UI checks

The final core-only wheel passed **all 12 groups** against real Uvicorn TCP
listeners and native Vulkan model inference. These were not mocked backend
responses:

1. Admin token rejection/acceptance, actual example-plugin discovery/routes,
   bundled static assets, OpenAI model discovery, and Ollama tags.
2. Malformed requests, unknown/missing models, startup diagnostics, deliberately
   rejected VRAM fit, and healthy inference after recovery.
3. OpenAI chat/completions, JSON/SSE, early stream close, drained request counts,
   and a subsequent successful request.
4. Ollama chat/generation in JSON and NDJSON with final completion markers.
5. Eight simultaneous cold requests; one backend load and one resident child.
6. Real Nomic embedding vectors for two inputs, then swap back to generation.
7. Chat creation/update/folders/export/delete/import/move; invalid updates
   preserved existing history.
8. Actual second Arc server using the first as an upstream; discovered ownership,
   JSON inference, and streamed proxy inference.
9. Recipe edit, persistence, reload/inference, and original recipe restoration.
10. Real installed CLI benchmark and tuning dry run; original recipe restored.
11. Real headless Chromium with actual API/model streaming, context meter,
    history/draft reload persistence, dashboard/chat assets, and external
    requests blocked throughout. Separate browser regressions cover additional
    settings/error interactions using controlled responses.
12. Real Textual headless app reading authenticated live status and refreshing
    the model table.

The model used for this suite was freshly downloaded
Qwen2.5-0.5B-Instruct-Q4_K_M. Its configured context was 32768; the suite also
edited and restored its temporary context. The embedding model was actually
fetched from its official HF repository. Final stop-all/shutdown assertions
confirmed test child cleanup.

## Real external chat client

The cached **OpenWebUI v0.3.35** container also passed an actual Chromium
workflow against native Arc/Vulkan inference: model discovery and selection,
streamed chat returning `4`, displayed answer, and saved-history reload.
Its automatic title/tag requests also reached the actual model. This used
loopback endpoints, scratch database state, a 1.5 GiB/no-swap CPU-only
container, and its supported remote-embedding mode. No API response was mocked.
The initial offline startup tried to load an uncached local embedding model;
remote-embedding configuration resolved that external client setup issue.
Two initial browser assertions used a nonexistent class or a hidden duplicate
text element. Screenshots already showed the correct answer; corrected visible
body assertions passed on a complete repeat. These harness failures are retained.
This confirms the installed client version, not every OpenWebUI release.

## Real first-run and failure workflows

- Fresh configuration: actual GPU detection, portable runtime download,
  Hugging Face GGUF download/registration, fit estimation, and setup-only.
- Actual installed `arc-llama run` with a hard-linked GGUF filename containing
  **spaces and Unicode**: startup, model list, JSON generation, SSE, and native
  signal shutdown. The registered launch path itself contained those characters.
- Real forced runtime re-download interrupted after receiving bytes; previous
  runtime SHA-256 and configuration bytes stayed unchanged, staging cleaned up.
- Real HF model download interrupted after partial bytes appeared; no completed
  GGUF published, existing configuration and prior model preserved. HF's partial
  cache is resumable; it is not treated as a registered complete model.
- Real loopback HTTP corrupt runtime payload rejected by checksum verification;
  failed destination removed. Real unsafe tar archive rejected before extraction.
- Real malformed, structurally invalid, and oversized registry files rejected.
- Installed CLI rejected the embedding GGUF as an incompatible speculative draft
  and left configuration unchanged. Successful draft-MTP native inference is
  covered by the large Qwen runs below.
- Actual `doctor` detected the B60/xe/Vulkan installation. It also reported
  render/video group guidance; those warnings did not prevent the native runs.

## Real container runtime

A candidate wheel installed over the cached SYCL runtime image passed real
JSON inference, SSE with completion marker, the stdlib Docker health check,
stop-all, and clean SIGTERM shutdown (exit zero, no container OOM). It used
Qwen2.5-0.5B at 4096 context, backend fingerprint **b1-61881b1**, a 4 GiB/no-swap
container, the real B60 device, and the production resident lock. The installed
candidate entrypoint, container bind environment, and health command were used;
upstream llama.cpp/oneAPI compilation was reused from the cached image, not
rebuilt. This tests container execution, not a new compiler build.

The first product-level container attempt returned healthy inside Docker but
its published host port was unreachable because Arc listened on container
loopback. `ARC_LLAMA_HOST=0.0.0.0` now supplies the container default; explicit
CLI/env overrides remain available. Documented host publication uses
`127.0.0.1:11437:11437`. The complete native test passed after this fix.
A separate earlier test-layer attempt omitted entrypoint execute permission;
that harness was corrected to match the repository's existing chmod step.
Container `/proc` sees its own PID namespace; the shared resident file lock
protected this run, but this does not certify host-wide external DRM-process
visibility from an arbitrary container namespace.

## Native SYCL, full context, vision input, and swaps

Existing Qwen3.6-35B-A3B UD-Q4_K_XL with embedded MTP and the same vision
projector was tested through real loopback HTTP, using **131072 context, q8_0
KV, full GPU text-layer offload, no-mmap, and draft-MTP**. The local recipe's
single GPU placement adjustment is `--no-mmproj-offload` (CPU vision projector).

- Actual Qwen → LFM2.5 → Qwen swaps, JSON and streaming answers, and red-image
  recognition passed in the earlier bounded native run.
- Later full-context prefill used **130429 actual prompt tokens**. Streaming
  returned `4`, with 130638 total tokens. JSON returned `4` with 130638 total
  tokens, reusing 130425 cached prompt tokens; this was cache reuse, not another
  cold full prefill. Red-image recognition and cleanup still passed afterward.
- Lowest sampled host MemAvailable in the final full-context run: **18899 MiB
  (~18.46 GiB)**; the earlier passing run reached 18050 MiB (~17.63 GiB). No OOM or xe/GuC reset/timeout messages appeared in the checked
  kernel test window. These measurements apply to the tested workload, not every
  possible long-lived workload or driver version.

The earlier all-GPU-projector recipe was deliberately not repeated after its
host disruption. Its exact original failure is still not conclusively
attributed. Passing the tested CPU-projector configuration does not prove that
old recipe safe.

## Agent and real optional integrations

- Qwen executed actual write/read tool calls, requested approval, created a
  checkpoint, and restored the original scratch file.
- Real `/v1/agent` SSE flow: generated a plan, accepted plan approval, waited for
  manual tool confirmation, wrote/read the file, cleaned pending approval state,
  and allowed checkpoint rollback. Denied plans performed no writes.
- Real interactive agent retained the given word across two conversation turns.
- A separate real two-write run approved the first write and denied the second.
  The second confirmation waited for a fresh decision, and the denied file was
  absent.
- Final installed-wheel HTTP returned 400 for malformed agent objects,
  noninteger turn counts, and approval strings instead of treating them as true.
- Real FastEmbed model/ONNX semantic indexing and search passed.
- Real MCP stdio handshake, tool discovery, call (`20+22=42`), unregistration,
  and shutdown passed with the supported SDK. A deliberately failing real MCP
  child also cleaned up after failed initialization.
- The small 0.5B model first omitted a requested tool call and later supplied
  incorrect arguments. Those failed model-behavior attempts remain in the logs;
  the capable Qwen completed the same integration paths. No agent framework
  change was made to hide a model deciding to stop early.
- The specified OpenCode cloud buddy returned HTTP 402. Its attempt changed no
  code; the manager implemented/reviewed the bounded agent-validation fix.

## Vision companion real rendering

The separately installed companion wheel completed an actual **512×512,
2-step CPU ComfyUI render**, using the existing Flux2 Klein 9B GGUF, text
encoder, and VAE. The returned PNG was visually inspected. Lowest sampled
host MemAvailable was 18538 MiB; generation took 469.2 seconds.

A fresh environment containing the actual core and companion wheels also
passed the complete **browser → core plugin → companion → ComfyUI → PNG →
queued native text** chain. Plugin discovery used its installed entry point.
The browser supplied only a prompt, discovered the advertised image model,
and displayed the real rendered 512-pixel image. An unknown image model was
rejected before evicting resident text. During the render, a concurrent text
request waited without loading a backend; after image completion, the text
backend reloaded and answered correctly. Leases and children cleaned up.
This run used Qwen2.5-0.5B/Vulkan, took 336.1 seconds for image generation, and
had minimum MemAvailable 19590 MiB.

Both renders hid GPU devices from ComfyUI, used read-only host files plus
scratch output/database paths, no swap, CPU/memory/time limits, and a 12 GiB
host-memory reserve watchdog. The earlier default **20-step** CPU run was
interrupted for runtime and is not a passing result. The complete render
pipeline passed at two steps; default-step performance is not certified.
ComfyUI XPU rendering was not tested.

The complete chain then passed again with the actual **Qwen3.6-35B-A3B
UD-Q4_K_XL / SYCL** model, preserving 131072 context, q8_0 KV, no-mmap,
draft-MTP, full GPU text offload, and CPU projector. It confirmed the same
invalid-model preservation, exclusive eviction, queued text, rendered browser
image, model reload, answer, and cleanup. Minimum MemAvailable was **19319 MiB
(~18.87 GiB)**. Image generation used the same two-step ComfyUI CPU workload.

## Fixes found by this validation

- Malformed OpenAI request bodies raised exceptions: validate object/model/stream.
- Ollama generation forwarded chat messages to the completions endpoint: send
  the completion prompt.
- Recently closed Linux backend sockets falsely blocked cold reloads in
  TIME_WAIT: use POSIX SO_REUSEADDR for the preflight probe; preserve Windows
  exclusive binding and active-listener rejection.
- TUI failed against authenticated servers: use its explicit token or the
  permitted loopback session-token endpoint.
- MCP 2.x was outside the code's SDK contract: pin optional dependency below 2;
  clean sessions/clients on failed startup.
- Agent body/field/approval shapes could cause 500s or truthy-string approval:
  validate before starting a run or changing approval state.
- Image leases previously left new text loads free to race with image work:
  admit local inference and model loads through shared leases, hold them through
  response cleanup, and close admission before exclusive draining/eviction.
  Nested loads reuse an active scope; cross-task streaming cleanup and cancelled
  waiters are covered. Shared work that misses the drain deadline rejects the
  exclusive request rather than overlapping it.
- Image UI requests omitted a model but the plugin supplied an invented ID:
  discover the companion's advertised model and validate explicit choices before
  eviction. Preserve backend errors and return 503 on transport/drain failure.
- Model-wait/TTFT metrics omitted image admission delay: start the request clock
  before admission. A real held lease plus native text request recorded the wait.
- Docker published ports were unreachable with container-loopback binding:
  set the container bind default and document local host publication; native
  container inference and shutdown then passed.
- Python 3.14 Path.is_file changes invalidated a strict stat-call-count assertion;
  adjust only that instrumentation expectation, keeping cache behavior checks.

The final suites and broad installed-wheel HTTP checks were repeated after
these fixes. Existing ownership/lifecycle safeguards also remain covered.

## Scope and saved evidence

Windows is the excluded native platform. Linux hardware execution covers the
available **B60**, both Vulkan and SYCL. No physical Alchemist card was available;
its automated architecture paths passed, but that is not native hardware
certification. Audio companion support is a documented extension contract, not a shipped
audio backend; its contract coverage does not certify an external transcription
service. ComfyUI XPU rendering is not claimed by the adapter's
CPU-tested support statement and is not certified here.

The restarted normal Arc service returned HTTP 200 for model discovery and
authenticated status, with 16 registered models. No model was started by that
health check. Temporary limits apply to tests only, not permanent service policy.

Logs, harnesses, results, and a SHA-256 source manifest are retained at
`/home/slowe/Documents/Codex/2026-09-28/l/arc-full-e2e/`. That evidence excludes
model weights, environments, caches, and private user configuration. Test
scripts still reference their temporary staging paths and the local models.
The GitHub candidate CI/Windows release gates remain outstanding because this
working tree has not been committed or pushed.
