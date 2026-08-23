# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Speech-to-text.** `POST /v1/audio/transcriptions` is now served alongside
  your LLMs, so one endpoint covers both chat and STT for clients like Home
  Assistant. Accepts multipart uploads (the OpenAI/Whisper convention) and the
  JSON server-local-path form, and passes `stream=true` through to models
  registered with `--mode streaming`. Backed by `llama-server -m … --mmproj …`,
  which upstream gained along with Qwen3-ASR support — the binary you already
  have, with a **SYCL** build, reusing the arch env profiles and device
  selector.
- **Text-to-speech.** `POST /v1/audio/speech` implements OpenAI's shape
  (`input`, `voice`, `response_format`, `speed`, `instructions`) and returns
  encoded audio, so the OpenAI SDKs and Home Assistant's TTS platform work
  unmodified. `wav`/`pcm` are written from the standard library and always
  work; `mp3`/`opus`/`aac`/`flac` go through libsndfile with an `ffmpeg`
  fallback.
- **Pluggable TTS engines.** An engine is one class — `build_plan` says how to
  launch a backend, `build_payload` maps an OpenAI speech body onto it — and
  registering it in `arc_llama/tts/` is the whole integration. The router
  lifecycle, eviction, VRAM fit guard, endpoint and in-flight accounting are
  engine-agnostic, and nothing outside that package dispatches on an engine
  name. `arc-llama audio engines` lists what a build has.
- **OmniVoice** ([k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)) as the
  first TTS engine: zero-shot voice cloning and voice design across 600+
  languages. It is a Python library rather than a binary, so arc-llama runs it
  in a sidecar under its own interpreter (`paths.tts_python`, or
  `arc-llama audio set-python`) — keeping torch and transformers out of
  arc-llama's environment, and letting eviction actually return its VRAM.
  A model may be named by Hugging Face repo id; the engine resolves it.
- **A voice registry.** `arc-llama audio voice add/list/rm` registers voices
  cloned from a reference clip (`--ref-audio`, `--ref-text`) or designed from
  attributes (`--instruct`). Voices are top-level config, not per-model, so the
  same clip names the same voice across engines. `--alias` makes clients that
  hardcode OpenAI voice ids resolve to yours, and an unrecognised `voice` falls
  back to the model's `default_voice` rather than failing the request. Adding a
  voice reaches a running backend without a reload. `--auto` registers a name
  for the model's own voice, which is what a fine-tuned model wants — its
  speaker is in the weights, so a clone or design prompt would fight it. Encoded reference prompts
  are cached per model, so the first use pays the encode and later starts do not.
- **Quantized (torchao int8) speech models.** A directory containing
  `quantized_state.pt` is detected and loaded by materialising the base model,
  re-applying `quantize_()` to `llm` and `audio_heads`, and reading the saved
  tensors into that structure — `from_pretrained` cannot do it alone, because
  quantized Linear weights are tensor subclasses that `save_pretrained` will
  not serialise. `base_model` and `compile` are settable via `--option`, and
  `dtype` defaults to `bfloat16` for such checkpoints. Missing or mismatched
  weights are a hard startup failure rather than a silent fall back to the
  base model, which would serve fluent audio in the wrong voice.
- A `tts` optional dependency group (`pip install "arc-llama[tts]"`) covering
  omnivoice, torch, torchaudio, torchao, numpy and soundfile. For work in this
  repo, `tool.uv.sources` routes torch/torchao/Triton to Intel's XPU index so
  `uv sync --extra tts` does not silently install the CUDA build.
- `arc-llama audio add <hf-spec> --from-hf` downloads an ASR repo's weights
  *and* its `mmproj-*.gguf` projector together and wires both up; for local
  files the projector is auto-detected when it sits beside the weights.
- Qwen3-ASR's native output framing (`language English<asr_text>…`, which
  llama.cpp forwards verbatim per ggml-org/llama.cpp#26749) is stripped from
  transcripts by default, since it otherwise breaks intent matching in voice
  assistants. Disable with `--no-strip-markers`.
- New `[[audio_models]]` and `[[voices]]` config tables, and the
  `arc-llama audio` command group. Audio models live in their own table so
  `scan`, `tune` and the auto-tuner never pick one up.
- `ServerCaps` now probes for `--mmproj`, so a `llama-server` built without
  multimodal support is refused at registration and at launch with an
  actionable message rather than silently transcribing nonsense.
- `LaunchPlan` carries an optional `health_timeout`. A speech backend may
  import torch and download several GB before answering `/health`, which the
  120 s budget sized for SYCL JIT does not cover — and a timeout firing
  mid-download looks exactly like a backend that never started.

### Fixed
- **ASR models no longer allocate a context they never use.** `llama-server`
  defaults to `-c 0` ("the GGUF's trained context") and Qwen3-ASR advertises
  65536, so a 1.7B q8 model was reserving several GB of KV cache — more VRAM
  than the 27B beside it. arc-llama now pins `-c` (4096 by default, in the
  recipe) and `-np 1`, and the fit guard accounts for the KV cache instead of
  just on-disk size.
- **Speech models are no longer over-charged against the VRAM budget.** A
  Hugging Face cache stores each weight once in `blobs/` and exposes it through
  `snapshots/<rev>/` as a symlink; `Path.stat()` follows symlinks, so the size
  walk counted every byte twice and the fit guard refused LLM loads that fit
  with room to spare. Sizes are now de-duplicated by inode, and only the
  checked-out revision is measured rather than every revision in the cache.
- **A TTS model addressed by Hugging Face repo id is measured at all.** The
  cache lookup guarded with `os.sep in repo_id`, which is always true on POSIX
  — `os.sep` is the same `/` a repo id uses — so it rejected every repo id and
  returned "size unknown" for precisely the models it exists to size.
- The VRAM fit guard's error is itemised: it names each co-resident and its
  estimate instead of only a total, since which model is mis-estimated decides
  the fix, and points at `vram_mb` as the override.
- Audio launch failures explain themselves: `/admin/status` carries a
  `launch_error` per model (shown in the web UI), and a sidecar that dies on
  `ModuleNotFoundError: omnivoice` or a torch without XPU support now comes
  with the fix rather than a bare traceback.

### Changed
- Audio models are **pinned by default** (`always_resident`), in both
  directions: they are exempt from single-resident eviction, *and* loading one
  evicts nothing. Being merely un-evictable was not enough — the first speech
  request still displaced the LLM, so the utterance paid exactly the cold start
  pinning exists to avoid and the speech model ended up pinned in its place.
  Footprints are still charged against the GPU's VRAM budget, and a pinned
  model that cannot fit alongside is refused with its options named rather than
  evicting its way in. `arc-llama audio add --swappable` opts out.
- `arc-llama doctor` reports the speech backends in use: whether `llama-server`
  has multimodal support, missing model or projector paths, missing voice
  reference clips, and each TTS engine's own prerequisites — for a Python
  engine, that its interpreter can actually import it.
- `arc-llama scan` now skips audio GGUFs instead of registering them as LLMs
  with a meaningless recipe: `mmproj-*.gguf` projectors, GGUFs with no
  `general.architecture`, and ASR weights sitting beside a projector.
  (Qwen3-ASR reports `architecture: qwen3vl`, so metadata alone cannot
  distinguish it.)
- Config schema version 2, adding `[[voices]]` and the TTS recipe fields
  (`python`, `device`, `dtype`, the `default_*` request fallbacks, `options`).

## [0.7.0] - 2026-08-19

### Added
- Memory-aware context/batch defaults now respect `ARC_LLAMA_MAX_CTX` as a global ceiling on auto-suggested context length. (#55)
- `arc-llama add` gained `--batch-size` to complement `--ubatch-size` and prints the chosen `ctx`/`ub`/`b` values after registration. (#55)
- SYCL launches now auto-detect and source Intel oneAPI `setvars.sh` when the runtime libraries are not visible, supporting non-standard install prefixes such as `/mnt/storage/opt/intel/oneapi`. Set `paths.oneapi_setvars` in the config to override the auto-detected path. (#56)
- Added an opt-in inference smoke test (`tests/test_smoke.py`). Set `ARC_LLAMA_SMOKE_MODEL` to run it locally on a machine with a model and Intel Arc GPU.
- Windows GPU detection via WMIC with PowerShell fallback.
- Windows ggml backend sibling discovery (`ggml*.dll`).
- Windows oneAPI `.bat` setvars sourcing and Level Zero DLL discovery.
- GitHub Actions CI matrix for Ubuntu and Windows.

### Changed
- Dockerfile now pins llama.cpp to `b10280` (was `b9946`). (#54)
- CI now runs `mypy src/arc_llama` and builds the Dockerfile on every PR.
- README updated with Windows-specific setup instructions.

### Fixed
- `.dockerignore` no longer excludes `docker-entrypoint.sh`, so the Dockerfile can build again.
- Cleaned up mypy errors in `detect.py`, `runtime.py`, `server.py`, and `cli.py`.
- Hardened `Config.save()` for concurrent Windows renames.
- Strengthened agent tool path traversal check for Windows drives.
- Tests now pass on Windows (library prefixes, `.dll`/`.bat` detection, symlinks, `pwd`, `curl`).

## [0.6.2] - 2026-08-11

Identical code to 0.6.1; packaging fix only. The 0.6.1 sdist accidentally included a local virtualenv (`.venv-test/`), whose absolute symlinks broke the wheel build, so 0.6.1 shipped no wheel and a 62 MB sdist. The sdist now explicitly excludes local virtualenvs and scratch files. If you pinned 0.6.1, move to 0.6.2.

## [0.6.1] - 2026-08-11

Bugfix release clearing the remaining concurrency and leak findings from the full-source audit. No new features, no config or API changes.

### Fixed
- A streamed request whose BackgroundTask never ran (mid-stream upstream death, shutdown with an open stream) leaked the global in-flight counter permanently — and since the auto-tuner gates every sweep on that counter, one leaked stream silently disabled background tuning until restart. The decrement now runs from the body generator's `finally`, is idempotent, and floors at zero. (#27)
- Model eviction no longer kills an incumbent's generation mid-stream: `_evict_for` waits (bounded) for the incumbent's per-model in-flight count to drain, and `rebuild_model` does the same before applying a recipe edit to a running model. (#28, #31)
- The lock-free fast path in `ensure_active` could hand out a server that an eviction was about to stop, or be missed by the drain because the request hadn't been counted yet. Acquisition is now atomic with the readiness check, and a `_stopping` mark closes the teardown window. (#29)
- The auto-tuner could start a benchmark load that evicted a just-arrived real request; `measure()` now re-checks the abort hook after the edit POST, the only window where that request can register. (#30)
- The end-of-sweep recipe restore no longer leaves the last candidate persisted when the edit endpoint fails: it retries, then falls back to a direct config write. (#37)
- Upstream proxying leaked the `httpx.AsyncClient` connection pool on every call, and the response too on mid-stream failure; both are now closed exactly once. (#38)
- `Autotuner.start()` is serialized with `stop()`, so a racing start/stop can no longer kill the loop or orphan a task. (#34)
- The autotuner reads running models via `Router.running_models()` instead of poking `router._servers` directly. (#33)
- `tune --dry-run` no longer marks the model as tuned (#22); draft-MTP is no longer auto-enabled on hybrid SSM models (#23); health/admin endpoints report a model as loaded only once its health check passed (#40); the deferred restore no longer consumes shared abort signals (#32); config writes are atomic (#36).

## [0.6.0] - 2026-07-30

### Added
- Background auto-tuner. Once a model has been used and the router is idle, `arc-llama serve` runs the same staged KV/ubatch/flash-attention sweep that `arc-llama tune` uses, over loopback HTTP, aborting instantly when a real request arrives. The tuned recipe is written via the existing `/admin/models/{name}/edit` endpoint.
- `[tune]` config section (`auto`, `idle_seconds`, `target`, `prompt_tokens`, `gen_tokens`, `min_uses`, `retune_on_fingerprint_change`) and per-model tune state ([`tune_state`, `tuned_at`, `tune_fingerprint`, `tune_error`]) persisted in the existing config.
- Fingerprint-based invalidation. A fingerprint covers the model file, `llama-server` binary, GPU `pci_slot`/`arch`/`backend`, arc-llama version, and a `TUNE_SCHEMA_VERSION` constant. Stale tuned recipes are treated as `untuned` again without a time schedule.
- Admin endpoints: `GET /admin/tune/status`, `POST /admin/tune/{name}` (queue now), `DELETE /admin/tune` (abort running sweep). `tune_state` is exposed in `GET /admin/status` model entries for the web UI/TUI.
- `arc-llama serve --auto-tune` / `--no-auto-tune` CLI overrides and an auto-tune banner at startup.
- `arc-llama tune --status` prints the per-model tune state table without measuring.
- Manual `arc-llama tune` (single model and `--all`) now records the same fingerprint and `tuned` state via a shared helper.

### Changed
- `tune_model` now accepts `should_abort` and `on_stage` callbacks. The recipe restore is wrapped in a `try/finally`, and `TuneReport.aborted` reports whether the sweep was abandoned mid-stage. Aborted sweeps restore only winners from fully completed stages.
- `Router` exposes `last_activity` and `inflight` counters; request completion bumps `last_activity` so long generations count as activity.
- MoE expert offload (`n_cpu_moe`) is now a measured tune axis: for MoE models that need offload, the sweep computes the minimum feasible layer count from per-layer expert-tensor GGUF accounting (against the winning KV type and workload context), measures it, and probes one step below/above. The `/admin/models/{name}/edit` endpoint now accepts `n_cpu_moe`, and ubatch candidates that would not fit under the chosen offload are skipped rather than measured into an OOM. `TUNE_SCHEMA_VERSION` bumped to 2, so previously tuned recipes are retuned.
- Registration-time MoE offload suggestion now uses the same VRAM estimator as the load-time guard (minimum feasible `--n-cpu-moe` *layer* count from tensor metadata), replacing the file-size/expert-count guess — `--n-cpu-moe N` is a layer count, not an expert count.

### Fixed
- Exceptions or `CancelledError` during a tune sweep no longer leave the last losing candidate's recipe persisted — the original/winning recipe is restored in `finally`.
- The router's VRAM fit guard now subtracts the expert tensor bytes that `--n-cpu-moe` keeps on the host, instead of counting full weights and refusing exactly the models expert offload was enabled to rescue. When offload bytes cannot be determined, the guard warns and permits the load rather than silently disabling the feature.

## [0.5.0] - 2026-07-23

Highlights: `arc-llama install-runtime` downloads a prebuilt portable Vulkan llama-server, removing the need to install oneAPI or build from source.

### Added
- `arc-llama install-runtime` command to download a prebuilt llama-server from official ggml-org/llama.cpp releases. Vulkan is the default backend, portable, and requires no oneAPI installation. SYCL remains optional. Verified end to end on Battlemage B60.
- `arc-llama tune` staged autotuner for KV cache type, ubatch, and flash attention. It measures performance on your card and writes the winning recipe. Added `arc-llama tune --all` to sweep every registered model in one run.
- `arc-llama benchmark` harness for prompt-eval and generation tokens per second.
- Vulkan backend support via `backend = "vulkan"` in the config, alongside SYCL.
- Windows support for the launcher, CLI, and config paths. CI is green on windows-latest and ubuntu-latest for Python 3.10 to 3.12.
- Auto-scan for new GGUFs on `serve` startup, and auto-detection of sidecar speculative-draft (MTP) models.
- Experimental agent loop, terminal agent UI, MCP client, checkpoints, chat persistence, and chat export/import. Gated behind `ARC_LLAMA_EXPERIMENTAL_AGENT`.

### Changed
- `arc-llama init` now writes a config even when no llama-server is present, and points users to `install-runtime`.
- `init` and `install-runtime` set each GPU's `backend` to match the actual binary. Previously, it always defaulted to SYCL, which mismatched Vulkan builds.
- `doctor` now surfaces AOT build guidance, device-ID VRAM fallback, metadata-based KV class, crash-log surfacing, config migration, and Prometheus-style metrics.

### Fixed
- Backend detection now scans sibling `libggml-*.so` shared libraries for modern modular builds, and no longer false-matches bare backend NAME strings in `libggml.so`. Previously, a downloaded Vulkan build reported its backend as "unknown", then as "sycl".
- Benchmark generation and prompt-eval measurement accuracy.
- MTP `ubatch_size` regression.
- MoE detection.

## [0.4.0]
Initial public releases.
