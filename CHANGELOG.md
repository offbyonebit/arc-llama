# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-08-24

### Fixed
- `arc-llama tune` now survives backend crashes/OOMs during candidate measurement instead of aborting the whole sweep. (#58)

### Changed
- Merged GitHub `main` back into the release branch so PyPI 0.7.0 and the repository are consistent.

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
