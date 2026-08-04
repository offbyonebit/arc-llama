# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
