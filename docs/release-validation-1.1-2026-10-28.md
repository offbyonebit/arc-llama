# Arc Llama 1.1 local validation — October 28, 2026

This extends the [October 25 record](release-readiness-1.1.md). The candidate stays `1.1.0.dev0` on `release/1.1`, based on `7973b9e` plus preserved local work. No release, tag, commit, push or service restart is part of this validation.

## Changes made during validation

- Fixed guided setup disappearing after a reload once the download had registered the model but setup was paused. Continue now survives reload; the browser regression reproduces this state.
- Replaced age-only Undo cleanup with ownership evidence and an OS lock held while Undo remains available. A subprocess test proves an old completed run is protected while alive and reclaimed after abrupt exit. Unknown legacy folders remain untouched. Marker initialization also handles the byte seeded by the Windows lock implementation; native Windows remains untested.
- Disconnected observers when Work step views are replaced; a browser regression checks that detached step views are no longer observed.
- Hardened the streaming adapter against late deltas and inconsistent completed text. These checks use synthetic protocol fixtures; OpenCode was not used for implementation or live validation.
- Repaired the stale MCP lock entry to match the existing `mcp>=1.30,<2` requirement. Set the async fixture loop scope explicitly and declared its minimum plugin version. Made the packed Landlock ctypes layout explicit for Python 3.14; updated filesystem traversal test instrumentation for Python 3.14's `Path.lstat` implementation.
- Refined the installed Work retry test: a blocked path does not guarantee that a model calls the failing tool. Failed verification and attempt selection are asserted directly. Fault runs seed an explicit partial A.txt fixture; the separate normal run requires the model to create both files. Reports retain injected fixtures and observed tool errors.

## Validation and evidence

All test services use isolated state and synthetic conversations. Installed checks run outside the checkout with an explicit site-packages import check. Large inference runs use bounded memory/CPU scopes and timeouts.

Final results: `scripts/check.sh` passed **2040 tests, 5 expected skips and 102/102 browser cases**. Installed-wheel suites on Python 3.10 and 3.14 each passed **2044 tests, 1 expected skip**, with warnings treated as errors. Clean installs passed on 3.10/3.12/3.14; all 91 modules imported or produced the two expected TUI install hints on 3.10/3.14.

Installed CPU, final-wheel Vulkan/SYCL, Work normal/retry/keep-best, MCP and Hermes checks passed within the scopes below. The five source-suite skips are four optional document-library cases and one opt-in inference test. Both matrix environments installed the document extras.

Artifact hashes and detailed results are recorded with the evidence, rather than embedded in the source distribution that they identify:

- Evidence directory: `/home/slowe/Documents/Codex/2026-10-23/do-you-have-access-to-the/validation-1.1-2026-10-28`.
- Local distributions, temporary environments and hardware logs: `/mnt/storage/tmp/arc-llama-1.1-validation-20261028`.
- `artifact-manifest.json` identifies the final wheel/sdist; `final-results.json` records completed checks.

| Check | Scope |
|---|---|
| `scripts/check.sh` | Ruff, mypy, JS syntax, Python suite and 102 deterministic browser cases |
| Installed-wheel full Python suite, `-W error` | Python 3.10 and 3.14; optional document libraries installed |
| Fresh installs without extras | Python 3.10, 3.12 and 3.14; eight CLI entry checks each; textual/fastembed/MCP remain absent |
| Installed CPU JSON/SSE and browser smoke | Pinned llama.cpp b11381; exact 32-token native budget, per-slot context 4096, compaction, continuation and undo transcript |
| Real 1.5B Workbench | Normal, retry and keep-best; review, reload, apply and undo; original project restored |
| MCP 1.30.0 | Actual SDK stdio fixture, text once, tool error, image, agent call and unregister on stop |
| Hermes ACP | Synthetic local inference: denied/approved edits, streaming/context and cancellation without writes |
| Build/package audit | Source/asset inclusion, local-only exclusions, sdist build, lock consistency |

The guided first-answer, 1.5B Work and MCP checks ran on the preceding wheel. A final ownership-marker initialization correction changed only `agent/changes.py`; all 136 other packaged files were byte-identical. The full suites, installed CPU smoke and Vulkan/SYCL probes ran against the final wheel. `scoped-rerun.json` records that boundary; the Windows-marker regression and subprocess lock test cover the final correction. Hermes ACP passed its functional assertions but logged a graceful-shutdown timeout warning during fixture teardown; no successful teardown timing guarantee is claimed.

### Native first-answer flow and GPU attribution

A clean installed-wheel configuration detected the Arc Pro B60 (24 GiB), installed the verified official Vulkan runtime b11969, and selected a model from the live catalog. The result was **Ornith-1.0-9B Q4_K_M**, 5,629,108,704 bytes, MIT license, pinned commit `3296bc7a404871a72ac3f1903f561459c09b5c17`. This is an observed dynamic selection, not a shipped recommendation or a general model-quality endorsement.

The real download showed byte/stage progress, paused after registration, survived reload, continued, loaded and answered “Hello, how are you doing today?” Browser access to external hosts was blocked during the first-answer check. This proves the local answer flow; it does not prove a completely disconnected operating system during runtime/model acquisition. Screenshots and phase records are in `native2/`. The initial failing pause/reload run is retained as a negative control.

Separate installed-service probes produced nonempty JSON and SSE answers with normal `stop` completion on:

- Vulkan b11969 with the dynamically selected 9B model.
- SYCL b11624 with the known-compatible Qwen2.5 1.5B Q4_K_M model and the existing local oneAPI installation.

For each probe, only the test service's own llama-server descendants were inspected. Their DRM records identify `xe`, PCI `0000:06:00.0`, resident B60 VRAM and increasing compute cycles across a request. This is direct device activity evidence on a machine that also contains an NVIDIA GPU. Request times include warm caches and differ in model/runtime; they are **not** a comparative performance benchmark.

## Remaining release gates

- **Native Windows:** the available Windows peer is offline. No Windows result is claimed, including installer recovery/locking and harness installation. The prepared diagnostic script and checklist remain local work to review/run there.
- **Hosted CI:** no remote run of this uncommitted candidate. Linux equivalents were run locally; a Windows CI pass still requires a later authorized push/PR.
- **Hardware breadth:** B60 only. No Alchemist, other vendors, multi-GPU or broad driver matrix. The optional optimized Arc Llama Runtime installer was attempted with the final wheel, but its latest-release endpoint returned 404 (`No Arc Llama Runtime release 'latest' found`). This does not distinguish an inaccessible repository from an absent release; no optimized binary was installed or exercised. SYCL validation used an existing older runtime with a different model, not automatic installation of every runtime flavor.
- **Harness breadth:** Hermes ACP used isolated synthetic inference. No live OpenCode check was run, per the user's instruction. This does not validate every harness download/install path.
- **Dependencies:** MCP SDK 2.x remains outside the supported 1.x range. Optional document libraries and the opt-in live-inference pytest have explicit skip reasons; independent installed inference covers the latter's practical path.
- **Scope of guarantees:** installer subprocess interruption tests do not establish power-loss durability or exhaustive installer concurrency. Keep the existing documented limitations.
- **Publishing:** final version, release tag, upload, release notes publication and service rollout remain intentionally undone.

The October 25 upgrade/rollback rehearsal is historical evidence. It was not rerun in full here; configuration preservation tests still run in the full suite. No schema migration was introduced by the changes above.
