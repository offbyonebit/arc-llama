# Verified investigation: host RAM OOM on 2026-10-01

Current status (2026-10-03): lifecycle/ownership safeguards are implemented, and bounded live HTTP generation, streaming, vision, and swaps passed with the Qwen vision projector on CPU. The local recipe is updated and the normal Arc endpoint is healthy after restart. Exact attribution of the earlier RAM/driver failure remains unresolved; see the dated follow-ups below.

This review supplements `2026-10-01-host-ram-oom-during-model-swap.md`. The original note is preserved, but its causal conclusions should not be treated as verified. All times below are CDT.

## What differs from the working local proxy

The model's ability to run is not disproved by this incident. The important difference is which competing model processes get stopped.

| Item | Local proxy | Arc Llama |
| --- | --- | --- |
| Before a model change | `_stop_all_except()` stops all other known SYCL/CUDA model services | `_evict_for()` considers only servers in this router's `_servers` |
| Stop mechanism | Awaits `systemctl --user stop`, then waits another six seconds | Awaits child shutdown, with three-second graceful and forced waits |
| Independent Qwen3.8 service | Included in the proxy's stop list | Outside Arc's managed server registry and resident lock |
| Qwen3.6 GGUF | Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | Same file |
| Context / KV / loading | 131072, q8_0 K/V, `--no-mmap`, GPU layers 999, draft-MTP, same mmproj | Same settings |
| Runtime | `llama.cpp/build/bin/llama-server` | Same executable path |
| Flash Attention | Default auto; upstream enables it for quantized V cache | Explicitly on |
| SYCL cache | Persistent cache disabled | Battlemage profile disables persistent cache |

The proxy's Gemma script uses `build-tile/bin`, so the runtimes are not identical for every model. The current source/scripts cannot establish every historical binary version or desktop workload. They do establish that context, no-mmap and MTP are not new Qwen settings introduced by Arc.

## Kernel and service evidence

- Kernel journal records **18 OOM kills between 11:07:03 and 11:14:33**, not seven between 11:11 and 11:14.
- Independent `qwen3.8-27b-server.service` PID **1504** appears in all 18 OOM task snapshots.
- Arc's first Qwen3.6 PID **83150** appears in eight snapshots, through 11:09:55; retry PID **85125** appears in ten, through 11:14:33.
- Old Gemma PID **75695** appears in none of those snapshots. This does not prove there was never a brief transition overlap, but does not support the original claim that Gemma remained resident throughout the OOM storm.
- Arc and the independent Qwen3.8 service both started at **06:46:03**. On inspection both remain active, and both are enabled user services. Qwen3.8 directly launches its script without taking Arc's resident lock.
- Intel `xe`/TTM buffer allocation and eviction appear in OOM stacks, including `ttm_bo_evict`, `ttm_bo_handle_move_mem`, and `xe_ttm_tt_populate`. Swap was exhausted; shared-memory accounting reached roughly 10.9 GiB at the first snapshot.
- A later read-only inspection of PID1504's DRM fdinfo reports approximately **21 GiB of system-location buffer allocation**, but **zero resident system bytes** at that inspection. This is allocation accounting, not proof of 21 GiB currently resident in RAM. Its small process RSS therefore cannot rule out GPU buffer pressure, and the later sample cannot precisely attribute incident-time memory.

The strongest explanation is competing model allocations on the B60, with GPU buffer eviction/backing contributing to host memory pressure. The old proxy's broader eviction policy would remove that independent competitor before loading Qwen3.6. The incident does not demonstrate that the model cannot run alone.

## Confirmed shutdown defect

`src/arc_llama/launcher.py:LlamaServer.stop()` catches the timeout of its final forced-kill wait, then unconditionally clears `self.process` and releases the resident lock. This happens on both Unix and Windows paths.

A safe isolated probe of the actual method mocked all process signaling, waits, taskkill, and lock operations. For each platform, the fake child remained alive after two timed-out waits, while the method discarded its handle and released ownership. No real model was launched or signaled.

The incident is consistent with this defect: retry cleanup returned around **11:12:29**, but PID85125 still appears in the kernel task list at **11:14:33**. Its later zombie state is unreaped-child evidence; a zombie itself does not retain the model's RAM.

Normal router eviction already awaits shutdown. The defect is reporting cleanup as complete when exit remains unconfirmed, rather than simply a missing await before every load.

## Corrections to the original note

1. `--no-mmap` does not require retaining every GPU weight as a full anonymous host copy. Upstream `src/llama-model-loader.cpp` uses staging buffers/chunked transfers for device tensors. Host tensors, temporary buffers and driver backing still require memory.
2. The default 8192 MiB prompt-cache limit is a cap, not an immediate allocation. Summing cache-save log entries does not measure live cache residency.
3. Qwen's failed-load tails do not reach prompt-cache initialization. Its default cache cannot be assumed to have allocated 8 GiB during these attempts.
4. Arc's Python process is not shown to be the dominant consumer. GPU-associated memory is not fully explained by ordinary per-process RSS.
5. An OOM-triggering thread name or chosen victim does not establish which workload caused the pressure.
6. Do not start remediation by shrinking the user's context, disabling MTP, or assuming the model is too large.

## Recommended correction order

1. Establish a single model owner on the B60. Remove the competing legacy service from automatic startup when Arc owns that GPU, and make conflicts visible before launching another model. Arc should not silently kill arbitrary external services; explicitly configured ownership or admission checks are preferable.
2. Retain child handles and resident ownership until exit is confirmed. A forced-stop timeout must prevent a replacement load, remain observable, and allow later cleanup/reaping. Cover both Unix and Windows failure paths.
3. Verify repeated swaps with the same model, context, KV types and MTP settings while sampling GPU allocations, host MemAvailable, swap, and process/cgroup state. Compare against the proxy with the same binary and workload.
4. Evaluate prompt-cache caps only if isolated measurements show material retained cache pressure; consider limits separately from the demonstrated ownership defects.

## Scope and evidence locations

Read-only sources: kernel journal for 11:05–11:15; Arc service journal and state logs; user service units; `/proc/1504/{status,fdinfo,cmdline}`; current Arc launcher/router/architecture code; `/mnt/storage/local-llm-proxy/proxy.py`; Qwen/Gemma launch scripts; local llama.cpp model-loader, context and server sources.

No production source changes, service shutdowns, model swaps, commits or pushes were performed during this investigation. Earlier test cleanup remains uncommitted. A controlled end-to-end run is still needed to establish that the recommended lifecycle changes eliminate the incident under the original workload.


## 2026-10-02 follow-up: reproduction, safeguards, and test safety

The initial contention explanation was incomplete. After the independent Qwen3.8 service was stopped and disabled, loading Qwen through the original proxy script with the current llama.cpp runtime also reproduced a sharp host-RAM drop (roughly 24 GiB available to 3 GiB). Guarded attempts terminated the model process. Later kernel logs showed Intel xe/GuC faults, engine resets, and scheduling timeouts. No additional live model loading is appropriate during this repair phase.

The Qwen text device allocation was approximately 20.8 GiB, with additional KV, compute, MTP, and vision allocations approaching the B60's 24 GiB capacity. GPU eviction/backing into host RAM remains a supported hypothesis; the exact allocation responsible and the difference from the historically working runtime have not been established. Disabling the competing service alone did not resolve the problem. The latest desktop disruption has not been conclusively attributed to either OOM or test signaling.

Implemented safeguards (uncommitted):

- Keep the child handle, log, and resident lock until shutdown is confirmed; report forced-stop timeouts on Unix and Windows.
- Check for independent llama-server allocations on the selected GPU before spawning, and return a structured conflict instead of silently stopping external services.
- Watch Linux host memory during startup and stop the managed load when available RAM falls below the reserve. This is a polling safeguard, not a guarantee against sudden kernel allocation bursts or driver faults.
- Make memory-pressure cleanup take precedence over health success. Await shutdown even after repeated cancellation, serialize concurrent stops, and clean up when cancellation occurs before the health task starts or during final task gathering.
- Correct a unit test that called stop with an invented PID without mocking process-group signaling. Block real kill/killpg/taskkill calls by default in tests, and mock host-memory pressure unless explicitly under test. The unsafe call is confirmed; it is not proof that it caused a desktop crash.

Validation: the complete Python suite ran in a read-only Bubblewrap sandbox with separate PID/network namespaces, no GPU devices, no desktop-session connection, a 2 GiB per-process address-space limit, and a CPU-time limit. Result: **1070 passed, 13 skipped** in 20.93 seconds. Ruff passed for the affected launcher/guard/test files; mypy passed for 48 source files; git diff --check passed. The suite emitted one existing FastAPI/Starlette TestClient deprecation warning. No real model, GPU probe, service change, commit, or push was performed during this repair phase.

Windows shutdown failure paths are covered by mocked tests on Linux; this is not native Windows validation. Live Qwen loading and release readiness remain unverified. The earlier statement that no source/service changes had been made describes the initial read-only review only, not the subsequent investigation and repair.


## 2026-10-03: successful live validation and tested workaround

Real inference was tested through the current checkout with the same llama-server runtime and Qwen3.6 GGUF. A small LFM generation passed first. Two Qwen → LFM → Qwen runs then passed, followed by a third run through an actual Uvicorn loopback HTTP listener (not an ASGI mock). All test instances used the production resident lock through a symlink and separate temporary state/ports; the existing Arc service remained idle during the model tests.

The Qwen launch preserved **131072 context, q8_0 K/V, full text-layer GPU offload, no-mmap, draft-MTP, and the same vision projector**. The single recipe adjustment was `--no-mmproj-offload`, which moves the vision projector to CPU and gives text/MTP allocations more GPU headroom. Image support remains available; its CPU encoding may be slower for larger images. This is a tested workaround, not proof of the exact original allocation responsible.

Containment was part of the live test: a systemd user cgroup with MemoryMax=8G, MemorySwapMax=0, a 180–240 second runtime limit, and a watchdog that stops only managed test children if host MemAvailable drops below 12 GiB. GPU driver failures cannot be fully isolated by a memory cgroup. The earlier unsafe all-GPU projector recipe was not rerun. Therefore, this test does not distinguish the individual contributions of CPU projector placement, hard memory containment, and a fresh host/driver state.

Actual HTTP results:

- Qwen returned a finished answer, “Two plus two equals four,” before and after swapping to LFM.
- LFM returned “2 plus 2 is 4.”
- Streamed Qwen output returned `4`.
- A **4125-token** prompt returned `4`: roughly 937 prompt tokens/s and 55 generated tokens/s for this synthetic sample. This is not a filled-128k-context or long-duration stress test.
- Qwen identified a generated solid-red image as “Red” using the CPU projector.
- The admin stop-all endpoint and app shutdown completed; no test child remained.
- Lowest sampled host MemAvailable was **22377 MiB (~21.85 GiB)** in the actual HTTP run. No new kernel messages appeared during the test window, including no OOM or xe/GuC errors.

The first short-budget pass generated reasoning but truncated before final content; it was followed by strict assertions requiring actual answer content with a 512-token allowance. Those stricter checks passed.

The CPU-projector option was then saved only to the local Qwen recipe, with a private backup of the previous configuration. The idle Arc service was restarted to pick up that option and the current lifecycle fixes. Startup took roughly 74 seconds; the normal `/v1/models` endpoint then returned HTTP 200. Hard cgroup limits applied to temporary validation services only; no permanent systemd memory policy was installed. Evidence and the reproduction harness are saved in `/home/slowe/Documents/Codex/2026-09-28/l/arc-live-validation/`.

This establishes real load/generation/streaming/vision/swap success for the tested configuration. It does not validate the previous all-GPU projector configuration, all registered models, full-context saturation, or native Windows operation. No commits or pushes were made.


## 2026-10-03 follow-up: actual full-context validation

A later run filled the configured 131072-token context with **130429 actual
prompt tokens**, preserving q8_0 KV, full GPU text offload, no-mmap, draft-MTP,
and the same CPU vision projector. Streaming returned `4`; the following JSON
request also returned `4` and reused 130425 cached prompt tokens. The second
request therefore validates cache reuse, rather than a second cold prefill.
Vision still identified the red image after the long-context requests, and
stop-all/shutdown completed. Minimum sampled host MemAvailable was **18050 MiB
(~17.63 GiB)**. The temporary process had MemoryMax=8G, MemorySwapMax=0, a longer
1800-second deadline, and the 12 GiB host-reserve watchdog. No OOM or xe/GuC
reset/timeout messages appeared in the checked kernel test window.

This supersedes the earlier follow-up's full-context limitation for this exact
CPU-projector configuration. It does not identify the original allocation
responsible, certify the old GPU projector recipe, or guarantee that every
future workload/driver behaves identically. See the comprehensive Linux report
in `../release-validation-linux-2026-10-03.md`. Evidence is retained under
`/home/slowe/Documents/Codex/2026-09-28/l/arc-full-e2e/`.


## Final admission and image/text handoff retest

Real companion integration exposed an independent arbitration defect: acquiring
an exclusive plugin lease did not prevent new local text loads from entering.
Local forwarded inference and direct model loads now hold shared admission;
exclusive work closes admission, waits for existing shared work, then evicts.
Streaming cleanup and cancellation retain/release admission correctly, and an
exclusive drain timeout rejects rather than overlapping still-admitted work.

After that fix the 130429-token fresh Qwen prefill, cached JSON request, red
image recognition, and cleanup passed again. Both long-context responses
used 130638 total tokens; the JSON request reused 130425 prompt tokens. The
final run's lowest sampled MemAvailable was 18899 MiB. The earlier successful
run's 18050 MiB measurement above remains valid for that earlier run.

A separate actual browser-to-ComfyUI CPU image render passed with the large
Qwen resident: Qwen stopped before rendering, queued text stayed unloaded
until the PNG completed, then Qwen reloaded and answered. It preserved the
131072 context and existing q8_0/MTP/full GPU text/CPU projector settings.
Minimum sampled MemAvailable was 19319 MiB. That image workload was 512x512
at two steps, not a completed default twenty-step performance test. This
independent arbitration fix does not conclusively attribute the original
all-GPU-projector host-memory failure.
