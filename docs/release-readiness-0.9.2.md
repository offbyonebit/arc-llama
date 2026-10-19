# Arc Llama 0.9.2 development readiness

Date: 2026-10-17. Canonical checkout: `/home/slowe/arc-llama`, `main`, based on
`bbed86b` plus the existing local work and this follow-up. Package and lockfile
version: **0.9.2.dev0**. This is unpublished development work; no commit, push,
service deployment, or public release is implied by this record.

## Scope completed locally

- Reversible chat compaction retains the full transcript, supports bounded
  summarization, and requires a persistence acknowledgment before applying it.
- Download destinations are selected per browser, validated on the server,
  retained on retries (including Windows drive roots), and checked for known-size
  free-space shortages both at submission and when a queued job starts.
- Generation keeps partial text and reasoning after Stop, stream errors, and
  premature EOF. Status survives disk reload, export, and browser history loading.
- Local streaming has independent configurable first-response/idle deadlines,
  an intentional unlimited mode, useful HTTP/SSE errors, and idempotent cleanup.
  Cleanup also covers a client disappearing before headers are sent. A literal
  `data: [DONE]` inside generated content no longer terminates the proxy early.
- Context budgeting uses a resident runtime's template/tokenizer and actual
  per-slot context, reserves the selected reply limit, and guards exact
  over-budget chat sends. The UI preserves the pending turn and offers compact,
  settings, and retry actions. Counting does not start models or contact external
  upstreams; unsupported/image/upstream counts are explicitly approximate.
- Published-wheel smoke automation is added for GitHub release publication,
  weekly checks, and manual runs. It exercises a core-only installed wheel with
  checksum-pinned CPU inference assets and checks installed HTML asset references.

- Work now provides project-specific instructions and permissions, persistent jobs
  independent of browser lifetime, exact command approvals, pause/resume, and
  restart recovery. Each job gets independent project copies with reviewed apply,
  conflict checks, and undo. Optional Linux commands use Bubblewrap and bounded
  systemd scopes with no host-home, network, or GPU access.

## Validation

- Complete non-inference Python suite: **1365 passed**, with the opt-in native
  smoke marker deliberately deselected for the separate real installed-wheel run.
- Real-browser regression suite: **49/49 passed**.
- Ruff (source, tests, scripts, registry tools), mypy (71 source files),
  `uv lock --check`, and `git diff --check`: **passed**.
- Wheel built from the source distribution, clean core-only install, archive
  content checks, and workflow YAML parsing: **passed**.
- Installed-wheel JSON/SSE inference, HTML asset requests, native budget/usage
  comparisons (including tool definitions when supported), parallel-slot capacity,
  and browser compaction/continue/undo: **passed**.
- Installed-wheel Work UI with a real local model: isolated file edit, browser
  reload, reviewed source apply, and undo: **passed**. Work recovery/API tests
  contribute 18 cases; two browser cases cover project jobs and manual tokens.
- Native Linux sandbox: copy-only writes, isolated home, absent host home/GPU
  devices, blocked host-network connections, and cancellation cleanup: **passed**.
- The new published-wheel workflow has not run on GitHub; Windows and additional
  physical GPU validation were not performed in this local follow-up.

The installed-wheel test uses Python 3.12, llama.cpp b11381 (CPU), and the pinned
Qwen2.5 0.5B Q4_K_M model. It runs under a 4 GiB memory/no-swap scope with a
200% CPU quota and isolated config, state, ports, and backend processes.
The runtime command explicitly uses `-ngl 0 --device none`; production model
configuration and the normal GPU resident are not used.

The native browser test verifies actual summarization, continuation, undo,
transcript retention, and native template/token counts against inference usage.
The configured 8192-token pool uses two parallel slots; the reported usable
context is verified as 4096 tokens. Package tests include all scripts/styles
referenced by the installed dashboard and chat pages. Optional TUI, semantic,
and MCP packages are absent in the installed environment.

## Backlog reconciliation

Issues were checked against GitHub on 2026-10-17. They remain open; locally
implemented changes should be linked and closed only after their reviewed code
ships and any remaining acceptance criteria are satisfied.

| Issue | Evidence and next action |
| --- | --- |
| [#81](https://github.com/offbyonebit/arc-llama/issues/81), interrupted answers | Stop existed previously; this follow-up adds persistent status, failure/EOF preservation, and reload coverage. Ready for release review. |
| [#82](https://github.com/offbyonebit/arc-llama/issues/82), stalled streams | This follow-up adds deadlines, HTTP/SSE errors, and cleanup/retry/disconnect coverage. Ready for release review. |
| [#84](https://github.com/offbyonebit/arc-llama/issues/84), context budget | This follow-up adds resident tokenizer/template counts, slot limits, reply limits, UI choices, and explicit fallbacks. Ready for release review for supported text runtimes. |
| [#79](https://github.com/offbyonebit/arc-llama/issues/79), published-wheel smoke | Workflow and local real-inference helper are implemented. A remote Actions run against a published artifact remains necessary after the workflow is committed and pushed. |
| [#83](https://github.com/offbyonebit/arc-llama/issues/83), split GGUF | Group download/registration and readiness checks already shipped in 0.9.1.2. Keep the immutable-revision and interruption/resume acceptance criteria under review rather than reimplementing grouping. |
| [#85](https://github.com/offbyonebit/arc-llama/issues/85), driver invalidation | Driver-sensitive fingerprints already shipped in 0.9.1.2. Narrow the remaining review to platform identity coverage, diagnostics, and failed-retune preservation. |
| [#80](https://github.com/offbyonebit/arc-llama/issues/80), context-depth generation | Standalone `benchmark --depths` exists. The ordinary generation measurement still starts from `Hello`; do not close the workload-ranking issue on the basis of the standalone command. |
| [#70](https://github.com/offbyonebit/arc-llama/issues/70), Linux Vulkan | The October 3 validation record already contains physical B60 Vulkan and SYCL results. Preserve its exact candidate/version coverage; this CPU run does not certify the new GPU candidate. |
| [#69](https://github.com/offbyonebit/arc-llama/issues/69), [#71](https://github.com/offbyonebit/arc-llama/issues/71), [#73](https://github.com/offbyonebit/arc-llama/issues/73) | Native Windows, community GPUs, and current-candidate native fresh-install evidence remain hardware/1.0 readiness work. |
| [#72](https://github.com/offbyonebit/arc-llama/issues/72) | The original all-GPU projector incident workload was not run. Keep the tested CPU-projector guidance and the separate contained-workload gate. |
| [#78](https://github.com/offbyonebit/arc-llama/issues/78), [#74](https://github.com/offbyonebit/arc-llama/issues/74) | Upgrade/rollback fixtures and the public stabilization period remain 1.0 gates. This local development build is not a completed public soak. |
| [#2](https://github.com/offbyonebit/arc-llama/issues/2) | Multi-GPU validation remains experimental/post-1.0. |

## Before publication

Review the complete working-tree diff, including the pre-existing compaction
and download work. Run the Windows/Python matrix in Actions from the chosen
release commit; local Linux checks do not stand in for that matrix. Update the
final version/date and archive hashes when creating a public release. Retain
all hardware coverage gaps above and restart the appropriate stabilization
window after release-blocking fixes. Keep the vision companion in its separate
repository.


## Follow-up verification: 2026-10-18

The October 17 totals above predate the Work result-checking and retry code.
This follow-up completes project controls for one to five attempts and a
one-to-720-minute admission budget, persisted settings, attempt history, and
the selected result. Disabling result checking selects a single attempt.
The UI and API prevent source actions and unstable diff review while a copy
is being replaced; verification is treated as active work in the UI.

Review found and fixed approval identifiers repeating between attempts, lost
pause requests during acceptance planning, missing attempt fields in older
saved jobs, and four type errors in the newer code. Attempt copy replacements
now persist their intent and stage in SQLite before filesystem operations.
Recovery repeats each stage safely and completes the selected copy before
exposing it. Shutdown retains ownership until filesystem threads finish.
Copy-removal failures leave the transition pending rather than silently
exposing the wrong result. The published-wheel asset checks include Work.

Validation in this follow-up:

- Complete non-inference Python suite: **1391 passed, 1 live-inference test
  deselected**. The final targeted Work checks pass all **44** cases.
- Full browser regression suite: **50/50 passed**; final Work-only checks:
  **3/3 passed**, covering saved attempt settings, verification toggles,
  active-state guards, history, and selected-result review.
- Ruff across source/tests/scripts/registry tools, mypy across **73 source
  files**, JavaScript syntax checks, lockfile check, and whitespace check pass.
- Source distribution and a wheel built from that distribution pass. A fresh
  core-only installed wheel passes CLI, package/static-asset, authenticated
  Work, retry/selection, reviewed apply, and undo checks with a **mocked model**.
- The earlier restricted-sandbox Work test run stalled; all Work cases passed
  outside that sandbox. No runtime synchronization was changed to mask it.

The new recovery tests interrupt every journaled copy-swap stage and check
that restart restores matching files and result metadata. Other regressions
cover cross-attempt stale approvals, legacy job resume, pauses during
acceptance planning, and blocking diff review during copy replacement.

At this stage, no real-model inference or physical GPU run had been repeated.
The later native inference results below supersede the real-model gap.
The October 17 native evidence remains historical; it does not certify the
new verification/retry path against a real model. Windows/Python CI on the
chosen commit, a remote published-wheel Actions run, additional hardware
coverage, upgrade/rollback fixtures, and the public stabilization window
remain outstanding as described above. No commit, push, deployment, release,
or issue closure was performed.


## Native Work inference follow-up: 2026-10-18

The core-only installed wheel was exercised against real, local CPU inference
with pinned llama.cpp b11381 and official Qwen2.5 Q4_K_M models. No API or model
responses were mocked. The source project fixtures, server configuration, state,
ports, and backend processes were isolated under /tmp. Each run used a systemd
scope with 4 GiB memory, no swap, a 200% CPU quota, and 256 tasks; inference used
zero GPU layers, --device none, two threads, and two 4096-token slots. The normal
GPU service was not changed.

Real inference exposed a parser bug: the planner returned a bare JSON criteria
array, but brace slicing discarded it. Work correctly reported unverified,
yet its checks were lost. The parser now accepts complete objects or arrays,
including fenced JSON and short preambles. Seven new regression cases cover
both formats and invalid input. The rebuilt wheel then retained and checked
the four real model-generated criteria for the two-file fixture.

| Model and case | Observed result |
| --- | --- |
| Qwen2.5 0.5B, normal two-file task | Passed: two actual file writes, four machine criteria and two safety checks, selected attempt 1, browser reload, reviewed apply, and undo. |
| Qwen2.5 0.5B, fault/retry stress | Failed safely: malformed tool arguments paused the job before execution. Two explicit admin resumes were tested; the model repeated malformed calls and the test remained failed. This does not certify unattended retries for this model. |
| Qwen2.5 1.5B, first attempt blocked | Passed: a real directory at B.txt in the first isolated copy caused write errors. Code checks failed, both repair passes ran, then a fresh second attempt passed all six checks and was selected. No protocol pauses or admin resumes were needed. |
| Qwen2.5 1.5B, later attempt worse | Passed: first attempt produced A.txt but could not write B.txt; the second could write neither. Work restored attempt 1, retained its failed-verification label, displayed the chosen result, and applied/undid only its reviewed changes. No protocol pauses or admin resumes were needed. |

The controlled faults were real filesystem failures in job copies only, not
synthetic inference or verification results. Successful native cases checked
the source was untouched before Apply and restored after Undo. JSON/SSE
inference and installed HTML asset requests passed as well. These tests establish
machine-checkable verification, repair feedback, retry copy replacement, earlier
best-result restoration, and browser review/apply/undo for the tested models;
they do not certify model-judged subjective criteria, physical GPUs, or Windows.

Final non-inference suite: **1398 passed, 1 live-inference test deselected**.
Ruff, mypy (73 source files), JavaScript syntax, and whitespace checks pass.
The optional native harness is browser-tests/installed-work-retry-smoke.mjs;
the helper now supports a checksum-pinned --model-size 1.5b and a bounded
--ui-check-timeout for slower CPU runs. Its default published-wheel model and
timeout remain 0.5B and 180 seconds.

Local reports, screenshots, logs, and the tested wheel are retained under
work/validation-2026-10-18/ (ignored and excluded from packaging). Tested wheel
SHA-256: `eebd2f0eea19c3a24d95f9d1a525429de6942a9413ef9092a6d9655c54495eaa`.
The 1.5B asset is pinned at Hugging Face revision
91cad51170dc346986eccefdc2dd33a9da36ead9, SHA-256
6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e.

Windows/GPU coverage, the remote Actions run from the chosen commit, and other
release gates above remain outstanding. No commit, push, deployment, or release
was performed.
