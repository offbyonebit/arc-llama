# Testing and diagnosing stalls

Run the core checks from the repository environment:

```bash
python -m pytest tests/ -q -ra --tb=short -p no:cacheprovider
ruff check src tests registry-repo/scripts
mypy --check-untyped-defs src
```

Pytest emits thread stacks after 30 seconds in a test. CI jobs have a hard
15-minute deadline, so a stalled test cannot occupy a worker indefinitely.
For local Linux investigations, use an outer deadline as well:

```bash
timeout 120s python -m pytest tests/test_config_atomic_save.py -vv -p no:cacheprovider
```

## Dashboard and package checks

See [Architecture and change boundaries](architecture.md) for component owners
and the invariants to preserve when extracting code.

Run the browser regression suite after dashboard or chat behavior changes:

```bash
cd browser-tests
npm test
```

The polling regression cases count status/job requests, retain model-card node
identity across unchanged responses, verify updated settings and keyboard focus,
and exercise failed completion-refresh retries. A held status response verifies
that simultaneous forced refreshes share one fresh follow-up read. Synthetic
visibility events check the hidden-tab timer cadence and one refresh on return;
they do not claim to measure a browser's own background throttling.

`browser-tests/run.mjs` owns only test selection, browser/context lifetime, and
reporting. `harness.mjs` owns shared fixtures, routes, and static serving;
`chat-cases.mjs`, `dashboard-cases.mjs`, and `library-cases.mjs` own their cases.
Importing the harness or cases starts no server, browser, or test run.
`node run.mjs --only FILTER` selects named cases; a filter matching nothing fails.

The harness serves the real frontend files against synthetic API responses.
It does not start models or change user configuration. Static UI tests also
check JavaScript syntax recursively, including dashboard and chat components.

For a structural pass that must not run inference, explicitly exclude the live
smoke test:

```bash
env -u ARC_LLAMA_SMOKE_MODEL python -m pytest tests/ -m "not live_inference" -q -ra --tb=short -p no:cacheprovider
```

When adding frontend assets, build a wheel and check that scripts referenced by
`index.html` and `chat.html` are included under `arc_llama/static/`. A browser
pass from a source checkout does not verify installed-package assets.

## Runtime compatibility guidance

The library's explicit compatibility action reads at most 256 KiB of the exact
selected GGUF at an immutable Hugging Face commit. A registered-model check reads
its local header. Missing/gated metadata, unsupported header layout, and network
errors remain unknown; repository names and memory fit are not support evidence.

Each explicit assessment has a 25-second request deadline, including admission
wait, network reads, and the loader probe. At most two assessments run per app.
Disconnects and server shutdown cancel pending checks; network streams close and
probe processes are killed and reaped before their temporary fixtures are removed.
Local file/stat reads run in a thread so slow storage does not block the event loop;
OS filesystem operations themselves cannot be interrupted by asyncio cancellation.
The browser has a 30-second fallback deadline and aborts checks invalidated by a
runtime change or a removed control. Stale responses cannot replace newer results.

Hugging Face metadata is capped at 2 MiB and the selected GGUF header at 256 KiB.
Hub access settings are honored; bearer credentials are stripped on cross-host
redirects. Diagnostics retain bounded loader output and record failure phase,
exception class, and HTTP status without logging exception messages or tokens.
Architecture recognition is probed with a temporary zero-tensor GGUF, CPU-only
arguments, and a ten-second subprocess deadline. An architecture-specific loader
rejection establishes a mismatch. Reaching an architecture-specific missing-key
error establishes recognition only. This does not validate tensor encodings,
backend kernels, projectors, draft models, or successful inference. No probe runs
on search, status polling, or page load. Definitive probe results are cached by
runtime identity; transient unknown results remain retryable. Runtime identity
tracks executable and adjacent shared-library file metadata, not a build support
manifest or a cryptographic hash of their contents. Status polling clears shown
assessments when this identity changes, and local assessments also reset when
registered file identity changes.

Run `tests/test_model_compatibility.py` and `tests/test_compatibility_io.py` for
deadlines, admission, cancellation cleanup, bounded IO, credentials, evidence rules,
identity changes, immutable revisions, and authentication. The browser case
`runtime-compatibility-guidance` checks exact-file requests, separation from
memory fit, downloaded-model assessment, and runtime-change invalidation.
`compatibility-abandoned-checks` verifies browser timeouts, aborted requests, and
stale responses. The same CI jobs run on pushes to
`feature/model-journey-and-organization` and pull requests targeting `main`.

## Restricted execution environments

A TestClient stall is not necessarily an application startup deadlock. On
2026-09-18, the persistence rollback test stalled inside the restricted agent
sandbox, as did entering `TestClient(FastAPI())` with no Arc Llama application.
The same rollback test passed outside that sandbox in 0.41 seconds. The full
baseline suite at commit `81c33f4` passed there with 936 passed and 13 skipped
in 18.94 seconds. The stack dump showed the main thread waiting in AnyIO's
blocking portal and its event-loop thread waiting in the selector.

When this signature recurs, first compare a minimal empty FastAPI TestClient
inside and outside the restricted environment. Use the approved environment
for the real checks; do not bypass lifespan, suppress tests, or change runtime
synchronization to mask an execution-environment problem. These observations
do not identify the exact sandbox restriction responsible.

The baseline skips required local GGUF fixtures or an explicitly configured
GPU inference smoke model. Unit-test success is not hardware validation.

## Next-release recovery and budget checks

`tests/test_streaming_inflight.py` exercises stalled headers, first streamed
bytes, idle gaps, slow prefills, unlimited mode, and cleanup/retry. It also
retains upstream failure and cancellation coverage. `tests/test_context_budget.py`
checks native template/tokenizer inputs, per-slot capacity, reply allowance,
explicit fallbacks, validation, deadlines, and lease cleanup. Browser cases in
`release-cases.mjs` cover saved interruptions/stops, exact budget guards,
model-change races, and download destination/retry persistence.

The published-wheel workflow runs on published releases, weekly, or manually.
Release runs install the GitHub release wheel; scheduled/manual runs install the
latest PyPI wheel. Its Python 3.12+ helper `scripts/published_smoke.py` downloads a
checksum-pinned llama.cpp b11381 CPU runtime and Qwen2.5 0.5B Q4_K_M model, creates
isolated config/state, and verifies real JSON/SSE inference plus installed HTML
script/style references. It never allocates GPU layers. This validates package
integration, not Intel GPU performance or compatibility.

For a local candidate, install the wheel without extras in a fresh environment,
then run the helper from outside the source checkout. On a Linux host, use an
outer time limit and a memory-limited scope. The optional
`--ui-check /absolute/path/to/browser-tests/installed-smoke.mjs` uses Playwright
and Chrome/Chromium to verify exact counts against native inference usage,
parallel-slot capacity, actual summarization, continuation, and undo with the
complete original transcript retained. It uses the installed package's UI, with
real HTTP endpoints and no inference mocks. Browser dependencies belong in the
separate browser-tests environment, not the installed core wheel.


## Work projects and jobs

`tests/test_work.py` and `tests/test_work_api.py` exercise independent snapshots,
read-only policy, path containment, exact approvals, persisted receipts, restart
recovery, single-worker ownership, pause/resume, shutdown during copying,
review revisions, source conflicts, apply rollback, undo, and retained audit
history. `tests/test_work_verify.py` covers acceptance checks, repair feedback,
evidence limits, and interrupted verification. `tests/test_work_attempts.py`
covers fresh attempts, budget admission, best-result selection, approval IDs
across attempts, legacy jobs, planning pauses, and restart at every copy-swap
stage. The browser `work-` cases cover project/job creation, reload/reconnect,
review/apply/undo, attempt settings, verification toggles, chosen results, active
verification guards, and model loading after manual admin-token entry.

For native candidate validation, pass
`--ui-check /absolute/path/to/browser-tests/installed-work-smoke.mjs` to the
published-wheel helper and set `ARC_WORK_SMOKE_ROOT` to a fresh
`/tmp/arc-work-*` directory containing `README.txt` with `Original project`.
The browser asks the installed local model to create a file in its job copy,
checks the source is untouched, reloads, reviews and applies the diff, and undoes
it. This uses actual HTTP/inference and the installed static assets.

Optional command isolation also needs a native Linux check outside restricted
CI containers. The 2026-10-17 host check confirmed copy-only writes, an isolated
HOME, absent host home/GPU devices, blocked host-network connections, and
termination of child processes on cancellation. This does not establish native
Windows or macOS command isolation; those platforms support file-only Work.


### Real Work verification and retry tests

Use a core-only installed candidate wheel, not an editable checkout. The helper's
`--model-size 1.5b` selects the official checksum-pinned Qwen2.5 1.5B Q4_K_M model;
the default remains 0.5B for the ordinary published-wheel smoke test. Install the
separate browser-tests dependencies and Chromium as described above. Run outside
the restricted agent sandbox, with a fresh empty `/tmp/arc-work-*` source fixture:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=256 \
  env ARC_WORK_SMOKE_ROOT=/tmp/arc-work-native-retry ARC_WORK_NATIVE_CASE=retry \
  timeout 510s /path/to/core-only/bin/python scripts/published_smoke.py \
  --work-dir /tmp/arc-work-native-assets --model-size 1.5b \
  --expected-version 0.9.2.dev0 --ui-check-timeout 450 \
  --ui-check browser-tests/installed-work-retry-smoke.mjs
```

Repeat with a different fresh source fixture and `ARC_WORK_NATIVE_CASE=keep-best`
to test restoring the earlier result when the second attempt is worse. `normal`
runs an unblocked single-attempt case. Project folders must be empty or absent;
the harness refuses to overwrite an existing fixture. The asset/state directory
may be reused after the prior run stops. The 1.5B model requires about 1.1 GB
of disk space; the helper verifies its pinned SHA-256 before inference.

The retry cases insert directories at output file paths in independent job
copies, so real write operations fail without touching source files. They assert
failed checks and two repair passes, fresh attempts, selected file contents,
the displayed verdict/history after reload, reviewed apply, and undo. No API or
model responses are mocked. At most two explicit admin resumes are attempted
for malformed model calls; these are recorded, never presented as an unattended
pass. A model that still cannot complete the case fails the harness. The native
record for October 18 includes the 0.5B failure and successful 1.5B runs without
admin resumes. These are opt-in tests, not additions to the default unit suite.


## Synthetic protocol and lifecycle regressions

The browser cases `chat-sse-framing` and `chat-sse-errors-keep-partial` send
synthetic streams through the actual Chat UI. They cover byte-fragmented UTF-8,
CR/LF and split CRLF, BOM, comments, multiline data, literal completion-marker
text, usage-only events, malformed event shapes, bounded line/event size,
premature EOF and response-body cancellation. A malformed response preserves
partial text and reasoning, records an interruption, and permits a new request
without automatically replaying the failed request. The parser caps individual
lines and accumulated event data at 1 Mi UTF-16 code units. Complete events are
required; a model finish reason alone is not transport completion.

`tests/test_runtime_faults.py` uses a synthetic archive and httpx MockTransport
to inject interrupted downloads, size/hash mismatches, extraction failures and
cancellation, backend mismatches, marker-write failures and publication failures.
Each case verifies the previous install/config survive, temporary files are
cleaned, and a clean retry succeeds. These are exception/cancellation tests,
not a power-loss durability claim or certification of concurrent installers.

`tests/test_runtime_interruption.py` runs a synthetic installer in a child
interpreter that calls `os._exit` (no exception handlers or `finally` blocks
run) at each publication boundary: after the old install is renamed to a backup,
after the new install is published, after the backup is renamed for deletion,
and partway through deleting it. Before this was fixed, a kill after the first
rename left the configured runtime path missing; a later offline retry did not
bring it back, and even a successful retry leaked the old install as a hidden
backup. The installer now repairs these states before any network access: an
install that is missing while its backup exists is restored from the backup,
and backups of installs that are present are deleted. A backup is renamed to a
trash name before deletion, so a half-deleted tree never looks like a
restorable backup. Recovery never promotes a staging directory, a backup with
no `llama-server`, or a symlink, and a failed restore or failed cleanup leaves
the backup in place. The tests check offline recovery, a later successful
retry, and both kinds of failed recovery.

Two overlapping installers are tested at two fixed interleavings: one paused
with its install moved aside, and one paused before deleting its backup. In
both, a complete install and a matching configuration remain, and the paused
installer either finishes or fails without replacing it. There is no installer
lock, and other interleavings are untested. The killed run's partial download
and staging directory are not reclaimed (they can be large); they are never
used. A backup half-deleted by Arc Llama 1.0.0 or earlier, before the trash
rename existed, could still be restored if its `llama-server` file survived.
These are Linux process-exit tests. They make no power-loss claim (nothing is
fsynced), and the Windows behavior of directory renames during overlap is not
verified natively.

`tests/test_launcher_stress.py` starts bounded native Linux fixture processes
through the real launcher in an isolated helper interpreter. Repeated graceful,
forced, partial-startup, crash, concurrent-stop and cancelled-stop cases check
process reaping, port rebinding, log retention and resident-lock release. Nested
children are synthetic and reaped; no installed harness or model is started.
The helper has outer test deadlines and child expiry alarms. Windows retains
its separate mocked lifecycle coverage; this suite does not certify Windows
process-tree behavior.

`tests/test_config_properties.py` generates deterministic nested TOML values
and checks migration idempotence and preservation of unknown fields through
repeated edits across supported schema versions. It adds no test dependency
or alternative configuration store. All fixtures remain separate from user
configuration and native harness conversation history.


## Research-derived search, MCP and download regressions

`tests/test_research_gaps.py` covers out-of-root file links, scan/read link swaps,
equal-stat edits that become newly relevant, invalid embeddings preserving the
previous index, structured MCP content, catalog refresh and shutdown, immutable
shard revisions, missing integrity metadata and checksum failure preserving an
existing model. Fixtures are synthetic and require no model downloads.

Search now hashes every eligible file before ranking; freshness checking grows
with repository content size. On POSIX, source opens pin parent directories and
refuse symlink traversal. Native Windows race behavior has not been certified.

Model downloads require immutable revision and file integrity metadata. They
verify the complete requested set in private staging before exposing new files.
Publication of an existing multi-file set remains per-file, not crash-atomic;
concurrent readers and interrupted replacement need a versioned-bundle protocol
for stronger guarantees. Failed transfers can be retried; partial-byte resume is
not promised. Hash verification adds one sequential read of downloaded content.

The MCP extra targets `mcp>=1.30,<2`. Notification handling starts a bounded
refresh task outside the SDK receive loop, removes stale registrations and
invalidates previously obtained handlers. Failed or continuously changing
catalogs remain unavailable instead of exposing old definitions. Shutdown
cancels refresh work before unwinding SDK session contexts.
