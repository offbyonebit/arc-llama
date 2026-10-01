# RC2 improvement and validation report

Date: 2026-10-01. Checkout: `/home/slowe/arc-llama`.
Branch: `maintenance/rc1-local-follow-up`.
Base HEAD: `5162625108225662345ef707c4e7414700f119a2`.
Validation was completed before committing or pushing this improvement pass.
No release was published.

## What changed

### Release preparation

- Prepared `0.9.0rc2` package metadata and changelog; corrected the root package
  version in `uv.lock`, which previously said `0.9.0.dev0`.
- Carried over the upstream Windows UTF-8 corrections in the two static UI
  test modules. The local maintenance branch predates these main-branch fixes.
- Built a wheel and a source distribution, including building the wheel from
  the source distribution, using `uv build`.
- Installed the actual published RC1 GitHub wheel into a temporary environment,
  created a config and a folder chat, upgraded to the candidate wheel, and
  verified existing settings/history and subsequent saves.

### Failure and recovery

- Chat writes use a unique sibling temporary file, flush/fsync, then replacement
  after closing the handle. Failed flushes and replacements leave the old chat
  intact and clean up temporary files. This avoids Windows open-handle rename
  behavior. This is an atomic file replacement guarantee, not a guarantee
  against every filesystem or power-loss failure.
- Invalid record shapes and timestamps are skipped without breaking chat lists.
  Legacy records with a null folder still load as root-folder chats.
- Request validation rejects malformed bodies/messages and invalid folder
  queries with HTTP 400. Save failures give actionable HTTP 503 guidance.
- Overwrite imports that move a chat remove its old file, avoiding duplicates.
- Cancelled model starters settle their shared future and stop the process.
  Failed readiness checks also stop partially started processes. Tests cover
  retry after cancellation and an interrupted readiness check with 40 waiters.
- Repository scans tolerate files disappearing while being inspected.

### Performance

Measured existing operations without adding user-facing features:

| Operation | Before median | After median | Fixture |
| --- | ---: | ---: | --- |
| Repeated folder polling | 19.03 ms | 0.338 ms | 50 chats, 50 messages/chat, 4 KiB/message |
| Repository map scan | 358.61 ms | 14.20 ms | 100 visible files, 10,000 ignored files |
| Repeated GGUF metadata reads | 6.346 s | 0.039 ms | Real LFM2.5-2.6B Q4_K_M model, warmed cache |

Folder polling and repository measurements use seven timed repetitions after
warming. Metadata measurements use five repetitions after warming. These
figures describe the fixtures and machine used; they do not promise the same
ratios on every installation or improved GPU token generation speed.

- Chat summary caching checks file size and nanosecond modification time and
  invalidates on saves, moves, deletions, and external edits.
- Repository walking prunes ignored directories before descent, including
  `.venv*` directories, and preserves sorted output.
- GGUF metadata caching is bounded to 128 records and keyed by resolved path,
  size, and nanosecond modification time. Callers receive independent copies.
  Initial parsing still incurs its original cost; the repeated work disappears.

### Cleanup

- Shared repository enumeration between the map and semantic index.
- Removed repeated ignore checks after filtered enumeration and reused each
  file stat for semantic staleness checks.
- Shared chat body/message validation and persistence error handling.
- Shared identifier normalization rather than repeated regular expressions.

### First-run usability

- Obvious missing local GGUF paths fail before runtime installation, explaining
  quoting and the Hugging Face input format. HF filename specs remain accepted.
- Windows GPU guidance now points to graphics drivers, Device Manager, and
  `arc-llama doctor`, rather than incorrectly claiming detection is unsupported.
- Installed candidate initialization detected the local Arc Pro B60 (24 GiB)
  and existing SYCL runtime with a temporary config.
- Setup accepted a model path containing spaces and Unicode and printed a
  launch plan. Fresh registration with the optimized candidate took 6.60 seconds;
  repeating setup kept one registered model and the same port. Setup-only did
  not start a new model process.

## Validation results

- Full Python 3.12 suite: **1,049 passed, 13 skipped**, in 22.05 seconds.
  The baseline was 1,026 passed with the same 13 skips: 23 regressions added.
- Ruff: passed for `src`, `tests`, and `registry-repo/scripts`.
- Mypy with `--check-untyped-defs`: passed, 47 source files.
- JavaScript syntax checks: passed for `app.js` and `chat.js`.
- Real Chromium browser regression suite: **8/8 passed**. Covered chat selection,
  sending, structured load failures, preserved edits during refresh, keyboard
  navigation, settings during polling, retry behavior, and dashboard measurements.
- Fresh Python 3.11 core-only wheel installation: passed CLI version/help checks,
  server version, bundled recipes, markdown safety, and vendor/UI asset presence.
  `textual`, `fastembed`, and `mcp` were absent.
- Published RC1 wheel to RC2 upgrade: existing config and folder chat preserved;
  appended history remained readable. Loading an unprotected config may generate
  an admin token through the existing config migration behavior.
- Real installed candidate server with temporary state and an ephemeral port:
  health, bundled assets, real nonstreaming B60 inference, SSE completion with
  `[DONE]`, early stream close followed by healthy service, and chat CRUD/invalid
  update protection all passed.
- Existing GPU backend was used as an upstream. Its model ownership and running
  service were preserved. The temporary candidate process was terminated.
- Offline `uv lock --check`: passed. Archive inspection confirmed package version
  and exclusion of local environments, node dependencies, and companion code.
- `git diff --check`: passed. HEAD remained unchanged.

## Remaining release gates and scope

The local candidate passed the checks available here. Publishing a release
still requires testing this exact candidate on the GitHub Linux/Windows matrix
and merging the approved branch. Main-branch green CI covers different source;
this report does not certify a GitHub CI run for the candidate.

A native Windows session was not available, so Windows GPU detection, runtime
installation and native inference are not newly certified by this pass. Tests
exercise Windows guidance and preserve upstream UTF-8 fixes, and file handles
are closed before replacement, but that does not replace Windows execution.

The real GPU check validates the candidate server/proxy against the existing
B60 native backend. New native cold start, GPU eviction under real memory
pressure, fresh runtime download, and other Intel cards/backends were not
exercised. Their automated/mock coverage is included in the full suite.

The 13 skips are explicit: nine missing GGUF metadata fixtures, three missing
MTP launcher fixtures, and the opt-in native inference smoke test. Independent
installed-server inference above ran successfully. One Starlette TestClient
`httpx` deprecation warning remains; it is not a test failure.

Temporary wheels, environments, logs, benchmark samples, and setup evidence
were written under `/tmp/arc-hardening`. They are local validation artifacts
and are not committed or published.
