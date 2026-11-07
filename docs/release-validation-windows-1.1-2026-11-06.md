# Arc Llama 1.1 native Windows validation — November 6, 2026

Branch `release/1.1`, based on `c254f527b791` (`1.1.0.dev0`) plus the fixes below. Native Windows 11 Pro
(26200), Python 3.12, Node 24, Intel Arc Pro B60 24 GiB (driver 32.0.101.8735), existing SYCL llama-server.
All servers used isolated configs, state folders and loopback ports. The real Arc Llama config, the user's
OpenCode and Hermes installations and their session databases were not touched. Nothing was published.

This is the first Windows run for the 1.1 candidate; the [October 28 record](release-validation-1.1-2026-10-28.md)
listed native Windows as a remaining gate.

## Results

| Check | Result |
|---|---|
| Ruff, mypy (95 files), JS syntax | Pass |
| Full Python suite | 2111 passed, 47 skipped (Linux/POSIX-only facilities and the opt-in live test), 0 failed |
| Browser suite | 117/117 after fix 4 |
| Wheel and sdist build; fresh core-only install | Pass; textual, fastembed and mcp absent; 7 CLI entry checks |
| Installed CPU smoke (`published_smoke.py`, Windows CPU llama.cpp b11381) | JSON, SSE, installed assets, `installed-smoke.mjs` UI check |
| Installed Work, 0.5B, `installed-work-smoke.mjs` | Edit in the job copy, reload, reviewed apply, undo, source preserved |
| Installed Work, 1.5B, `installed-work-retry-smoke.mjs` on the final wheel | normal: verified first attempt; retry: attempt 1 failed by injected fault, attempt 2 verified and chosen; keep-best: both failed, better attempt 1 kept |
| `gpus`, `init`, `doctor` on a fresh config | One B60 at `0000:06:00.0` after fix 1 |
| Native B60 SYCL, installed wheel | See below |
| `install-harness opencode` / `hermes` | OpenCode v2.2.5 in 6 s; Hermes v0.22.5 in about 2 min, `hermes acp --check` passed |
| `check_opencode_profiles.py`, `check_opencode_question.py` | Pass (isolated XDG folders) |
| `check_hermes.py` (real Hermes ACP, fake inference) | Pass after fixes 2 and 3: denied edit, approved edit, cancellation; no processes or private homes left |

### Native B60 SYCL

The GPU harness built its config from live detection and ran the wheel that contains fix 1. Qwen2.5-Coder 7B
Q4_K_M, context 4096, q8_0 KV, all layers offloaded:

- Three concurrent cold requests: one load, one child, three correct answers (122 s including the first SYCL load).
- Warm generation about 92 tok/s; SSE ends with `[DONE]`; malformed body 400; unknown model 404; Ollama `/api/chat`.
- 7B → 0.5B → 7B swaps with one resident child; admin stop-all left every model idle; no `llama-server` left.
- Ornith-1.5 35B-A3B Q4_K_M (context 8192) answered at about 39 tok/s. It is a reasoning model: with a
  48-token budget the whole reply was reasoning (`finish_reason: length`, empty content), as expected.

## Defects found and fixed

1. **Windows GPU detection listed PCIe switch ports and the audio function as GPUs, without PCI locations.**
   Reported in the 0.9.0rc2 Windows validation and still present. A B60 showed up as five Battlemage GPUs (E211
   plus E2F0, E2F1, E2F7, E2FF) with synthetic `unknown:` slots and a "multiple GPUs" device-order warning.
   `_enum_pci_locations` now keeps only display-class instances (or instances with no class, a card without a
   driver), and passes cfgmgr32 the full `PCI\...` instance ID. Regression: `TestEnumPciLocations`, a fake
   registry of this machine's devices.
2. **Hermes could not run in Work on Windows.** `hermes_environment` set `HOME` but not `USERPROFILE`, which is
   the only home Python reads on Windows, so `hermes acp` exited at startup ("Could not determine home
   directory"). The installer's `acp --check` used the full environment and passed, hiding it. On Windows the
   environment now sets `USERPROFILE` and private `APPDATA`/`LOCALAPPDATA` and passes non-secret system
   variables (`TEMP`, `COMSPEC`, `PATHEXT`, …). Regressions include a Windows-only test that starts a child
   with the environment and checks `Path.home()`.
3. **Every Hermes run on Windows ended in an error and left two processes running for ten minutes.**
   `hermes acp` starts a gateway daemon that Hermes deliberately detaches from the process tree and that exits
   only after ten idle minutes. It holds a lock file in the private home, so removing the home raised from
   `run_hermes`'s cleanup. Teardown on Windows now ends the gateway that the home's own `gateway.pid` /
   `gateway.lock` record names, and only when that record's `hermes_home` is this run's home. `hermes gateway
   stop` was tried first; it takes about 10.7 s to start and refuses while a cancelled session is live. Cleanup
   errors no longer fail a run. Linux is not affected by the cleanup error (open files can be deleted there);
   whether the gateway also lingers there was not checked.
4. **`frontend-integration-controller` could only pass on non-Windows hosts.** The dialog correctly shows the
   `curl.exe` command when the browser reports Windows; the test now expects the platform's spelling.
5. **Installed smoke scripts assumed Linux.** `published_smoke.py` now downloads the pinned Windows CPU build
   (checked against GitHub's SHA-256) and ends the whole server tree on failure, so a failed run no longer
   leaves `llama-server` holding the runtime. The Work smoke scripts use the OS temp folder instead of `/tmp`.

## Environment notes (not product defects)

- **Windows disk cleanup deletes uv-installed packages in `%TEMP%`.** Idle maintenance (`SilentCleanup`,
  Storage Sense) removed 1005 of 3056 files from a virtual environment and the extracted Hermes source under
  `%TEMP%` during this run. uv hard-links files from its cache, so they carry old timestamps. Keep validation
  environments outside `%TEMP%`. Arc Llama's default state folder is not under `%TEMP%`.
- The 1.5B CPU model sometimes writes tool calls as JSON text instead of calling the tool; Work then correctly
  reports failed verification (observed in one retry run, where it kept the better attempt 1).

## Not covered

Python 3.10 and 3.14 on Windows, Vulkan on Windows, the optimized Arc Llama Runtime, hosted Windows CI, other
GPUs, and the upgrade/rollback rehearsal on Windows.
