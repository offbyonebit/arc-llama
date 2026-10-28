# Arc Llama 1.1 candidate: installed-product readiness

> Historical record from October 25. See [October 28 validation](release-validation-1.1-2026-10-28.md) for the later candidate, hardware evidence and remaining gates.

Date: 2026-10-25. Validated by Claude on the canonical checkout. Nothing was committed, pushed or published, and the
running `arc-llama.service` was not touched.

## Candidate identity

| Item | Value |
|---|---|
| Source | `/home/slowe/arc-llama`, branch `release/0.9.2-work-mode`, HEAD `a27f6ec4efcb` plus uncommitted work (55 changed/new paths at validation time, all preserved) |
| Version | `1.1.0.dev0` (raised from `1.0.0`; see defect 1) |
| Wheel | `arc_llama-1.1.0.dev0-py3-none-any.whl`, SHA-256 `8f9bc164688e6604e563ae20ca249e0c7c6d8ccaea6b22a7395160e0b3c0869d` |
| sdist | `arc_llama-1.1.0.dev0.tar.gz`, SHA-256 `81e5f7663a0df529320af2b5a16e845d77451a8b5f8a9f0fa008fedb5399439c` |
| Build | `uv build --out-dir /mnt/storage/tmp/rc11/dist .` (uv 0.11.8, hatchling) |
| Host | Linux 7.0.0-28-generic x86_64, AMD Ryzen 5 5600; CPython 3.12.3 and 3.10.20; Node 18.19.1; Playwright Chromium 1228 |

These artifacts are local only. A candidate built after further edits needs new hashes.

## Results

| Check | Evidence class | Result |
|---|---|---|
| `scripts/check.sh` (ruff, mypy, JS syntax, pytest, browser suite) | Linux, CPU, source checkout | Pass: 1784 passed, 5 skipped; browser 94/94 |
| Wheel contents | Static inspection | Pass: every tracked/new `src/arc_llama` file present (134 entries); static assets including `harnesses.js/.css`, `chat/streaming.js`; data `recipes.json`, `arc-llama-runtime.allowed_signers`; no caches, tests or stray top-level files |
| sdist contents | Static inspection | Pass after fix (defect 2) |
| Core-only install, Python 3.12 and 3.10 | Linux, CPU | Pass: `textual`, `fastembed`, `mcp` and document libraries absent. `numpy` is present as a dependency of the core `gguf` package. All checklist CLI commands exit 0 (`--version`, `--help`, `run --help`, `serve --help`, `recipes --help`, plus `install-harness --help` and `install-runtime --help`) |
| Every module imports, core-only | Linux, CPU | Pass on 3.12 and 3.10. `arc_llama.tui` and `arc_llama.agent_tui` exit with the documented "requires textual" message. A missing `mcp` or `fastembed` gives an install hint. |
| `published_smoke.py`, 0.5B model, with `installed-smoke.mjs` | Linux, CPU only (`n_gpu_layers 0`, `--device none`) | Pass after fix (defect 3): JSON and SSE inference, installed assets, exact native budget (32 tokens), per-slot context 4096, compaction, continue after compaction, undo keeps the full transcript |
| `published_smoke.py` with `installed-work-smoke.mjs` | Linux, CPU | Pass after fix (defect 4): file edit in the job copy, reload, reviewed apply, undo, source preserved |
| `published_smoke.py`, 1.5B model, `installed-work-retry-smoke.mjs`, `retry` | Linux, CPU | Pass: attempt 1 failed verification because of the injected fault; attempt 2 verified and was chosen |
| Same, `keep-best` | Linux, CPU | Pass: both attempts failed; the better attempt 1 was kept |
| MCP extra: `[mcp]` with `mcp==1.30.0` | Linux, CPU, synthetic FastMCP stdio server | Pass after fix (defect 5): 3 tools registered; text result returned once; an `isError` result stays an error; an image passed as an image; a synthetic agent conversation called the MCP tool and finished; tools unregistered on stop |

The 1.5B retry and keep-best cases ran on the wheel built before defect fixes 5–6, which don't touch the
Workbench runner. Every other installed row above was rerun on the final wheel.

### Commands

The smoke runs were made from `/tmp`, outside the checkout, each in a bounded scope:

```bash
systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% -p TasksMax=256 \
  env -u ARC_LLAMA_ADMIN_TOKEN timeout 600s /mnt/storage/tmp/rc11/core3.12/bin/python -I \
  /home/slowe/arc-llama/scripts/published_smoke.py --work-dir /mnt/storage/tmp/rc11/smoke \
  --expected-version 1.1.0.dev0 --ui-check /home/slowe/arc-llama/browser-tests/installed-smoke.mjs --ui-check-timeout 420
```

- **Work smoke:** the same command with `ARC_WORK_SMOKE_ROOT=/tmp/arc-work-rc11-final` (containing `README.txt`)
  and `--ui-check .../installed-work-smoke.mjs`.
- **1.5B cases:** `--model-size 1.5b` and `ARC_WORK_NATIVE_CASE=retry|keep-best` with an empty
  `/tmp/arc-work-*` root. The harness asserts that job copies live under `/tmp/arc-work-*`, so the asset/state
  directory was a temporary `/tmp/arc-work-rc11-assets`, deleted afterwards.
- **Assets:** llama.cpp b11381 CPU build and Qwen2.5 0.5B/1.5B Q4_K_M, each verified by the helper's pinned SHA-256
  before use.

## Upgrade and rollback rehearsal (1.0.0 → 1.1 → 1.0.0 → 1.1)

**Inputs:**
- **Wheels:**
  - `arc_llama-1.0.0-py3-none-any.whl`, SHA-256 `f2b11d418b09a9f8bf65e2aaf51f0528cbf78b478943e0b4f8458ab8751f42d2`
    (PyPI; matches the release notes);
  - the candidate `arc_llama-1.1.0.dev0-py3-none-any.whl`, SHA-256
    `8f9bc164688e6604e563ae20ca249e0c7c6d8ccaea6b22a7395160e0b3c0869d` (a rebuild was byte-identical to the
    artifact above).
- **Environments:** one virtual environment per wheel. Isolated `HOME`, `XDG_CONFIG/STATE/DATA/CACHE_HOME` and
  `HERMES_HOME` under `/mnt/storage/tmp/upg`, and a free loopback port per start. The production service, the real
  config and personal conversations were not touched, and OpenCode was not used.
- **Synthetic state, created through supported paths:**
  - a hand-edited config with custom values and unknown keys at top level, in `[server]` and in a model entry;
  - the model registered with 1.0.0's `arc-llama add` and `model-runtime`, a 2 KB synthetic GGUF;
  - its per-model `llama_server`: a scripted OpenAI-compatible stand-in that logs every request;
  - two chats through `/v1/chats`, one ending in an `interrupted` assistant reply;
  - two Workbench projects in temporary folders;
  - one completed job, and one job stopped mid model-call by shutting down the server.
- **Inventory:** an independent inventory read the files directly (TOML, chat JSON, `work.sqlite3` opened
  read-only). Driver and logs are under `/mnt/storage/tmp/upg` (`rehearse.py`, `rehearsal.log`, `report.json`).

**Results.** All 33 rehearsal checks passed (Linux, CPU only, stand-in model server). At each of the three
transitions:
- **Startup and config:** the server started and the config was preserved.
- **Chats:** visible, including the interrupted reply.
- **Work:** workspaces and jobs visible.
- **No reruns:** the completed job stayed completed with 0 new model calls.
- **Cut-off job:** paused for manual resume with 0 replayed calls (docs/work.md policy).
- **Persistence:** a chat edit, a workspace edit and a `model-runtime --env` change persisted across a restart.
- **Unknown keys:** survived that version's saves.
- **Inventory:** matched.

Also passing:
- **Rollback of new settings:** 1.0.0 keeps 1.1's `agent.opencode_bin`/`hermes_bin` when it saves; it just doesn't
  use them.
- **Config checker against the real 1.0.0 release:** `scripts/check_config_versions.py --versions 1.0.0` upgraded
  57/57 values, and rollback reset 0 settings.

**Configuration vs chat/job compatibility:**
- **Config:** compatible both ways with no loss.
- **Chats and Work jobs:** stored in formats identical in 1.0.0 and 1.1 (no diff in `chat_store.py` or `workbench/`
  since v1.0.0), so no migration runs in either direction.

**Backup and restore after a failed upgrade:** a failed upgrade was simulated by letting the candidate edit a chat
and then damaging `work/work.sqlite3`.
- **Documented config-only backup:** did not recover. Work was unreadable ("database disk image is malformed").
- **Full backup** (`cp -a` of the config folder and the state directory, now in docs/upgrading.md): restored the
  original inventory exactly, and 1.0.0 then ran on it with no new model calls.

**Rehearsal limitations:**
- Linux and CPU only; no Windows, no GPU, no real model.
- Work conversations made with OpenCode or Hermes, and installed runtimes and harnesses, were not part of the
  synthetic state.
- Only one 1.0.0 → 1.1 path was rehearsed; no older releases were included beyond the config round-trip fixtures.

## Defects found and fixed

1. **The candidate's version equalled the published 1.0.0.** A wheel built from the tree would be indistinguishable
   from the release, and the checklist requires a newer version. Set to `1.1.0.dev0` in `pyproject.toml` and
   `uv.lock`. Choose the final 1.1 version at release time.
2. **The sdist leaked local-only files.** `CLAUDE.md` and `LOCAL_REPO_BOUNDARIES.md` are hidden through
   `.git/info/exclude`, which hatchling does not read. The `docs/release-validation-windows-2026-10-22/` screenshot
   folder also escaped the `docs/release-validation-*.md` pattern. Added excludes. Regression test:
   `tests/test_packaging.py`.
3. **`installed-smoke.mjs` was stale.** It clicked `#compact-button`, which now lives in the closed Context popover,
   so it failed identically against the published 1.0.0 wheel. The script now opens the Context popover the way the
   browser suite's `openContext` helper does. With the fix it passes on both 1.0.0 and the candidate.
4. **`installed-work-smoke.mjs` and `installed-work-retry-smoke.mjs` were stale.** Both opened `/work`; the
   project-job UI moved to `/workbench`. Both scripts were updated and now pass.
5. **MCP tool results reached the model twice.** With the real SDK (1.30.0), FastMCP returns a plain value as text
   and also as `structuredContent = {"result": value}`, and the client appended the JSON copy. The client now
   recognises the `result` wrapper as already represented. Regression:
   `test_structured_content_already_in_text_is_not_sent_twice`.
6. **Cut-off tool calls ran with `{}`** (reproduced 2026-10-24 during the Grok R&D review; not yet fixed in the
   candidate). With `finish_reason: "length"` or arguments that aren't a JSON object, both the agent loop and the
   interactive loop ran the tool with empty arguments. They now return a "Not executed … send the call again" error
   to the model. Regression: `test_cut_off_or_malformed_tool_calls_are_not_executed` and
   `test_complete_tool_call_still_runs`.

7. **The documented backup didn't protect chats or Work jobs.** `docs/upgrading.md` only described copying the
   config file; the rehearsal showed that leaves chats and jobs unrecoverable after a failed upgrade. The doc now
   gives a stop-and-copy procedure for the config folder and state directory, verified by the rehearsal.
8. **No automated coverage of the real 1.0.0 release.** Config round-trip fixtures stopped at 0.9.1.2. Added
   fixtures written by the real 1.0.0 wheel and by the candidate; `tests/test_config_versions.py` now covers 1.0.0
   (upgrade keeps every value, and rollback loses nothing). The script's default versions include 1.0.0.

Already present in the candidate (Codex): model downloads pinned to one commit with LFS/blob hash verification, and
`mcp>=1.30,<2`.

## Skips

| Skip | Reason |
|---|---|
| 4 × `tests/test_documents.py` | `docx`, `openpyxl` and `pptx` belong to the optional `documents` extra and aren't installed in the dev venv |
| 1 × `tests/test_smoke.py` | Opt-in live inference test; needs `ARC_LLAMA_SMOKE_MODEL`. Installed CPU inference was covered by `published_smoke.py` instead. |
| `pytest arc-llama-vision/tests` | Not run: separate companion package, out of scope here |

## Coverage gaps (not run)

- **Intel GPU:** no B60, Alchemist or other GPU inference was run for this candidate. Every inference row above is
  CPU only.
- **Windows:** nothing was run on Windows. Also untested there: in-app harness install (checklist in
  `/home/slowe/codex-windows-harness-install-checklist.md`), Hermes on Windows (still marked experimental), and the
  command sandbox (Landlock is Linux-only).
- **SYCL runtime and Arc Llama Runtime:** not exercised. The pinned smoke runtime is the upstream CPU build.
- **Real harness installs from the candidate wheel:** not repeated here. OpenCode v1.19.4 and Hermes v0.22.5 were
  installed for real and passed `scripts/check_hermes.py` and `check_opencode_profiles.py` from the source tree on
  2026-10-24.
- **MCP SDK range:** `>=1.30,<2` allows only 1.30.0, the last 1.x release. MCP SDK 2.x (2.0.1–2.3.0 on PyPI) is
  unsupported and will conflict for anyone who already has it installed.
- **sdist size:** 14.4 MB, almost all of it `docs/showcase` PNGs. 1.0.0's sdist (14.6 MB) was the same size.
  Consider excluding the showcase images from the sdist.
- **Hardware matrix:** the checklist items needing hardware (doctor/setup on Arc GPUs, Vulkan/SYCL, interrupted
  downloads, unsafe archives on device) and the offline-network UI pass were not run.
- **CI:** Linux and Windows CI and the bare-wheel CI job have not run on this uncommitted candidate.
