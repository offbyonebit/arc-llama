# Native Windows release validation — 2026-10-22

## Scope and result

Repository: https://github.com/offbyonebit/arc-llama
Branch: `release/0.9.2-work-mode`
Validated commit: `3af34e51db1634e307f454026cf7819be7caf62a` (`0.9.2.dev0`).
Native Windows execution on 2026-10-22, America/Chicago; started approximately 13:46 CDT.

This is a recorded validation run for [#69](https://github.com/offbyonebit/arc-llama/issues/69) and part of [#73](https://github.com/offbyonebit/arc-llama/issues/73). **This is not an unconditional release pass.** The original runtime installation failed TLS verification. A documented Windows-CA retry enabled real JSON, streaming, browser chat, cold model starts, and single-resident model switching. Intermediate API/UI requests also failed because the runtime path was reported missing; the cause was not established.

The two small GGUFs exercise single-resident eviction, not eviction under real VRAM exhaustion. Issue #73's real-memory-pressure acceptance criterion remains unvalidated. SYCL was not tested. Earlier rc1 native Windows details could not be recovered: the checked-in rc1 report records Linux, and rc2 explicitly says native Windows was unavailable. No historical Windows pass is inferred or fabricated.

## Host and runtime

| Item | Observed value |
| --- | --- |
| Windows | Microsoft Windows 11 Pro, 64-bit, version 25H2, build **26200.9457** |
| OS evidence | Win32_OperatingSystem: `10.0.26200`; registry UBR: `9457`. Registry ProductName still says Windows 10 Pro; CIM Caption says Windows 11 Pro. |
| Intel GPU | Intel Arc Pro B60, reported **24,382 MiB** by arc-llama and Vulkan |
| Intel driver | **32.0.101.8735** |
| Other GPU | NVIDIA GeForce RTX 4060 Ti; driver **32.0.15.9186** |
| Other display adapter | Parsec Virtual Display Adapter, driver 0.45.0.0 |
| Backend tested | **Vulkan**, automatically selected by fresh `run` |
| Device enumeration | Vulkan0 = NVIDIA RTX 4060 Ti; **Vulkan1 = Intel Arc Pro B60** |
| Generated launch plans | `GGML_VK_VISIBLE_DEVICES=1` for both models, selecting the B60; no manual GPU-index override |
| Python | **CPython 3.12.10**, newly created venv, system-site-packages disabled |
| arc-llama | Editable `0.9.2.dev0` from the exact commit above |
| llama.cpp | Release **b11823**, `0.7.0-dev`, build 11823, commit **5c19a5a75**, Clang 20.1.8 for Windows x86_64 |
| Runtime asset | `llama-b11823-bin-win-vulkan-x64.zip`, 36,875,569 bytes |
| Installer recorded SHA-256 | `5f75e297498e1036a70f42f02b40bfdd0fe2bda0749985f7a59dabc624b88ab0` |
| Recipe | Context 32,768; KV q8_0/q8_0; `-ngl 999`, parallel 1, ubatch 1024, batch 2048, flash attention on |
| Router | `single_resident=true`; background autotune left enabled at its default |

`gpus` and `doctor` additionally reported four Intel B-series registry entries (DEV_E2F0, E2F1, E2F7, E2FF), unknown VRAM and missing driver details. They are disabled in the automatically generated config. CIM video-controller enumeration reported only the B60, RTX and Parsec adapter. This discrepancy is recorded, not treated as five confirmed physical Intel GPUs. PCI location discovery fell back to synthetic slots.

See [hardware.log](release-validation-windows-2026-10-22/hardware.log), [runtime-inspection.log](release-validation-windows-2026-10-22/runtime-inspection.log), and [runtime-build-and-plans.log](release-validation-windows-2026-10-22/runtime-build-and-plans.log).

## Isolation and installation

Fresh checkout, venv, downloads and state are all under:

```text
C:\Users\user\Documents\Codex\2026-10-22\repo-https-github-com-offbyonebit-arc\work\windows-validation-2026-10-22
```

The checkout is `arc-llama\`, the new venv is `venv\`. Before any arc-llama command, `env.ps1` sets process-local variables:

```powershell
$env:APPDATA       = "$ValidationRoot\appdata"
$env:LOCALAPPDATA  = "$ValidationRoot\localappdata"
$env:HF_HOME       = "$ValidationRoot\hf-cache"
$env:UV_CACHE_DIR  = "$ValidationRoot\uv-cache"
$env:PIP_CACHE_DIR = "$ValidationRoot\pip-cache"
$env:TEMP         = "$ValidationRoot\temp"
$env:TMP          = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = "$ValidationRoot\venv\Scripts;" + $env:PATH
```

The real APPDATA/LOCALAPPDATA arc-llama config and models and existing venvs were not used or edited. No source fixes, dependency pins, commits, pushes, PRs, or issue comments were made. The test suite ran with its existing per-test config-home fixture in addition to the outer scratch environment. It overlapped the inference checks; therefore the unexplained intermediate runtime-path failure cannot be attributed to arc-llama, tests, or host tooling from this run alone.

Commands below ran in PowerShell, with stdout/stderr captured using `2>&1 | Tee-Object LOG`; explicit exit codes were appended for commands that returned normally. Interactive server sessions were stopped with an actual Ctrl+C sent to their PTY. PowerShell/PTY sessions returned 1 on interruption; their logs do not consistently contain a final application-shutdown-complete line. Independent process and listener checks are the cleanup evidence.

| Command / check | Result | Full output |
| --- | --- | --- |
| `git clone --branch release/0.9.2-work-mode --single-branch https://github.com/offbyonebit/arc-llama.git arc-llama` | First sandbox attempt: `getaddrinfo() thread failed to start`. Network-enabled retry succeeded. | `clone.log`, `clone-retry.log` |
| `git rev-parse HEAD`; `git branch --show-current` | Exact requested commit and branch, as recorded above. | This report |
| `py -0p`; `python --version` | Launcher said no installed Pythons; direct Python was available and reported 3.12.10. | Environment tool output transcribed here |
| `python -m venv ...\venv` | First sandbox attempt failed: ensurepip subprocess exited 1. Retrying creation outside the sandbox succeeded in the same newly created venv. | `venv-retry.log`; initial failure transcribed here |
| `python -m pip --version` | pip 25.0.1 from the new venv, Python 3.12. | Installation tool output |
| `python -m pip install -e ".[dev]"` | **Exit 0**, editable package and dev dependencies installed without pins. | `pip-install.log` |
| `python -m pip freeze` | Full resolved versions saved. Includes pytest 9.1.1, pytest-asyncio 1.4.0, FastAPI 0.161.0, Starlette 1.9.0, httpx 0.28.1, huggingface-hub 2.2.0. | `pip-freeze.txt` |
| CIM OS/GPU/process queries | Sandbox CIM reads initially denied; read-only network/hardware-enabled retry succeeded. Baseline llama-server process list was empty. | `hardware.log`, `process-baseline.json` |
| `arc-llama --version` | **Exit 0**, `arc-llama, version 0.9.2.dev0`. | `version.log` |
| `arc-llama gpus` | **Exit 0**, B60 23.8 GB plus four unknown-VRAM entries described above. | `gpus.log` |
| `arc-llama doctor` | **Exit 0**. Config absent at the scratch path; Level Zero loader available; vulkaninfo available; sycl-ls/clinfo/oneAPI setvars absent. Windows kernel/group checks unavailable by design. | `doctor.log` |
| `Test-Path $env:APPDATA\arc-llama\config.toml` after diagnostics | **False**: empty-config starting state confirmed. | Diagnostic tool output |

## Fresh install, API and browser

Initial command, without CA overrides:

```powershell
arc-llama run 'Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M' --name qwen05
```

**Exit 1** at runtime installation: `[SSL: CERTIFICATE_VERIFY_FAILED] ... unable to get local issuer certificate`. Detection and scratch config creation had succeeded. This is the original fresh-install result; it is not erased by the subsequent success.

For the explicit retry, 88 public certificates from Windows ROOT and CA stores were exported using `ssl.enum_certificates` and `ssl.DER_cert_to_PEM_cert` to `windows-trust.pem`. In that command process only:

```powershell
$env:SSL_CERT_FILE = "$ValidationRoot\windows-trust.pem"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
# Repeat the same arc-llama run command.
```

Certificate verification remained enabled. No trust stores or security protections were modified. This retry downloaded and verified the official portable runtime itself; it did not use a preinstalled llama-server. The model also downloaded into the new scratch models directory. Launch readiness estimated 1,685/24,382 MiB. API listening address was `http://127.0.0.1:11437`.

| Exercise | Result | Evidence |
| --- | --- | --- |
| Original `run`, empty config | **FAIL**, TLS certificate verification, exit 1. | `run-first.log` |
| Windows-CA retry | Runtime b11823 and Q4_K_M downloaded, config/model registered, API started. First request loaded model cold. | `ca-export.log`, `run-windows-ca.log` |
| `GET /v1/models` | **HTTP 200**, qwen05 and filename alias listed. | `json-request.log` |
| JSON `POST /v1/chat/completions` | **HTTP 200**, assistant: `Hello! It's nice to meet you.` | `json-request.json`, `json-response.json`, `json-request.log` |
| Browser open `/`, then `/chat?model=qwen05` and send chat | Initial message failed visibly: `runtime_missing`; load endpoint HTTP 503. | `windows-ui-first-failure.png`, server log |
| Initial streaming `POST /v1/chat/completions` | **HTTP 503**, structured `runtime_missing` diagnostic, exit 1. | `stream-request.json`, `stream-response.sse`, `stream-request.log` |
| Ctrl+C after first server | No llama-server process remained. | `process-after-ctrlc-first.json` |
| Repeat normal `run` with Windows CA settings | Installer re-downloaded and verified b11823; existing scratch Q4 model reused; server started. | `run-recovery.log` |
| Retry streaming request | **HTTP 200**, incremental `chat.completion.chunk` deltas, finish_reason stop, final `data: [DONE]`, exit 0. | `stream-retry-request.json`, `stream-retry-response.sse`, `stream-retry.log` |
| Browser reload and resend | **PASS**, visible assistant response: `Hello, welcome to Windows validation!` | `windows-ui-chat-success.png`, server log |
| Ctrl+C after recovery | No llama-server process remained. | `process-after-ctrlc-recovery.json` |

The JSON payloads and complete raw SSE responses are included beside the logs. Prompts were synthetic greetings; no private documents or user data were sent to the local model.

### Runtime-path anomaly

Between successful requests, the browser and API reported `llama-server runtime not found`. Filesystem observations also reported the executable absent/inaccessible in those checks, while sibling runtime DLLs remained. The first process snapshot showed PID 16284; a later failure snapshot showed no llama-server. Recovery streaming used PID 38136. The browser response completed on recovery, but its model pill later said Not loaded and another filesystem observation reported the executable unavailable. Windows Defender's `Get-MpThreatDetection` returned no entries; this does not exclude other causes or establish that security software removed it.

The normal installer recovered the runtime twice, preserving each attempt's log. A later read-only inspection found `llama-server.exe` present (9,216 bytes), successfully returned its version, and enumerated both GPUs. **No root cause is established and no claim is made that the file was permanently deleted.** The overlapping suite and execution contexts are a confounder; a serial reproduction is needed to classify this. No antivirus exclusions, certificate bypasses, source patches, or process-kill workarounds were applied.

## Second registration and model switching

Both models came from public repository `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (HF requests resolved revision `9217f5db79a29953eb74d5343926648285ec7e67`).

| Name | GGUF | Bytes | Backend port |
| --- | --- | ---: | ---: |
| qwen05 | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 | 18080 |
| qwen05-q8 | `qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 | 18081 |

```powershell
arc-llama add 'Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0' --name qwen05-q8 --from-hf
# After registration, restart the server so its in-memory config includes both:
arc-llama run qwen05
```

Registration **exited 0**. `run qwen05` used the normal installer to recover the unavailable executable, then loaded both registrations. Sequential JSON completion requests used the same API process:

1. Request `model=qwen05`: **HTTP 200**; exactly **one** llama-server process, **PID 33072**, command line points to Q4_K_M.
2. Request `model=qwen05-q8`: **HTTP 200**; exactly **one** llama-server process, **PID 38856**, command line points to Q8_0. The Q4 PID was no longer present in the complete llama-server list.
3. Authenticated `/admin/status`: **HTTP 200**, `single_resident=true`; qwen05 `loaded=false`, `state=idle`, `pid=null`; qwen05-q8 `loaded=true`, `state=ready`, `pid=38856`.
4. Send Ctrl+C while Q8_0 is resident: **zero llama-server processes**, and no listeners on 11437, 18080 or 18081.

Evidence: `add-second.log`, `run-switch.log`, `switch-requests.log`, both `switch-*-response.json`, both `switch-process-*.json`, `switch-admin-status.json`, `process-after-ctrlc-switch.json`, `shutdown-check.log`, and `listeners-after-ctrlc.json`.

**Policy-driven first-model eviction passed. Real VRAM-pressure eviction remains untested:** the models comfortably fit the B60, and no artificial memory exhaustion was attempted.

## Automated checks

### Full suite

```powershell
pytest tests -q
```

**FAIL — exit 1. 29 failed, 1,587 passed, 25 skipped, 14 warnings in 586.58 seconds (9m46s).**

Observed failure groups (classification describes the tracebacks, not a proven root cause):

| File | Failures | Observed symptoms |
| --- | ---: | --- |
| `test_agent_changes.py` | 1 | Symlink creation denied: WinError 1314, required privilege absent. No system setting was changed to hide this. |
| `test_harness_hermes.py` | 7 | Fake harness produces no expected text/approval/output files; startup/cleanup assertions fail. |
| `test_harness_opencode.py` | 13 | Fake `.py` executable not found/started as expected; missing log.jsonl, 30-second timeouts, cleanup timeout and command-discovery mismatches. |
| `test_launcher.py` | 1 | Windows backslash executable path compared with POSIX `/bin/llama-server`. |
| `test_model_runtime.py` | 3 | Capability probe/flash-attention isolation expectations and private runtime path resolution fail. |
| `test_server_harness.py` | 1 | Relative project path resolves without expected `home` segment. |
| `test_work.py` | 1 | `WorkConflictError: Source changed independently: hello.txt`. |
| `test_work_profiles.py` | 1 | Harness tool list differs from expected fake response. |
| `test_work_verify.py` | 1 | Unsafe/malformed-criteria normalization retains an unexpected item. |

Timeouts emitted stacks but the suite eventually completed without interruption. Warnings include unraisable Windows asyncio subprocess transport cleanup errors (`Event loop is closed` / closed pipe). Full warning text is retained. Exact failed node IDs:

```text
tests/test_agent_changes.py::test_symlinks_are_not_followed
tests/test_harness_hermes.py::test_hermes_streams_reasoning_and_text_and_cleans_up
tests/test_harness_hermes.py::test_hermes_edits_wait_for_single_use_approval[True-approve]
tests/test_harness_hermes.py::test_hermes_edits_wait_for_single_use_approval[False-deny]
tests/test_harness_hermes.py::test_hermes_rejects_unsupported_client_rpc
tests/test_harness_hermes.py::test_hermes_cancel_reaps_process_and_removes_home
tests/test_harness_hermes.py::test_hermes_startup_timeout_cleans_up
tests/test_harness_hermes.py::test_hermes_child_does_not_inherit_credentials_or_approval_bypasses
tests/test_harness_opencode.py::test_find_binary
tests/test_harness_opencode.py::test_run_end_to_end_auto_confirm
tests/test_harness_opencode.py::test_run_routes_permission_through_confirm_callback[True]
tests/test_harness_opencode.py::test_run_routes_permission_through_confirm_callback[False]
tests/test_harness_opencode.py::test_missing_binary_and_missing_callback_are_clean_errors
tests/test_harness_opencode.py::test_missing_project_folder_is_reported_not_raised
tests/test_harness_opencode.py::test_auto_confirm_never_answers_a_question_the_harness_raised
tests/test_harness_opencode.py::test_a_cancelled_run_terminates_the_harness_even_though_its_awaits_are_cancelled
tests/test_harness_opencode.py::test_permission_is_seen_even_when_opencode_list_endpoint_fails
tests/test_harness_opencode.py::test_a_slash_command_runs_through_the_harness_instead_of_a_prompt
tests/test_harness_opencode.py::test_a_command_the_harness_rejects_is_reported
tests/test_harness_opencode.py::test_list_commands_reads_the_harness_and_caches
tests/test_harness_opencode.py::test_generated_session_reports_id_and_uses_task_title
tests/test_launcher.py::TestBuildPlan::test_includes_model_and_port
tests/test_model_runtime.py::test_model_runtime_launch_isolation_and_literal_arguments
tests/test_model_runtime.py::test_config_roundtrip_old_defaults_and_model_path
tests/test_model_runtime.py::test_probe_environment_and_cache_are_model_specific
tests/test_server_harness.py::test_missing_folder_suggests_the_closest_existing_one_and_relative_means_home
tests/test_work.py::test_work_isolation_review_apply_undo_and_conflicts
tests/test_work_profiles.py::test_probe_tools_reports_what_the_harness_sends_the_model
tests/test_work_verify.py::test_normalise_drops_unsafe_and_malformed_criteria
```

Complete combined stdout/stderr, including all timeout thread dumps and final failure tracebacks, is in [pytest.log](release-validation-windows-2026-10-22/pytest.log). The suite was run unchanged, with no test selection, xfail additions, or dependency downgrades.

### Published config compatibility

```powershell
python scripts/check_config_versions.py --versions 0.9.1.2
```

First attempt **exited 1**: uv could not install the published package because of `invalid peer certificate: UnknownIssuer`. The failure and traceback are in `config-versions.log`.

Explicit retry, leaving the script unchanged:

```powershell
$env:UV_NATIVE_TLS = 'true'
python scripts/check_config_versions.py --versions 0.9.1.2
```

**Exit 0**. Upgrade from 0.9.1.2 preserved all **53 values**. Rollback through the published release reset four settings (reported, not counted as upgrade failures):

| Setting | Current value | After old release saves |
| --- | --- | --- |
| `models[0].env.PROBE_KEY` | `probe-value` | default / missing |
| `models[0].llama_server` | `probe-llama_server` | empty string |
| `server.inference_first_response_seconds` | 601.5 | 600.0 |
| `server.inference_stream_idle_seconds` | 121.5 | 120.0 |

Full retry output: [config-versions-windows-ca.log](release-validation-windows-2026-10-22/config-versions-windows-ca.log). The script created its own throwaway release venv; existing venvs were untouched.

## Evidence and remaining gates

The adjacent `release-validation-windows-2026-10-22/` folder contains full command logs, request/response bodies, process snapshots, hardware/runtime details, the exact resolved dependency list and `env.ps1`. Screenshots include both the first failure and successful browser retry. Empty initial/intermediate process captures mean the CIM query returned no rows; the final snapshot explicitly contains `[]` and count 0.

Release follow-up requires resolving or independently reproducing the TLS defaults and intermittent runtime-path behavior, assessing the actual suite failures below, and exercising real VRAM-pressure eviction for #73. This record does not validate Linux, SYCL, other driver/GPU combinations, experimental Work agents, or an historical rc1 Windows run.

Final process audit after the suite: no validation Python/arc-llama/pytest/llama-server process rows and no listening ports on 11437, 18080 or 18081. See `final-validation-processes.json` and `final-listeners.json`.
