# Windows portability fixes — 2026-10-22

Changes for `release/0.9.2-work-mode`, based on `3af34e51db1634e307f454026cf7819be7caf62a`. The original Windows validation report and its failures are preserved separately.

## Changes

- **Windows runtime TLS:** the automatic runtime downloader uses CPython's verified Windows ROOT/CA trust store when no `SSL_CERT_FILE` or `SSL_CERT_DIR` override is supplied. Linux retains httpx's existing defaults. Explicit CA settings and caller-supplied clients retain their existing behavior. Verification and hostname checks remain enabled.
- **Config-version checker:** native Windows trust is selected for uv only on Windows. Probe subprocesses get isolated HOME, USERPROFILE, APPDATA and LOCALAPPDATA; uv receives the same environment. Existing caller TLS preferences are preserved.
- **Project paths:** reject rooted and drive-relative Windows paths such as `\etc\passwd` and `C:outside.txt`. `is_absolute()` alone does not reject those Windows forms. The change is equivalent to the old check for normal POSIX paths.
- **Harness cleanup:** report a missing working directory consistently before process creation; close the tool-probe listening socket. Existing Linux process-group shutdown semantics are retained, with an explicit platform guard that also passes Windows type checks.
- **Portable test executables:** native Windows console launchers replace Unix shebang assumptions in test doubles. A Windows-only **dev** dependency on distlib supplies those launchers; installed application dependencies do not change. Linux doubles still use executable Python scripts. The adapters are not taught to run arbitrary scripts through a shell.
- **Storage-safe synthetic files:** the only 26 GiB synthetic test fixture is explicitly sparse on Windows. Linux behavior stays the same.
- **Tests and exact bytes:** compare native paths, set Windows USERPROFILE when faking a home directory, and restore exact LF bytes in the undo test. Production conflict detection remains byte-exact. Only the symlink test lacking WinError 1314 privilege is skipped; Linux retains real symlink coverage.
- **Existing Linux key-cache bug:** the unmodified Linux baseline exposed revoked keys staying cached when atomic file replacements shared an mtime. The cache now checks file identity, size and mtime. A regression test replaces the key file with equal-length content and the exact same timestamp, then checks that the old key is rejected.

## Verification

Both full suites use warnings as errors:

```text
pytest tests -W error -q -ra --tb=short -p no:cacheprovider
```

| Check | Result |
| --- | --- |
| Original native Windows suite | 29 failed, 1,587 passed, 25 skipped (recorded before fixes) |
| Original Ubuntu WSL baseline | 12 failed, 1,603 passed, 26 skipped; original source, fresh Linux venv |
| Final native Windows suite | **1,627 passed, 26 skipped**, exit 0 (316.51 s) |
| Final Ubuntu WSL suite | **1,627 passed, 26 skipped**, exit 0 (73.30 s) |
| Windows ruff | All checks passed |
| Windows mypy | No issues in 83 source files |
| Linux ruff | All checks passed, exit 0 |
| Linux mypy | No issues in 83 source files, exit 0 |
| Config compatibility, no external TLS override | Exit 0; upgrade preserved all 53 values; same four expected rollback losses |
| Fresh automatic runtime install, no CA overrides | Pass: downloaded and verified official Vulkan b11826 |
| Serial real inference stability | 13/13 HTTP 200 completions over 195.6 seconds; runtime exists and SHA-256 is identical in all 13 samples |
| Original b11823 follow-up | 7 JSON completions and one SSE completion on the original b11823; result recorded below |
| Package wheel build | Passed; distlib metadata confirms Windows/dev-only dependency |
| Final affected router suite | Windows 20 passed (20.56 s); Linux 20 passed (0.82 s) |
| Patch integrity | `git diff --check` and `git apply --reverse --check` pass |

Windows is Windows 11 Pro 25H2 build 26200.9457, Python 3.12.10, Intel Arc Pro B60 driver 32.0.101.8735, portable Vulkan. The fresh default install resolved b11826 (commit a08ab75ae), newer than the first validation's b11823. Serial tests use only scratch config, models and runtime directories. No Windows test suite overlaps the serial inference check.

Linux regression testing used the existing Ubuntu WSL installation (kernel 5.15.167.4-microsoft-standard-WSL2), **Python 3.12.3**, a newly bootstrapped venv and extracted source snapshots under `/tmp`. No existing Linux venv or system Python packages were changed. Windows public CA certificates were supplied to the Linux test dependency installer; product Linux TLS behavior is unchanged. The original baseline's harness failures were reproduced before testing the patch. Node.js is unavailable in that Linux environment, so Node-dependent static UI tests remain skipped; this is not Linux GPU-inference validation.

The first strict targeted Windows rerun was 220 passed, 2 skipped, with one socket ResourceWarning failure. That finding was fixed before the full runs. The first strict Linux candidate found the pre-existing key-cache issue, the same socket cleanup problem and an unreaped test double; all findings and final reruns are retained in the evidence. No warning filters or broad xfails were added to get a passing suite.

The full-suite results above cover the final production code. Afterwards, only `tests/helpers.py` and `tests/test_router.py` changed to make the 26 GiB fixture actually sparse on Windows. The affected router suite then passed **20/20 on Windows and 20/20 on Linux**. A final Windows assertion also verifies actual allocated bytes (under 1 MiB), not just the sparse attribute. Those targeted runs are recorded separately; the full-suite counts are not inflated to include the added fixture test.

The fixture uses Windows `FSCTL_SET_SPARSE` followed by native `SetFilePointerEx`/`SetEndOfFile`. Plain CRT `truncate()` zero-fills the extension even when the sparse flag is set. The first fixture attempt therefore failed with disk-full errors; its two failures and subsequent cleanup are preserved in `fixes-sparse-windows-first-failure.log`. The corrected 26 GiB fixture passes without that allocation. Linux continues to use its original sparse `truncate()` behavior. See [Microsoft's sparse-file documentation](https://learn.microsoft.com/windows/win32/api/winioctl/ni-winioctl-fsctl_set_sparse).

## Environment failures retained

One full Windows follow-up exhausted drive C during its later tests. The logging sink also reported `There is not enough space on the disk`, so that run has no trustworthy complete result. Its output is retained as `fixes-full-windows-disk-full.log`. Only completed test scratch directories under this validation workspace were removed; no existing user data, model library or venv was removed. The next full run started with **41,739,042,816 bytes free**.

The cleanup audit found 11 real OpenCode children from the original failed fixture run. Their live `cmd.exe` parents all recorded original pytest **PID 34008** as their parent, with creation times matching the initial tests. Those specific children were stopped and the audit was saved. They had locked databases inside original test scratch directories. The original final audit only covered Python/pytest/llama-server and did not detect them. Native executable fixtures now prevent the invalid fake-script path from falling back to installed OpenCode. No unrelated OpenCode process was stopped.

The initial Ubuntu setup also lacked ensurepip/python3-venv. Rather than install system packages, a pip wheel was bootstrapped inside a new `--without-pip` venv. These environment adjustments are recorded separately from the source fixes.

## Runtime-path investigation

The earlier intermittent `runtime_missing` result is **not claimed fixed by an unrelated patch**. A fresh serial run without overlapping Windows tests completed 13 real requests with the executable present and unchanged throughout. The same-build follow-up reused the original scratch **b11823 / 5c19a5a75**, without reinstalling it. All seven JSON requests and an SSE request returned HTTP 200, with `[DONE]` observed. The executable existed and had SHA-256 `d58a2f3f0632a9e9c53cc13ae896b4b34a68b17d6826820ee2ef06fc59f068a7` in every sample over 70.3 seconds. Ctrl+C left zero llama-server processes and no listeners on the validation ports. This did not reproduce the earlier intermittent error; its cause remains unproven.

There is no speculative redownload-on-request or weakened missing-file check. The original failure remains in the initial report for follow-up if reproduced. The successful fresh install establishes the TLS correction; a bounded stability run does not prove the absence of every intermittent host/storage issue.

Real VRAM-pressure eviction, Linux GPU inference, SYCL, other Windows editions/drivers and the full Python-version CI matrix remain outside this follow-up. The original run already verified single-resident model switching and zero remaining llama-server processes after Ctrl+C.

## Review and artifacts

- The separately delivered validation archive includes `windows-portability-fixes.patch`: source and test changes, suitable for review or `git apply` from the repository root.
- The archive also includes `windows-portability-fixes-evidence/`: complete final test/lint/type/config logs, original Linux baseline, intermediate strict-test failures, fresh-install/stability logs and a source hash manifest.
- The original validation report and archive are unchanged.

No pushes, PRs or issue comments were made during validation. The user's existing arc-llama config, model library and venvs were not edited.

Final process audit: no remaining validation Python/pytest/arc-llama/llama-server processes; no listeners on 11437, 18080 or 18081. Completed test scratch directories were removed after their logs were saved. Final C free space: **39,385,325,568 bytes**. No unrelated processes or user files were removed.
