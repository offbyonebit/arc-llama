# Work harnesses

Work currently supports the built-in agent, OpenCode, and Hermes Agent. External harnesses require `ARC_LLAMA_EXPERIMENTAL_AGENT=1`. Choose the agent in Work after choosing a project folder and a model.

## Harness conversation history

**Sessions** and `/sessions` list saved Work conversations alongside the newest
200 top-level, unarchived sessions from each installed harness: OpenCode and
Hermes. History is read directly from their local SQLite databases without
starting a harness, contacting a provider, or loading a model. The same Work
admin authentication protects the list and transcripts.

Open a harness session to see its user/assistant text and restore its project
folder when available. A follow-up saves a separate Work copy and uses the
currently selected local model, tools, approvals, and normal bounded Work
conversation context. This imports the conversation; it does not resume the
native harness session or replay its tool calls, attachments, credentials,
system instructions, or permissions. Long histories can be compacted before
continuing. Sessions with over 2000 messages or more than 4 MiB of source or
converted history show an explanation instead of a partial import.

Native sessions have a harness label and no Delete button. Deleting a saved
Work copy never deletes the original; it can appear again from harness history.
A missing database is normal, and an unreadable/incompatible database produces
a warning while saved Work conversations remain usable.

OpenCode history follows `$XDG_DATA_HOME/opencode/opencode.db` (default
`~/.local/share/opencode/opencode.db`). Hermes history follows
`$HERMES_HOME/state.db` (default `~/.hermes/state.db`). External history requires
the same experimental-agent switch as Work's harness integrations.

## Installing a harness from Arc Llama

If OpenCode or Hermes isn't installed, Work's Agent menu shows it as **(install…)**. Choosing it opens a dialog that
shows the version, the source repository, the license (both are MIT), the download size and where the copy will be
kept. Nothing is downloaded until you choose Install. **System → Agents for Work** lists each harness and where Work
will find it, and lets you install, update or remove the copies Arc Llama manages. The same is available from the
command line:

```bash
arc-llama install-harness opencode
arc-llama install-harness hermes
arc-llama install-harness opencode --list
arc-llama install-harness opencode --remove v1.19.4
```

**Tested versions:** Arc Llama installs the version it was verified against (OpenCode v1.19.4, Hermes v0.22.5).
*Install newest instead* (`--latest`) installs the newest upstream release, which is recorded as untested.

**Where it lives:** each version gets its own folder under `<state_dir>/harnesses/<name>/`. A successful install
records the executable in the config (`agent.opencode_bin` or `agent.hermes_bin`), so Work uses that copy ahead of
`ARC_LLAMA_*_BIN` and PATH. Your own OpenCode or Hermes installation, `~/.hermes`, and OpenCode's data are never
changed. Removing the selected copy clears the setting, and Work then falls back to the environment variable or PATH.
A copy can't be removed while a Work task is using it.

**OpenCode:**
- The installer downloads the official CLI archive for your platform from
  [GitHub releases](https://github.com/anomalyco/opencode/releases), choosing the musl build or the baseline build
  for CPUs without AVX2 when needed.
- It checks the archive against GitHub's SHA-256 digest and refuses an archive that has no digest.
- It extracts safely, runs `opencode --version`, and then publishes the folder with the same rename-and-recover steps
  as llama.cpp runtimes. An install killed partway is repaired by the next install.

**Hermes:**
- Hermes publishes no binaries and supports only editable installs, so Arc Llama downloads the tagged source and
  checks it against a pinned SHA-256. A newest install has no pinned hash and relies on GitHub's TLS.
- The source is kept in the install folder. `uv sync --frozen --extra acp` builds a private environment from the
  project's hash-locked `uv.lock`.
- uv also downloads the Python version Hermes requires (3.14), kept under `harnesses/tools/`. If uv isn't on PATH, a
  pinned, digest-checked uv release is downloaded there as well. The user's uv configuration and indexes are not used
  (`UV_NO_CONFIG`).
- The install counts as complete only after `hermes acp --check` passes, run with a temporary `HERMES_HOME`. A failed
  or cancelled install removes its folder.
- Hermes on Windows has not yet been validated through Arc Llama, and the dialog says so.

Installing needs `ARC_LLAMA_EXPERIMENTAL_AGENT=1` and the admin token, the same as using external harnesses. One
install runs at a time and can be cancelled. Disk space is checked first: about 200 MB for OpenCode, and up to 1 GB
while Hermes installs.

## Hermes

To use your own Hermes instead, install it with ACP support using the [official ACP instructions](https://hermes-agent.nousresearch.com/docs/user-guide/features/acp). `hermes acp --check` should succeed. Arc Llama finds `hermes` on PATH; `ARC_LLAMA_HERMES_BIN` can name another executable.

Each task starts Hermes through its ACP stdio protocol, with the selected local model routed through Arc Llama. Work supplies the project folder and earlier conversation, displays streamed replies and tool activity, tracks actual prompt usage, and keeps the conversation in its saved sessions. File changes use Work's normal review and Undo flow.

Hermes runs in default ACP approval mode. File edits and commands flagged by Hermes's native checks ask in Work. Routine terminal commands can run under Hermes's own policy. Work shows this policy when Hermes is selected. Auto-approve is unavailable for Hermes and the API rejects requests that try to enable it. The client grants only individual approvals, never a persistent permission. Native file-edit approvals for paths outside the chosen project are denied; this is not an OS sandbox for terminal commands.

The initial integration uses a temporary, isolated Hermes home. It does not change the user's Hermes installation or import its global credentials, MCP configuration, skills, or memory. Primary and auxiliary inference use the same Arc Llama endpoint, with cloud fallback disabled. OpenCode tool profiles and native slash commands are unavailable for Hermes; Work's own `/compact` remains available.

Before the first request, Work can only estimate the user's composed prompt for an external harness. Once Hermes makes an inference request, the private relay counts the full prompt including Hermes instructions and tools. Exactness still depends on whether the local runtime can tokenize that request.

## Verification

`scripts/check.sh` runs all tests. `scripts/check.sh --live` also exercises the installed Hermes ACP process against fake local inference: streamed output, prompt counts, approved and denied edits, and cancellation. It does not call a cloud provider or load a model. The same flag retains the real OpenCode profile checks.


Work associates each generated harness session with its saved conversation using
IDs only. Sessions shows the Work conversation once while its copy is saved,
with original harness entries accessible under an expandable disclosure;
deleting the Work copy reveals the untouched harness entries again. Associations
are stored separately so a browser save cannot erase them. Older sessions with
the same title, project folder and harness collapse into an expandable group;
each retains its own Open action and identity. New OpenCode session titles use
the user's task rather than the project/context wrapper.
