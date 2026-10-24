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

## Hermes

Install Hermes with ACP support using the [official ACP instructions](https://hermes-agent.nousresearch.com/docs/user-guide/features/acp). `hermes acp --check` should succeed. Arc Llama finds `hermes` on PATH; `ARC_LLAMA_HERMES_BIN` can name another executable.

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
