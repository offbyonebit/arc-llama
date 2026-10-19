# Work harnesses

Work currently supports the built-in agent, OpenCode, and Hermes Agent. External harnesses require `ARC_LLAMA_EXPERIMENTAL_AGENT=1`. Choose the agent in Work after choosing a project folder and a model.

## Hermes

Install Hermes with ACP support using the [official ACP instructions](https://hermes-agent.nousresearch.com/docs/user-guide/features/acp). `hermes acp --check` should succeed. Arc Llama finds `hermes` on PATH; `ARC_LLAMA_HERMES_BIN` can name another executable.

Each task starts Hermes through its ACP stdio protocol, with the selected local model routed through Arc Llama. Work supplies the project folder and earlier conversation, displays streamed replies and tool activity, tracks actual prompt usage, and keeps the conversation in its saved sessions. File changes use Work's normal review and Undo flow.

Hermes runs in default ACP approval mode. File edits and commands flagged by Hermes's native checks ask in Work. Routine terminal commands can run under Hermes's own policy. Work shows this policy when Hermes is selected. Auto-approve is unavailable for Hermes and the API rejects requests that try to enable it. The client grants only individual approvals, never a persistent permission. Native file-edit approvals for paths outside the chosen project are denied; this is not an OS sandbox for terminal commands.

The initial integration uses a temporary, isolated Hermes home. It does not change the user's Hermes installation or import its global credentials, MCP configuration, skills, or memory. Primary and auxiliary inference use the same Arc Llama endpoint, with cloud fallback disabled. OpenCode tool profiles and native slash commands are unavailable for Hermes; Work's own `/compact` remains available.

Before the first request, Work can only estimate the user's composed prompt for an external harness. Once Hermes makes an inference request, the private relay counts the full prompt including Hermes instructions and tools. Exactness still depends on whether the local runtime can tokenize that request.

## Verification

`scripts/check.sh` runs all tests. `scripts/check.sh --live` also exercises the installed Hermes ACP process against fake local inference: streamed output, prompt counts, approved and denied edits, and cancellation. It does not call a cloud provider or load a model. The same flag retains the real OpenCode profile checks.
