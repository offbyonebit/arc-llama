# Work projects and persistent jobs

Work is a local, admin-controlled project workbench. It uses registered local
chat models and a separate project copy for every job. Opening or closing the
browser does not own a job's lifetime.

## Projects

Register an absolute folder path on the server, select a local model, and save
project instructions. Enable or disable file edits for future jobs. Optional
command permission also requires an OS sandbox and approval for each command.
A job captures its project settings when created; changing the project later does
not silently widen an existing job's permissions.

The copy contains current regular files, including uncommitted changes. It is
not a Git checkout and does not share file inodes with the source. Git metadata,
virtual environments, dependency folders, build outputs, .env files, and symbolic
links are omitted. Copy limits are 4000 files, 16 MiB per file, and 128 MiB total.
Select a smaller folder when a project exceeds those limits. There can be 20
retained copies; Discard removes a copy while keeping its audit history.

## Jobs and recovery

A serial background worker handles preparing, queued, running, waiting-for-
approval, verifying, settling, paused, interrupted, completed, failed, and
discarded states. A SQLite
transaction saves each model response, tool receipt, state transition, and event.
The OS grants one server ownership of a Work state directory, preventing two
servers from executing the same queued actions.

Pause takes effect after the current step. A browser reload can reconnect to the
same saved job. On server restart, an interrupted model call pauses for manual
resume. A completed tool receipt is reused, not executed again. If the server
stopped while a tool was executing and its outcome is unknown, resume requires
acknowledgment after reviewing the copy; the old action becomes an uncertainty
receipt and is never automatically replayed. A restart during preparation fails
that incomplete copy and requires a new job. Transient execution errors preserve
progress for review/resume; the 30-model-turn limit ends that attempt. If the
project permits further attempts and enough budget remains, Work can start a
fresh one.

A command approval names the exact server-generated tool-call identifier. It
cannot approve a later command. Denials are recorded as tool results. Work's
initial tool set is read_file, list_directory, literal search_files, write_file,
apply_patch, and optional run_command. MCP and user skill handlers are not
executed by Work v1. Tool results are bounded to 32000 characters.

## Checking results and trying again

Result checking is enabled by default. Before work starts, a model proposes up
to eight acceptance criteria using the task, project instructions, and file
listing. Code checks file existence, literal contents, and file counts. Safety
checks flag unexpected file deletion or emptying. Criteria that require judgment
go to the same local model in a fresh conversation, with bounded file listings
and a diff. A separate conversation is not an independent model or proof of
correctness; unsupported passes become unknown. These checks do not execute a
test suite or establish that the chosen criteria cover the whole task.

The result distinguishes checked-by-code, judged-only, partly verified, failed,
and unverified outcomes. A verified result requires at least one passing code
criterion and no failed or unknown criteria; model judgments may also contribute.
Failed checks can send the agent back for up to two repair passes per attempt,
subject to the turn limit. A completed job can still have failed or incomplete
verification. Always review the criteria, evidence, activity, and produced files.

Projects can request one to five sequential attempts (one by default) and an
attempt budget of one to 720 minutes (30 by default). Multiple attempts require
result checking. Each attempt starts from a fresh copy of the original baseline
and uses the same planned criteria. A complete verified result stops retries.
Otherwise the budget permits another attempt only when elapsed time plus the
average completed-attempt duration fits. This is an admission estimate, not a
hard timeout: an attempt already running is allowed to finish, and approval waits
and pauses count toward elapsed time.

Work retains the best copy: complete attempts precede turn-limited attempts,
then verified, judged-only, partly verified, unverified, and failed outcomes,
then the number of passing checks; ties keep the earlier attempt. The Attempts
section shows the history and the selected result. Only the selected copy is
available for review/apply. Disabling result checking runs a single attempt.
Verification cannot be paused, and source changes cannot be applied or the copy
discarded while verification or a copy transition is active. Copy replacements
are journaled before filesystem changes; restart recovery completes the saved
transition before exposing a result. A recovered new attempt pauses for manual
resume. A restart during verification leaves an
unverified result for review rather than claiming success.

## Review, apply, undo, and discard

Review changes compares the copy with its original baseline. Apply requires the
revision of that reviewed diff. It refuses a stale review or an independent
change to any affected source file or to the source directory's identity.
The agent has no apply-to-source tool; only an explicit admin action can apply.

An apply intent is persisted before source writes. Each file replacement is
atomic, and ordinary I/O failures attempt to roll back already-written files.
If the server stops during apply, the saved receipt supports an explicit undo;
the application is not automatically repeated on restart. Undo checks all affected
files before restoring original content and refuses to overwrite later edits.
File/directory type changes require manual resolution. File mode changes are
not part of the content diff.

Discard deletes only the job's independent copy and baseline. Applied changes
must be undone before discarding, so the baseline needed for undo is retained.

## Optional commands

Commands require Linux, Bubblewrap, prlimit, and a working systemd user session.
There is no unrestricted-shell fallback. Availability in the UI checks installed
prerequisites; a disabled user namespace or unavailable user session produces a
tool error instead of bypassing isolation.

Bubblewrap unshares namespaces and clears the environment. System tools under
/usr, /bin, and the library directories are read-only. Only the project copy and
an isolated temporary directory are writable. The host home, GPU devices, source
project, service credentials, and host network are not mounted or passed through.
Commands use OS-installed tools; the project's excluded virtual environment is
not automatically exposed.

A systemd scope provides aggregate 1 GiB memory, no swap, 64-task, and one-core
CPU limits. prlimit additionally bounds address space, CPU time, individual file
size, and file descriptors. Commands have a 60-second wall-clock timeout, 128 KiB
output cap, and a project-size/file-count watchdog. Namespace children are tied
to the command's parent lifetime. These limits apply to commands, not to the
separately managed inference server.

## Storage and APIs

State is under `<state_dir>/work/`: work.sqlite3, worker.lock, and copies keyed by
random job identifiers. The state directory is private to its user on POSIX.
Source projects are never deleted by job cleanup.

All endpoints below require the admin token, including reads:

- `GET /admin/work`: projects, recent jobs, and command prerequisites.
- `POST /admin/work/workspaces`, `PUT /admin/work/workspaces/{id}` (including `verify`, `max_attempts`, and `time_budget_minutes`).
- `POST /admin/work/jobs`: workspace_id and task, with an optional local model.
- `GET /admin/work/jobs/{id}?after=EVENT_SEQUENCE`: state and paginated events.
- `GET /admin/work/jobs/{id}/changes`: files, diff, and review revision.
- `POST /admin/work/jobs/{id}/{pause|resume|approve|apply|undo|discard}`.

Resume accepts acknowledge_uncertain when needed. Approve requires call_id and a
boolean approved. Apply requires review_revision from the changes response.
