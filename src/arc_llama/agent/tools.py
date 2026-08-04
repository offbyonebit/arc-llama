"""Sandboxed filesystem and shell tools for the arc-llama coding agent.

All operations are constrained to a project root directory. Paths that escape
that root (via .. or absolute paths outside the root) are rejected.

Tools can be registered dynamically through ``ToolRegistry`` so that user
skills and MCP-style extensions can extend the agent without modifying core
code.
"""
from __future__ import annotations

import inspect
import io
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

GIT_MUTATION_COMMANDS = frozenset(
    {"git commit", "git push", "git pull", "git fetch", "git merge", "git rebase",
     "git reset", "git checkout", "git switch", "git branch", "git tag",
     "git stash", "git cherry-pick", "git revert"}
)

# Wrappers that prefix a command without changing what it ultimately invokes.
# Stripping them lets us catch e.g. `sudo git push` or `env git commit`. This is
# a guardrail, not a sandbox: we only unwrap a fixed, well-known set rather than
# trying to defeat arbitrary shell metaprogramming. Best-effort by design.
_COMMAND_WRAPPERS = frozenset(
    {"sudo", "env", "command", "exec", "time", "nice", "xargs"}
)

# Shell operators that separate independent commands on one line. Splitting on
# these lets the denylist catch a mutation that happens after `&&`, `;`, `|`,
# etc. We deliberately keep this to single-character control operators; bash
#isms like `$(...)` or backticks are out of scope for a best-effort guard.
_SHELL_OPERATOR_RE = re.compile(r"[;|&]+")


class ToolError(Exception):
    """Raised when a tool cannot execute or violates safety rules."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class ToolResult:
    """Result returned from a tool execution."""

    content: str
    error: bool = False
    checkpoint_id: str | None = None


@dataclass
class ToolContext:
    """Runtime dependencies made available to every tool handler."""

    root: Path
    client: httpx.AsyncClient
    extra: dict[str, Any] = field(default_factory=dict)
    checkpoint_store: Any | None = None
    run_id: str | None = None


@dataclass
class Tool:
    """Definition and handler for a single agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Any
    requires_confirmation: bool = False
    is_async: bool = False


class ToolRegistry:
    """Dynamic registry of agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, replacing any existing tool with the same name."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        requires_confirmation: bool = False,
    ) -> Any:
        """Decorator that registers a function as a tool.

        The decorated function is called as ``func(arguments, ctx)``. If the
        function is a coroutine it is awaited automatically.
        """
        def decorator(func: Any) -> Any:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=func,
                    requires_confirmation=requires_confirmation,
                    is_async=inspect.iscoroutinefunction(func),
                )
            )
            return func

        return decorator

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Return the tool definition or None if it is not registered."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return sorted(self._tools.keys())

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def requires_confirmation(self, name: str) -> bool:
        """Return True if the named tool should prompt for confirmation."""
        tool = self._tools.get(name)
        return tool.requires_confirmation if tool else False

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        """Execute a registered tool."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", error=True)

        try:
            result = tool.handler(arguments, ctx)
            if tool.is_async:
                result = await result
        except ToolError as e:
            return ToolResult(str(e.message), error=True)
        except Exception as e:
            return ToolResult(f"Error executing {name}: {e}", error=True)

        if isinstance(result, ToolResult):
            return result
        return ToolResult(str(result))


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

TOOLS = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    requires_confirmation: bool = False,
) -> Any:
    """Decorator that registers a function on the global tool registry."""
    return TOOLS.register_function(
        name=name,
        description=description,
        parameters=parameters,
        requires_confirmation=requires_confirmation,
    )


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _resolve_path(path: str, root: Path) -> Path:
    """Resolve *path* under *root*, rejecting directory traversal.

    Accepts both relative paths and absolute paths that point inside *root*.
    Uses ``os.path.commonpath`` so drive-letter differences on Windows don't
    accidentally pass the ``relative_to`` check.
    """
    root = root.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()

    try:
        common = Path(os.path.commonpath([resolved, root]))
    except ValueError as e:
        # Different drives on Windows.
        raise ToolError(f"Path escapes project root: {path}") from e
    if common != root:
        raise ToolError(f"Path escapes project root: {path}")
    return resolved


def _git_mutation_prefix(command: str) -> str | None:
    """Return the matched git mutation prefix if *command* invokes one.

    This is a guardrail, not a sandbox: it blocks the obvious and the
    common-bypass forms of the git-mutating commands in
    ``GIT_MUTATION_COMMANDS``. It is not a complete shell parser and does not
    try to defeat arbitrary metaprogramming. The goal is to stop the agent from
    quietly rewriting history via the confirmation-gated ``run_command`` tool
    even when it wraps or chains the invocation.

    The matching pipeline is:

    1. Split the line on shell control operators (``;``, ``&&``, ``||``,
       ``|``) so a mutation after a separator is caught.
    2. For each segment, strip leading ``FOO=bar`` env-var assignments (as in
       ``FOO=1 git commit``) and unwrap a leading wrapper prefix
       (``sudo``, ``env``, ``command``, ``exec``, ``time``, ``nice``,
       ``xargs``) recursively.
    3. Collapse internal whitespace and lowercase before comparing against
       the denylist, so ``git\\tcommit`` or trailing spaces cannot slip past.

    Returns the matched prefix (e.g. ``"git commit"``) for reporting, or
    ``None`` if no segment resolves to a denied command.
    """
    for segment in _split_command_segments(command):
        normalized = _normalize_segment(segment)
        if normalized is None:
            continue
        for prefix in GIT_MUTATION_COMMANDS:
            if normalized == prefix or normalized.startswith(prefix + " "):
                return prefix
    return None


def _split_command_segments(command: str) -> list[str]:
    """Split *command* into independent shell segments on control operators.

    ``&&`` and ``||`` are two-character operators but ``re.split`` on ``[;|&]+``
    treats them as a single separator class, which is what we want here: any
    run of those characters is a boundary between two commands. Quoting is not
    honoured; a literal ``;`` inside a quoted string would split too. That is
    acceptable for a best-effort guard because over-splitting only risks a
    false *match*, never a missed one -- and we only match against a denylist
    of git mutations, so a false positive on a benign command is the safe
    direction.
    """
    return _SHELL_OPERATOR_RE.split(command)


def _normalize_segment(segment: str) -> str | None:
    """Strip env-var assignments and wrapper prefixes, return the core command.

    Returns None when the segment is empty after stripping.

    Leading ``KEY=VALUE`` pairs (the form the shell treats as transient env
    assignments to the following command, e.g. ``FOO=1 git commit``) are
    dropped. Wrappers in ``_COMMAND_WRAPPERS`` are unwrapped recursively so
    ``sudo git push`` and ``env FOO=1 git commit`` resolve to ``git push`` /
    ``git commit``. We stop unwrapping at the first token that is not a known
    wrapper and not an env assignment, so a wrapper named identically to a
    real binary the user might invoke (``time git status`` -- ``time`` is both
    a shell builtin and a binary) is still treated as a wrapper here. The
    trade-off is a rare false positive on a benign wrapped command, which is
    preferable to a missed mutation.
    """
    try:
        tokens = shlex.split(segment)
    except ValueError:
        # Unbalanced quotes / parse error: fall back to a naive whitespace
        # split rather than letting the whole line through unexamined. A
        # malformed segment is more likely to be noise than a deliberate
        # bypass, and normalizing conservatively still catches the obvious
        # forms.
        tokens = segment.split()
    while tokens:
        head = tokens[0]
        # Strip a leading env-var assignment: KEY=VALUE (VALUE may be empty).
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", head):
            tokens = tokens[1:]
            continue
        # Unwrap a single wrapper token; `env` may itself carry KEY=VALUE
        # args, which the loop above will strip on the next pass.
        if head in _COMMAND_WRAPPERS:
            tokens = tokens[1:]
            continue
        break
    if not tokens:
        return None
    # Collapse all internal whitespace and lowercase so `git\tcommit` and
    # `git  commit` match `git commit` from the denylist.
    return " ".join(tokens).lower()


def _ensure_checkpoint(ctx: ToolContext) -> str | None:
    """Create a checkpoint before the first mutation in a run.

    Returns the checkpoint id, or None if checkpointing is not configured.
    """
    if ctx.checkpoint_store is None or ctx.run_id is None:
        return None
    key = "checkpoint_id"
    checkpoint_id = ctx.extra.get(key)
    if checkpoint_id:
        return checkpoint_id

    from arc_llama.agent.checkpoints import CheckpointStore

    if not isinstance(ctx.checkpoint_store, CheckpointStore):
        return None

    cp = ctx.checkpoint_store.create(ctx.run_id, ctx.root)
    ctx.extra[key] = cp.id
    return cp.id


def read_file(path: str, root: Path) -> ToolResult:
    """Read the contents of a single file."""
    try:
        target = _resolve_path(path, root)
    except ToolError as e:
        return ToolResult(str(e), error=True)
    if not target.exists():
        return ToolResult(f"Error: file not found: {path}", error=True)
    if target.is_dir():
        return ToolResult(f"Error: {path} is a directory", error=True)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(f"Error reading {path}: {e}", error=True)
    return ToolResult(text)


def apply_patch(
    path: str,
    old_string: str,
    new_string: str,
    root: Path,
    replace_all: bool = False,
) -> ToolResult:
    """Apply a surgical search/replace edit to a single file."""
    try:
        target = _resolve_path(path, root)
    except ToolError as e:
        return ToolResult(str(e), error=True)
    if not target.exists():
        return ToolResult(f"Error: file not found: {path}", error=True)
    if target.is_dir():
        return ToolResult(f"Error: {path} is a directory", error=True)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(f"Error reading {path}: {e}", error=True)

    if old_string not in text:
        return ToolResult(
            f"Error: old_string not found in {path}",
            error=True,
        )

    count = text.count(old_string)
    if not replace_all and count > 1:
        return ToolResult(
            f"Error: old_string occurs {count} times in {path}; "
            "set replace_all=true to replace all occurrences",
            error=True,
        )

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)

    try:
        target.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return ToolResult(f"Error writing {path}: {e}", error=True)

    replaced = count if replace_all else 1
    return ToolResult(f"Patched {path} ({replaced} replacement(s))")


@tool(
    name="read_file",
    description="Read the contents of a single file relative to the project root.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file."},
        },
        "required": ["path"],
    },
)
def _read_file_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return read_file(arguments.get("path", ""), ctx.root)


def read_multiple_files(paths: list[str], root: Path) -> ToolResult:
    """Read several files and return them in one result."""
    parts = []
    for p in paths:
        res = read_file(p, root)
        header = f"===== {p} ====="
        parts.append(header)
        parts.append(res.content)
    return ToolResult("\n\n".join(parts))


@tool(
    name="read_multiple_files",
    description="Read the contents of several files at once.",
    parameters={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of relative file paths.",
            },
        },
        "required": ["paths"],
    },
)
def _read_multiple_files_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return read_multiple_files(arguments.get("paths", []), ctx.root)


async def read_pdf(path: str, root: Path, client: httpx.AsyncClient) -> ToolResult:
    """Extract text from a PDF file using the /admin/parse-pdf endpoint."""
    try:
        target = _resolve_path(path, root)
    except ToolError as e:
        return ToolResult(str(e), error=True)
    if not target.exists():
        return ToolResult(f"Error: file not found: {path}", error=True)
    if not target.is_file():
        return ToolResult(f"Error: {path} is not a file", error=True)
    if target.suffix.lower() != ".pdf":
        return ToolResult(f"Error: only PDF files are supported: {path}", error=True)

    try:
        content = target.read_bytes()
        files = {"file": (target.name, io.BytesIO(content), "application/pdf")}
        response = await client.post("/admin/parse-pdf", files=files)
        response.raise_for_status()
        data = response.json()
        return ToolResult(data.get("text", ""))
    except httpx.HTTPError as e:
        return ToolResult(f"Error calling PDF parser: {e}", error=True)
    except Exception as e:
        return ToolResult(f"Error reading PDF: {e}", error=True)


@tool(
    name="read_pdf",
    description="Extract text from a PDF file relative to the project root.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the PDF file."},
        },
        "required": ["path"],
    },
)
async def _read_pdf_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return await read_pdf(arguments.get("path", ""), ctx.root, ctx.client)


def write_file(path: str, content: str, root: Path) -> ToolResult:
    """Write *content* to *path* under *root*."""
    try:
        target = _resolve_path(path, root)
    except ToolError as e:
        return ToolResult(str(e), error=True)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolResult(f"Error writing {path}: {e}", error=True)
    return ToolResult(f"Wrote {path}")


@tool(
    name="write_file",
    description="Write text to a file. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to write."},
            "content": {"type": "string", "description": "Full file contents."},
        },
        "required": ["path", "content"],
    },
    requires_confirmation=True,
)
def _write_file_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    checkpoint_id = _ensure_checkpoint(ctx)
    result = write_file(
        arguments.get("path", ""),
        arguments.get("content", ""),
        ctx.root,
    )
    return ToolResult(
        content=result.content,
        error=result.error,
        checkpoint_id=checkpoint_id,
    )


@tool(
    name="apply_patch",
    description="Apply a surgical search/replace edit to a file. "
                "old_string must match exactly once unless replace_all is true.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file."},
            "old_string": {
                "type": "string",
                "description": "Exact text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of just the first.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
    requires_confirmation=True,
)
def _apply_patch_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    checkpoint_id = _ensure_checkpoint(ctx)
    result = apply_patch(
        arguments.get("path", ""),
        arguments.get("old_string", ""),
        arguments.get("new_string", ""),
        ctx.root,
        bool(arguments.get("replace_all", False)),
    )
    return ToolResult(
        content=result.content,
        error=result.error,
        checkpoint_id=checkpoint_id,
    )


def list_directory(path: str, root: Path) -> ToolResult:
    """List the contents of a directory."""
    target = _resolve_path(path, root)
    if not target.exists():
        return ToolResult(f"Error: directory not found: {path}", error=True)
    if not target.is_dir():
        return ToolResult(f"Error: {path} is not a directory", error=True)
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return ToolResult(f"Error listing {path}: {e}", error=True)
    lines = []
    for entry in entries:
        marker = "D" if entry.is_dir() else "F"
        lines.append(f"{marker} {entry.name}")
    return ToolResult("\n".join(lines) if lines else "(empty directory)")


@tool(
    name="list_directory",
    description="List files and directories in a relative path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory path (default '.')."},
        },
    },
)
def _list_directory_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return list_directory(arguments.get("path", "."), ctx.root)


def run_command(command: str, root: Path, timeout: float = 60.0) -> ToolResult:
    """Run a shell command with *root* as the working directory."""
    bad_prefix = _git_mutation_prefix(command)
    if bad_prefix:
        return ToolResult(
            f"Error: git history-mutating command '{bad_prefix}' is not allowed.",
            error=True,
        )
    env = os.environ.copy()
    if sys.platform != "win32":
        env.update({"PS1": "", "TERM": "dumb"})
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            f"Error: command timed out after {timeout}s: {command}",
            error=True,
        )
    except OSError as e:
        return ToolResult(f"Error running command: {e}", error=True)

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        if output:
            output += "\n"
        output += f"[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code {result.returncode}]"
    return ToolResult(output.strip(), error=result.returncode != 0)


@tool(
    name="run_command",
    description="Run a shell command in the project root and return stdout/stderr.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    },
    requires_confirmation=True,
)
def _run_command_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return run_command(arguments.get("command", ""), ctx.root)


def search_files(pattern: str, root: Path, path_glob: str = "*") -> ToolResult:
    """Search file contents under *root* for *pattern*.

    Uses a simple line-by-line grep. *path_glob* filters files, e.g. '*.py'.
    """
    matches: list[str] = []
    try:
        for p in sorted(root.rglob(path_glob)):
            if not p.is_file():
                continue
            # Skip very large files and binary files.
            if p.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if pattern in line:
                            rel = p.relative_to(root).as_posix()
                            matches.append(f"{rel}:{i}: {line.rstrip()}")
            except OSError:
                continue
    except OSError as e:
        return ToolResult(f"Error searching files: {e}", error=True)
    if not matches:
        return ToolResult(f"No matches for '{pattern}'")
    return ToolResult("\n".join(matches[:200]))


@tool(
    name="search_files",
    description="Search file contents for a literal string under the project root.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text to search for."},
            "path_glob": {
                "type": "string",
                "description": "Optional glob to filter files, e.g. '*.py'.",
            },
        },
        "required": ["pattern"],
    },
)
def _search_files_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return search_files(
        arguments.get("pattern", ""),
        ctx.root,
        arguments.get("path_glob", "*"),
    )


# ---------------------------------------------------------------------------
# Chat-history tools (require chat_store in ToolContext.extra)
# ---------------------------------------------------------------------------

@tool(
    name="list_chats",
    description="List saved chat conversations ordered by most recently updated first. Optionally filter by folder.",
    parameters={
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": "Optional folder name to filter by. Omit to list all folders.",
            },
        },
    },
)
def _list_chats_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = ctx.extra.get("chat_store")
    if store is None:
        return ToolResult("Chat history is not available in this environment.", error=True)
    folder = arguments.get("folder")
    summaries = [c.summary() for c in store.list_chats(folder=folder)]
    return ToolResult(json.dumps(summaries, indent=2))


@tool(
    name="read_chat",
    description="Read the full contents of a saved chat by its id.",
    parameters={
        "type": "object",
        "properties": {
            "chat_id": {"type": "string", "description": "The id of the chat to read."},
        },
        "required": ["chat_id"],
    },
)
def _read_chat_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = ctx.extra.get("chat_store")
    if store is None:
        return ToolResult("Chat history is not available in this environment.", error=True)
    chat = store.get(arguments.get("chat_id", ""))
    if chat is None:
        return ToolResult("Chat not found.", error=True)
    return ToolResult(json.dumps(chat.to_dict(), indent=2))


@tool(
    name="search_chats",
    description="Search saved chat titles and messages for a keyword or phrase. Optionally restrict to a folder.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword or phrase to search for."},
            "limit": {
                "type": "integer",
                "description": "Maximum number of chats to return (default 20).",
            },
            "folder": {
                "type": "string",
                "description": "Optional folder name to restrict the search to.",
            },
        },
        "required": ["query"],
    },
)
def _search_chats_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = ctx.extra.get("chat_store")
    if store is None:
        return ToolResult("Chat history is not available in this environment.", error=True)
    query = arguments.get("query", "")
    limit = int(arguments.get("limit", 20))
    folder = arguments.get("folder")
    results = store.search(query, limit=limit, folder=folder)
    payload = [
        {"chat": chat.summary(), "matching_message_indices": indices}
        for chat, indices in results
    ]
    return ToolResult(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Checkpoint tools (require checkpoint_store in ToolContext)
# ---------------------------------------------------------------------------

@tool(
    name="list_checkpoints",
    description="List checkpoints for the current agent run, oldest first.",
    parameters={"type": "object", "properties": {}},
)
def _list_checkpoints_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from arc_llama.agent.checkpoints import CheckpointStore

    if ctx.checkpoint_store is None or ctx.run_id is None:
        return ToolResult("Checkpointing is not available in this environment.", error=True)
    if not isinstance(ctx.checkpoint_store, CheckpointStore):
        return ToolResult("Checkpointing is not available in this environment.", error=True)

    checkpoints = ctx.checkpoint_store.list(ctx.run_id)
    payload = [
        {
            "id": cp.id,
            "created_at": cp.created_at,
            "files": cp.files,
        }
        for cp in checkpoints
    ]
    return ToolResult(json.dumps(payload, indent=2))


@tool(
    name="restore_checkpoint",
    description="Restore the project root to a previous checkpoint.",
    parameters={
        "type": "object",
        "properties": {
            "checkpoint_id": {
                "type": "string",
                "description": "The id of the checkpoint to restore.",
            },
        },
        "required": ["checkpoint_id"],
    },
    requires_confirmation=True,
)
def _restore_checkpoint_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from arc_llama.agent.checkpoints import CheckpointStore

    if ctx.checkpoint_store is None or ctx.run_id is None:
        return ToolResult("Checkpointing is not available in this environment.", error=True)
    if not isinstance(ctx.checkpoint_store, CheckpointStore):
        return ToolResult("Checkpointing is not available in this environment.", error=True)

    checkpoint_id = arguments.get("checkpoint_id", "")
    try:
        ctx.checkpoint_store.restore(ctx.run_id, checkpoint_id, ctx.root)
    except FileNotFoundError as e:
        return ToolResult(str(e), error=True)
    except Exception as e:
        return ToolResult(f"Error restoring checkpoint: {e}", error=True)
    return ToolResult(f"Restored checkpoint {checkpoint_id}")


@tool(
    name="repo_map",
    description="Return a concise symbol-level map of the project codebase.",
    parameters={"type": "object", "properties": {}},
)
def _repo_map_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from arc_llama.agent.repo_map import build_repo_map

    return ToolResult(build_repo_map(ctx.root))


@tool(
    name="semantic_search",
    description="Search the codebase using local semantic embeddings (requires the 'semantic' extra).",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query, e.g. 'where is authentication handled?'",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results (default 5).",
            },
        },
        "required": ["query"],
    },
)
def _semantic_search_tool(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from arc_llama.agent.repo_map import SemanticIndex

    index = ctx.extra.get("semantic_index")
    if index is None:
        # Fallback to a local index directory adjacent to the project root.
        index = SemanticIndex(ctx.root / ".arc_llama_semantic_index")
        ctx.extra["semantic_index"] = index
    try:
        results = index.search(ctx.root, arguments.get("query", ""), top_k=int(arguments.get("top_k", 5)))
    except RuntimeError as e:
        return ToolResult(str(e), error=True)
    except Exception as e:
        return ToolResult(f"Semantic search failed: {e}", error=True)
    return ToolResult(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Backwards-compatible exports
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = TOOLS.definitions


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    root: Path,
    client: httpx.AsyncClient,
) -> ToolResult:
    """Dispatch a tool call to the appropriate implementation.

    This function is kept for backwards compatibility. New code should use
    ``TOOLS.execute`` with a ``ToolContext``.
    """
    ctx = ToolContext(root=root, client=client)
    return await TOOLS.execute(name, arguments, ctx)
