"""OpenCode harness adapter.

arc-llama starts a private ``opencode serve`` (loopback, random port and
password, one per run), points it at arc-llama's own OpenAI-compatible
endpoint, and translates its event stream into the vocabulary used by
``run_agent``. The harness keeps its own tools, MCP servers, skills and
``AGENTS.md``; arc-llama only supplies the model and the surface.

Verified against OpenCode 1.19.0.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("arc_llama.agent.harness.opencode")

PROVIDER_ID = "arc-llama"
LISTEN_RE = re.compile(r"listening on (https?://[^\s]+)")
STARTUP_TIMEOUT = 45.0
PERMISSION_POLL_SECONDS = 0.75
# Tools that change the machine; everything else (read, grep, ...) stays at the harness default.
GUARDED_PERMISSIONS = ("edit", "bash")
# Commands that still ask when auto-approve is on. Patterns are matched against the whole command line.
ALWAYS_ASK_COMMANDS = (
    "rm -r*", "rm -fr*", "rm -f -r*", "rm --recursive*", "rmdir *",
    "sudo *", "su *", "doas *",
    "git push*--force*", "git push -f*", "git push*+*", "git reset --hard*", "git clean -*", "git checkout -- *",
    "dd *", "mkfs*", "fdisk *", "parted *", "shred *", "truncate *",
    "chmod -R *", "chown -R *", "chmod 777*",
    "curl *| *sh*", "curl *| *bash*", "wget *| *sh*", "wget *| *bash*",
    "shutdown*", "reboot*", "poweroff*", "systemctl *stop*", "systemctl *disable*", "pkill *", "kill -9 *", "killall *",
)


# Tool-name prefixes (after "<server>_") that only look things up. In "Ask" mode these run freely;
# every other tool from an MCP server asks first. Auto-approve lets all of them run.
READ_ONLY_PREFIXES = ("list", "get", "read", "search", "query", "find", "describe", "show", "view", "status", "inspect")
WEB_TOOLS = ("webfetch", "websearch", "codesearch")
# OpenCode gates these three tools with one permission, so they can only be switched off together.
EDIT_TOOLS = ("edit", "write", "apply_patch", "multiedit")
PERMISSION_FOR_TOOL = {tool: "edit" for tool in EDIT_TOOLS}
# Tools a profile does not list individually: internal ones, and ones governed elsewhere (web, skills).
HIDDEN_TOOLS = ("invalid", "question", "skill", *WEB_TOOLS)


class OpenCodeError(RuntimeError):
    """The harness could not be started or reported a failure."""


def find_binary(configured: str | None = None) -> str | None:
    """Resolve the opencode executable: explicit path, $ARC_LLAMA_OPENCODE_BIN, then PATH."""
    for candidate in (configured, os.environ.get("ARC_LLAMA_OPENCODE_BIN")):
        if candidate:
            found = shutil.which(candidate)
            if found:
                return found
    return shutil.which("opencode")


def mcp_server_names(root: Path | str | None = None, extra_files: tuple[Path, ...] = ()) -> list[str]:
    """Names of the MCP servers the user's OpenCode setup declares (their tools are named `<server>_<tool>`).

    Reads the global config and the project's own config; values are never read, only the keys."""
    candidates: list[Path] = [Path.home() / ".config" / "opencode" / "opencode.json"]
    configured = os.environ.get("OPENCODE_CONFIG")
    if configured:
        candidates.append(Path(configured))
    if root:
        candidates += [Path(root) / "opencode.json", Path(root) / ".opencode" / "opencode.json"]
    candidates += list(extra_files)
    names: list[str] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        servers = data.get("mcp") if isinstance(data, dict) else None
        if isinstance(servers, dict):
            names += [n for n in servers if isinstance(n, str) and n and n not in names]
    return names


def build_config(
    base_url: str,
    model: str,
    *,
    auto_confirm: bool = False,
    local_only: bool = True,
    mcp_servers: tuple[str, ...] | list[str] = (),
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inline OpenCode config that routes every request through arc-llama.

    A ``profile`` ({"mcp": [...], "skills": [...], "web": bool}) is an allow-list over what the user's
    own OpenCode setup offers. Servers it leaves out are not started at all (their tools never reach the
    model), and skills and web tools it leaves out are denied."""
    allowed_servers = [n for n in mcp_servers if n in set(profile.get("mcp") or [])] if profile else list(mcp_servers)
    action = "allow" if auto_confirm else "ask"
    # With auto-approve, routine edits and commands run unattended but destructive commands still ask.
    # OpenCode applies the last matching rule, so the catch-all comes first.
    bash_rule: Any = {"*": "allow", **{pattern: "ask" for pattern in ALWAYS_ASK_COMMANDS}} if auto_confirm else "ask"
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{PROVIDER_ID}/{model}",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "arc-llama",
                "options": {"baseURL": base_url.rstrip("/") + "/v1", "apiKey": "sk-no-key"},
                "models": {model: {"name": model}},
            }
        },
        "permission": _permissions(action, bash_rule, auto_confirm, allowed_servers),
        "share": "disabled",
        "autoupdate": False,
    }
    if profile:
        rules = config["permission"]
        config["mcp"] = {name: {"enabled": name in allowed_servers} for name in mcp_servers}
        for name in mcp_servers:
            if name not in allowed_servers:
                rules[f"{name}_*"] = "deny"
        rules["skill"] = {"*": "deny", **{name: "allow" for name in profile.get("skills") or []}}
        for tool in profile.get("tools_off") or []:
            rules[PERMISSION_FOR_TOOL.get(tool, tool)] = "deny"
        if not profile.get("web"):
            for tool in WEB_TOOLS:
                rules[tool] = "deny"
    if local_only:
        # Without this a session can still reach any cloud provider in the user's own config.
        config["enabled_providers"] = [PROVIDER_ID]
    return config


def _permissions(action: str, bash_rule: Any, auto_confirm: bool, mcp_servers: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Edit/bash as configured, plus what to do about tools inherited from the user's OpenCode setup.

    Order matters: OpenCode applies the last matching rule, so each server's blanket "ask" comes before
    the read-only allowances. With auto-approve the inherited tools are simply left allowed."""
    rules: dict[str, Any] = {"edit": action, "bash": bash_rule}
    if auto_confirm:
        return rules
    for tool in WEB_TOOLS:
        rules[tool] = "ask"
    for server in mcp_servers:
        rules[f"{server}_*"] = "ask"
        for prefix in READ_ONLY_PREFIXES:
            rules[f"{server}_{prefix}*"] = "allow"
    return rules


class OpenCodeTranslator:
    """Stateful, I/O-free mapping from OpenCode events to arc-llama agent events."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self._roles: dict[str, str] = {}
        self._called: set[str] = set()
        self._finished: set[str] = set()
        self._said: set[str] = set()
        self._usage: dict[str, tuple[int, int]] = {}
        self._activity: str | None = None
        self._part_types: dict[str, str] = {}
        self._busy = False
        self.done = False

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        props = event.get("properties") or {}
        sid = props.get("sessionID") or (props.get("info") or {}).get("id")
        if self.session_id and sid and sid != self.session_id and event.get("type", "").startswith(
            ("message.", "session.")
        ):
            return []
        kind = event.get("type", "")
        if kind == "message.updated":
            info = props.get("info") or {}
            if info.get("id") and info.get("role"):
                self._roles[info["id"]] = info["role"]
            tokens = info.get("tokens")
            if info.get("role") == "assistant" and isinstance(tokens, dict):
                cache_value = tokens.get("cache")
                cache = cache_value if isinstance(cache_value, dict) else {}
                counts = [tokens.get("input"), tokens.get("output"),
                          cache.get("read", 0), cache.get("write", 0)]
                if all(type(n) is int and n >= 0 for n in counts) and any(counts):
                    prompt, completion = counts[0] + counts[2] + counts[3], counts[1]
                    mid = str(info.get("id") or "")
                    if self._usage.get(mid) != (prompt, completion):
                        self._usage[mid] = (prompt, completion)
                        return [{"type": "usage", "prompt_tokens": prompt,
                                 "completion_tokens": completion}]
            return []
        if kind == "message.part.delta":
            part_type = self._part_types.get(str(props.get("partID") or ""))
            if part_type in ("reasoning", "text"):
                return self._phase("thinking" if part_type == "reasoning" else "answering")
            return []
        if kind == "message.part.updated":
            return self._part(props.get("part") or {})
        if kind == "session.status":
            status = (props.get("status") or {}).get("type")
            if status == "busy" and not self._busy:
                self._busy = True
                return [{"type": "status", "message": "Working…"}]
            if status == "idle" and self._busy:
                return self._finish()
            return []
        if kind == "session.idle":
            return self._finish() if self._busy else []
        if kind == "session.error":
            err = props.get("error") or {}
            msg = (err.get("data") or {}).get("message") or err.get("name") or "OpenCode reported an error"
            self.done = True
            return [{"type": "error", "message": str(msg)}]
        return []

    def _finish(self) -> list[dict[str, Any]]:
        if self.done:
            return []
        self.done = True
        return [{"type": "done"}]

    def _phase(self, phase: str) -> list[dict[str, Any]]:
        if self._activity == phase:
            return []
        self._activity = phase
        return [{"type": "activity", "phase": phase}]

    def _part(self, part: dict[str, Any]) -> list[dict[str, Any]]:
        ptype, mid = part.get("type"), str(part.get("messageID") or "")
        if ptype in ("reasoning", "text") and self._roles.get(mid) == "assistant":
            self._part_types[str(part.get("id") or "")] = ptype
        if ptype == "reasoning" and self._roles.get(mid) == "assistant":
            return self._phase("thinking") if not (part.get("time") or {}).get("end") else []
        if ptype == "tool":
            return self._tool(part)
        if ptype == "text" and self._roles.get(mid) == "assistant":
            pid = str(part.get("id") or "")
            if (part.get("time") or {}).get("end") and pid not in self._said and part.get("text"):
                self._said.add(pid)
                return [*self._phase("answering"), {"type": "assistant", "content": part["text"]}]
            if not (part.get("time") or {}).get("end"):
                return self._phase("answering")
        return []

    def _tool(self, part: dict[str, Any]) -> list[dict[str, Any]]:
        call_id, name = str(part.get("callID") or part.get("id") or ""), part.get("tool") or "tool"
        state = part.get("state") or {}
        status, out = state.get("status"), []
        if status in ("running", "completed", "error") and call_id not in self._called:
            self._called.add(call_id)
            out.append({"type": "tool_call", "id": call_id, "name": name, "arguments": state.get("input") or {}})
        if status in ("completed", "error") and call_id not in self._finished:
            self._finished.add(call_id)
            content = state.get("output") if status == "completed" else state.get("error")
            out.append({"type": "tool_result", "id": call_id, "name": name,
                        "content": str(content if content is not None else ""), "error": status == "error"})
        return out


class _Server:
    """One private ``opencode serve`` process."""

    def __init__(self, binary: str, root: Path, env: dict[str, str]) -> None:
        self.binary, self.root, self.env = binary, root, env
        self.proc: asyncio.subprocess.Process | None = None
        self.url = ""

    async def start(self) -> str:
        try:
            self.proc = await asyncio.create_subprocess_exec(
                self.binary, "serve", "--hostname", "127.0.0.1", "--port", "0",
                cwd=str(self.root), env=self.env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            what = "The project folder does not exist" if exc.filename == str(self.root) else "OpenCode could not be started"
            raise OpenCodeError(f"{what}: {exc.filename or self.root}") from None
        except OSError as exc:
            raise OpenCodeError(f"OpenCode could not be started: {exc.strerror or exc}") from None
        assert self.proc.stdout is not None
        try:
            self.url = await asyncio.wait_for(self._read_url(self.proc.stdout), STARTUP_TIMEOUT)
        except (asyncio.TimeoutError, OpenCodeError):
            await self.stop()
            raise OpenCodeError("opencode did not report a listening address") from None
        return self.url

    async def _read_url(self, stream: asyncio.StreamReader) -> str:
        while True:
            line = await stream.readline()
            if not line:
                raise OpenCodeError("opencode exited during startup")
            match = LISTEN_RE.search(line.decode(errors="replace"))
            if match:
                # Keep draining so the child never blocks on a full pipe.
                asyncio.get_running_loop().create_task(self._drain(stream))
                return match.group(1).rstrip("/")

    @staticmethod
    async def _drain(stream: asyncio.StreamReader) -> None:
        with contextlib.suppress(Exception):
            while await stream.read(4096):
                pass

    def terminate_now(self) -> None:
        """Ask the process to exit without awaiting anything, so a cancelled task cannot skip it."""
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_later(5.0, self._kill_if_alive)

    def _kill_if_alive(self) -> None:
        proc = self.proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def stop(self) -> None:
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()


async def run_opencode(
    task: str,
    model: str,
    base_url: str,
    root: Path,
    *,
    auto_confirm: bool = False,
    confirm_callback: Callable[[str, str, dict[str, Any]], Awaitable[bool]] | None = None,
    local_only: bool = True,
    timeout: float = 600.0,
    binary: str | None = None,
    command: str | None = None,
    arguments: str = "",
    tool_profile: dict[str, Any] | None = None,
    session_title: str | None = None,
    **_ignored: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Drive OpenCode for one task, yielding ``run_agent``-shaped events.

    With ``command`` set, the harness's own slash command of that name runs (with ``arguments``)
    instead of a plain prompt, so user commands, MCP prompts and skills behave as they do in OpenCode."""
    exe = find_binary(binary)
    if exe is None:
        yield {"type": "error", "message": "opencode was not found. Install OpenCode or set ARC_LLAMA_OPENCODE_BIN."}
        return
    if not auto_confirm and confirm_callback is None:
        yield {"type": "error", "message": "confirmation callback required when auto_confirm is false"}
        return

    from arc_llama.agent.harness.inference import InferenceObserver

    root = Path(root).resolve()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    inference = InferenceObserver(base_url, task, queue)
    password = secrets.token_urlsafe(24)
    env = {
        **os.environ,
        "OPENCODE_SERVER_PASSWORD": password,
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            build_config(inference.url, model, auto_confirm=auto_confirm, local_only=local_only,
                         mcp_servers=mcp_server_names(root), profile=tool_profile)
        ),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    server = _Server(exe, root, env)
    yield {"type": "status", "message": "Starting OpenCode…"}
    try:
        url = await server.start()
    except OpenCodeError as exc:
        await inference.close()
        yield {"type": "error", "message": str(exc)}
        return
    except BaseException:
        server.terminate_now()
        await asyncio.shield(inference.close())
        raise

    params = {"directory": str(root)}
    auth = httpx.BasicAuth("opencode", password)
    pump: asyncio.Task[None] | None = None
    poller: asyncio.Task[None] | None = None
    command_task: asyncio.Task[None] | None = None
    try:
        async with httpx.AsyncClient(base_url=url, auth=auth, timeout=30.0, params=params) as client:
            async def pump_events() -> None:
                try:
                    async with client.stream("GET", "/event", timeout=None) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data:"):
                                with contextlib.suppress(ValueError):
                                    await queue.put(json.loads(line[5:]))
                except (httpx.HTTPError, asyncio.CancelledError):
                    pass
                finally:
                    await queue.put(None)

            pump = asyncio.create_task(pump_events())
            created = await client.post("/session", json={"title": (session_title or task).strip().split("\n")[0][:80] or "arc-llama"})
            created.raise_for_status()
            session_id = created.json()["id"]
            yield {"type": "harness_session", "session_id": "oc-" + session_id}
            translator = OpenCodeTranslator(session_id)

            handled: set[str] = set()      # permission ids already put to the user (events and polling can both report one)

            async def poll_permissions() -> None:
                seen: set[str] = set()
                while True:
                    await asyncio.sleep(PERMISSION_POLL_SECONDS)
                    try:
                        pending = (await client.get("/permission")).json()
                    except (httpx.HTTPError, ValueError):
                        continue
                    for req in pending if isinstance(pending, list) else []:
                        if req.get("sessionID") != session_id or req["id"] in seen:
                            continue
                        seen.add(req["id"])
                        await queue.put({"type": "_permission", "request": req})

            poller = asyncio.create_task(poll_permissions())
            if command:
                # OpenCode answers this request only when the command has finished, so run it alongside
                # the event stream and only look at the result if it failed to start.
                async def run_command() -> None:
                    try:
                        sent = await client.post(
                            f"/session/{session_id}/command",
                            json={"command": command, "arguments": arguments, "model": f"{PROVIDER_ID}/{model}"},
                            timeout=None,
                        )
                        if sent.status_code >= 400:
                            detail = " ".join(sent.text.split())[:300]
                            await queue.put({"type": "_failed", "message": f"OpenCode could not run /{command} (HTTP {sent.status_code}){': ' + detail if detail else ''}."})
                    except httpx.HTTPError as exc:
                        await queue.put({"type": "_failed", "message": f"OpenCode could not run /{command}: {exc}"})

                command_task = asyncio.create_task(run_command())
            else:
                sent = await client.post(
                    f"/session/{session_id}/prompt_async",
                    json={
                        "model": {"providerID": PROVIDER_ID, "modelID": model},
                        "parts": [{"type": "text", "text": task}],
                    },
                )
                sent.raise_for_status()

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while not translator.done:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    yield {"type": "error", "message": f"OpenCode run exceeded {timeout:.0f}s"}
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    if not translator.done:
                        yield {"type": "error", "message": "OpenCode event stream closed unexpectedly"}
                    return
                if event.get("type") == "_inference":
                    observed = event["event"]
                    if observed.get("type") == "activity":
                        for phase in translator._phase(observed["phase"]):
                            yield phase
                    else:
                        yield observed
                    continue
                if event.get("type") == "_failed":
                    yield {"type": "error", "message": event["message"]}
                    return
                if event.get("type") == "permission.asked":
                    # The stream is the reliable source: OpenCode's list endpoint can answer 400 for a
                    # pending web fetch (its own response validation fails), so polling alone can hang.
                    req = event.get("properties") or {}
                    if req.get("sessionID") == session_id and req.get("id"):
                        event = {"type": "_permission", "request": req}
                if event.get("type") == "_permission":
                    req = event["request"]
                    if req.get("id") in handled:
                        continue
                    handled.add(req["id"])
                    ask_arguments = {"patterns": req.get("patterns") or [], "metadata": req.get("metadata") or {}}
                    tool = str(req.get("permission") or "tool")
                    yield {"type": "confirm_required", "id": req["id"], "tool": tool, "arguments": ask_arguments}
                    # Whatever the harness still asks about is a decision for the user. Auto-approve only
                    # changes what the harness is configured to ask about (see build_config); it never
                    # answers a question the harness did raise.
                    approved = bool(await confirm_callback(req["id"], tool, ask_arguments)) if confirm_callback else auto_confirm
                    await client.post(
                        f"/permission/{req['id']}/reply",
                        json={"reply": "once" if approved else "reject"},
                    )
                    continue
                for out in translator.feed(event):
                    yield out
            with contextlib.suppress(httpx.HTTPError):
                await client.post(f"/session/{session_id}/abort")
    except httpx.HTTPError as exc:
        yield {"type": "error", "message": f"OpenCode request failed: {exc}"}
    finally:
        # A browser that disconnects cancels this task, and every await below is then cancelled
        # again. Ask the process to exit synchronously first so it can never be left running.
        server.terminate_now()
        background = [t for t in (poller, pump, command_task) if t is not None]
        for task_ in background:
            task_.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.shield(asyncio.gather(*background, return_exceptions=True))
            await asyncio.shield(server.stop())
        with contextlib.suppress(BaseException):
            await asyncio.shield(inference.close())


_COMMAND_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
COMMAND_CACHE_SECONDS = 60.0


async def list_commands(root: Path | str, model: str = "arc-llama", *, binary: str | None = None) -> list[dict[str, Any]]:
    """The slash commands OpenCode would offer for a project: user commands, MCP prompts and skills.

    Starts a short-lived private server, so results are cached briefly per folder."""
    exe = find_binary(binary)
    if exe is None:
        raise OpenCodeError("opencode was not found. Install OpenCode or set ARC_LLAMA_OPENCODE_BIN.")
    root = Path(root).resolve()
    now = asyncio.get_running_loop().time()
    cached = _COMMAND_CACHE.get(str(root))
    if cached and now - cached[0] < COMMAND_CACHE_SECONDS:
        return cached[1]
    password = secrets.token_urlsafe(24)
    env = {
        **os.environ,
        "OPENCODE_SERVER_PASSWORD": password,
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            build_config("http://127.0.0.1:1/v1", model, auto_confirm=False, local_only=True, mcp_servers=mcp_server_names(root))
        ),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    server = _Server(exe, root, env)
    url = await server.start()
    try:
        async with httpx.AsyncClient(base_url=url, auth=httpx.BasicAuth("opencode", password), timeout=30.0,
                                     params={"directory": str(root)}) as client:
            response = await client.get("/command")
            response.raise_for_status()
            found = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OpenCodeError(f"Could not list OpenCode commands: {exc}") from exc
    finally:
        server.terminate_now()
        with contextlib.suppress(BaseException):
            await asyncio.shield(server.stop())
    commands = [
        {
            "name": str(item["name"]),
            "description": str(item.get("description") or ""),
            "source": str(item.get("source") or "command"),
            "takes_arguments": bool(item.get("hints")),
        }
        for item in found if isinstance(item, dict) and item.get("name")
    ]
    _COMMAND_CACHE[str(root)] = (now, commands)
    return commands


def filter_commands(commands: list[dict[str, Any]], profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Hide the commands a profile switches off: skills it does not list and prompts of servers it leaves out."""
    if not profile:
        return commands
    skills, servers = set(profile.get("skills") or []), set(profile.get("mcp") or [])
    return [
        c for c in commands
        if not (c["source"] == "skill" and c["name"] not in skills)
        and not (c["source"] == "mcp" and c["name"].split(":", 1)[0] not in servers)
    ]


_TOOL_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
TOOL_CACHE_SECONDS = 300.0


async def probe_tools(root: Path | str, config: dict[str, Any], *, binary: str | None = None, timeout: float = 90.0) -> list[str]:
    """The tool names OpenCode would offer the model under ``config``, captured from a real request.

    Runs a private OpenCode against a throwaway local endpoint that answers one chat request and
    records the tools it was sent. It is the only way to see MCP and plugin tools together, and
    what permission rules really remove, because OpenCode lists neither through its own API."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    exe = find_binary(binary)
    if exe is None:
        raise OpenCodeError("opencode was not found. Install OpenCode or set ARC_LLAMA_OPENCODE_BIN.")
    root = Path(root).resolve()
    captured: list[list[str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if body.get("tools"):
                captured.append(sorted(t["function"]["name"] for t in body["tools"] if isinstance(t, dict) and "function" in t))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ):
                self.wfile.write(f"data: {json.dumps({'id': 'probe', 'object': 'chat.completion.chunk', 'model': body.get('model'), **chunk})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    listener = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=listener.serve_forever, daemon=True)
    thread.start()
    password = secrets.token_urlsafe(24)
    cfg = json.loads(json.dumps(config))
    cfg["provider"][PROVIDER_ID]["options"]["baseURL"] = f"http://127.0.0.1:{listener.server_address[1]}/v1"
    env = {
        **os.environ,
        "OPENCODE_SERVER_PASSWORD": password,
        "OPENCODE_CONFIG_CONTENT": json.dumps(cfg),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    server = _Server(exe, root, env)
    model = cfg["model"].split("/", 1)[1]
    try:
        url = await server.start()
        async with httpx.AsyncClient(base_url=url, auth=httpx.BasicAuth("opencode", password), timeout=30.0,
                                     params={"directory": str(root)}) as client:
            created = await client.post("/session", json={"title": "arc-llama tool scan"})
            created.raise_for_status()
            session_id = created.json()["id"]
            try:
                (await client.post(f"/session/{session_id}/prompt_async", json={
                    "model": {"providerID": PROVIDER_ID, "modelID": model}, "parts": [{"type": "text", "text": "hi"}]})).raise_for_status()
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout
                while not captured and loop.time() < deadline:
                    await asyncio.sleep(0.5)
            finally:
                with contextlib.suppress(httpx.HTTPError):
                    await client.delete(f"/session/{session_id}")        # leave no trace in the user's history
    except (httpx.HTTPError, OpenCodeError) as exc:
        raise OpenCodeError(f"Could not read OpenCode's tools: {exc}") from exc
    finally:
        listener.shutdown()
        server.terminate_now()
        with contextlib.suppress(BaseException):
            await asyncio.shield(server.stop())
    if not captured:
        raise OpenCodeError("OpenCode did not report its tools in time.")
    return captured[-1]


def group_tools(names: list[str], servers: list[str]) -> list[dict[str, str]]:
    """Label each tool as built-in, from one MCP server, or from a plugin."""
    builtin = {"bash", "read", "glob", "grep", "list", "task", "todowrite", "todoread", "lsp", "patch",
               "list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource", *EDIT_TOOLS, *HIDDEN_TOOLS}
    out = []
    for name in names:
        owner = max((s for s in servers if name.startswith(s + "_")), key=len, default=None)
        out.append({"id": name, "group": owner or ("builtin" if name in builtin else "plugin")})
    return out


async def list_tools(root: Path | str, model: str = "arc-llama", *, binary: str | None = None) -> list[dict[str, str]]:
    """Every tool the user's OpenCode offers for a project, grouped. Cached briefly because it starts OpenCode."""
    root = Path(root).resolve()
    now = asyncio.get_running_loop().time()
    cached = _TOOL_CACHE.get(str(root))
    if cached and now - cached[0] < TOOL_CACHE_SECONDS:
        return cached[1]
    servers = mcp_server_names(root)
    names = await probe_tools(root, build_config("http://127.0.0.1:1", model, auto_confirm=False, local_only=True, mcp_servers=servers), binary=binary)
    tools = group_tools(names, servers)
    _TOOL_CACHE[str(root)] = (now, tools)
    return tools


async def list_capabilities(root: Path | str, model: str = "arc-llama", *, binary: str | None = None) -> dict[str, Any]:
    """What a profile can switch on or off for a project: MCP servers, skills and the web tools."""
    commands = await list_commands(root, model, binary=binary)
    warnings: list[str] = []
    try:
        tools = [t for t in await list_tools(root, model, binary=binary) if t["id"] not in HIDDEN_TOOLS]
    except OpenCodeError as exc:
        log.warning("Could not list tools for profiles: %s", exc)
        warnings.append(str(exc))
        tools = []
    return {
        "mcp": mcp_server_names(Path(root).resolve()),
        "skills": [{"name": c["name"], "description": c["description"]} for c in commands if c["source"] == "skill"],
        "web": list(WEB_TOOLS),
        "tools": tools,
        "tools_complete": not warnings,
        "warnings": warnings,
    }
