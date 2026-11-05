"""OpenCode harness adapter.

arc-llama starts a private ``opencode serve`` (loopback, random port and
password, one per run), points it at arc-llama's own OpenAI-compatible
endpoint, and translates its event stream into the vocabulary used by
``run_agent``. The harness keeps its own tools, MCP servers, skills and
``AGENTS.md``; arc-llama only supplies the model and the surface.

OpenCode 2 replaced the 1.x HTTP API and event stream, so each server is asked which one it speaks
(``_V1Api`` / ``_V2Api``) and runs are translated by the matching translator. The inline config and
its permission rules are read the same way by both. Verified against OpenCode 1.19.5 and 2.2.5.
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
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from arc_llama.agent.questions import clean_questions

log = logging.getLogger("arc_llama.agent.harness.opencode")

PROVIDER_ID = "arc-llama"
LISTEN_RE = re.compile(r"listening on (https?://[^\s]+)")
STARTUP_TIMEOUT = 45.0
PERMISSION_POLL_SECONDS = 0.75
# OpenCode 2 loads commands in the background after it starts listening; wait this long for them.
COMMAND_LOAD_SECONDS = 10.0
# OpenCode 2 names permissions after its own tools; Work's approval sheet speaks the 1.x names.
V2_PERMISSION_NAMES = {"shell": "bash"}
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


# Comments, trailing commas and the strings that may hold either; OpenCode reads every config file as JSONC.
_JSONC_NOISE = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])', re.S)


def _jsonc(text: str) -> Any:
    for _ in range(2):          # a comma before a comment only trails once the comment is gone
        text = _JSONC_NOISE.sub(lambda m: m.group(1) or "", text)
    return json.loads(text)


def _config_files(root: Path | str | None) -> list[Path]:
    """Every file OpenCode reads its config from, as OpenCode 2 finds them (1.x reads a subset)."""
    names = ("opencode.json", "opencode.jsonc")
    if sys.platform == "darwin":
        system = Path("/Library/Application Support/opencode")
    elif sys.platform == "win32":
        system = Path(os.environ.get("ProgramData") or "C:\\ProgramData") / "opencode"
    else:
        system = Path("/etc/opencode")
    folders = [system, Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"]
    if os.environ.get("OPENCODE_CONFIG_DIR"):
        folders.append(Path(os.environ["OPENCODE_CONFIG_DIR"]))
    if root:
        # Project files are found in the folder and every folder above it, each also in its .opencode folder.
        for folder in reversed([Path(root), *Path(root).parents]):
            folders += [folder, folder / ".opencode"]
    files = [folder / name for folder in folders for name in names]
    if os.environ.get("OPENCODE_CONFIG"):
        files.append(Path(os.environ["OPENCODE_CONFIG"]))
    return files


def mcp_server_names(root: Path | str | None = None, extra_files: tuple[Path, ...] = ()) -> list[str]:
    """Names of the MCP servers the user's OpenCode setup declares (their tools are named `<server>_<tool>`).

    Reads every config file OpenCode would (see ``_config_files``); values are never read, only the keys."""
    names: list[str] = []
    for path in [*_config_files(root), *extra_files]:
        try:
            data = _jsonc(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        servers = data.get("mcp") if isinstance(data, dict) else None
        if isinstance(servers, dict):
            # OpenCode 2 nests servers under "servers" beside its own settings; 1.x lists them directly.
            if isinstance(servers.get("servers"), dict):
                servers = servers["servers"]
            else:
                servers = {k: v for k, v in servers.items() if k != "timeout" and isinstance(v, dict)}
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
    questions: bool = False,
) -> dict[str, Any]:
    """Inline OpenCode config that routes every request through arc-llama.

    ``questions`` keeps OpenCode's ``question`` tool; without it the tool is denied, because a question
    nobody relays to the user leaves the run waiting until it times out.

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
        # OpenCode 2 drops the enabled-only entries above; its policies keep a left-out server from
        # starting at all. OpenCode 1.x ignores the key.
        config["policies"] = [{"action": "mcp.use", "resource": name, "effect": "deny"}
                              for name in mcp_servers if name not in allowed_servers]
        rules["skill"] = {"*": "deny", **{name: "allow" for name in profile.get("skills") or []}}
        for tool in profile.get("tools_off") or []:
            rules[PERMISSION_FOR_TOOL.get(tool, tool)] = "deny"
        if not profile.get("web"):
            for tool in WEB_TOOLS:
                rules[tool] = "deny"
    if not questions:
        config["permission"]["question"] = "deny"
    # OpenCode 2 hides MCP and plugin tools inside one "execute" tool (Code Mode) unless that tool is
    # denied; then it offers them directly, where profiles and approvals can see them. 1.x has no such tool.
    config["permission"]["execute"] = "deny"
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
        self._streamed: dict[str, str] = {}
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
                out = self._phase("thinking" if part_type == "reasoning" else "answering")
                text = props.get("delta")
                if (part_type == "text" and isinstance(text, str) and text
                        and str(props.get("partID") or "") not in self._said):
                    pid = str(props.get("partID") or "")
                    self._streamed[pid] = self._streamed.get(pid, "") + text
                    out.append({"type": "assistant_delta", "content": text})
                return out
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
                return self._settle(pid, part["text"])
            if not (part.get("time") or {}).get("end"):
                return self._phase("answering")
        return []

    def _settle(self, key: str, full: str) -> list[dict[str, Any]]:
        """The completed text of one answer part, after its words may already have streamed."""
        streamed = self._streamed.pop(key, "")
        if streamed:
            if not full.startswith(streamed):
                if streamed.rstrip() == full.rstrip():
                    # The final text dropped trailing whitespace already shown.
                    return self._phase("answering")
                return [{"type": "error", "message": "OpenCode's completed answer differs from its streamed text; retry the request."}]
            # The words already reached the page as deltas; send only what was missed.
            tail = full[len(streamed):]
            return [*self._phase("answering"), *([{"type": "assistant_delta", "content": tail}] if tail else [])]
        return [*self._phase("answering"), {"type": "assistant", "content": full}]

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


class OpenCodeV2Translator(OpenCodeTranslator):
    """The same mapping for OpenCode 2, whose events are ``{"type", "data"}`` and name each step.

    Text and reasoning arrive as ``session.text.*`` / ``session.reasoning.*`` per assistant message
    and ordinal, tools as ``session.tool.*`` by call id, and the run ends with ``session.execution.*``."""

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__(session_id)
        self._tool_names: dict[str, str] = {}

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        kind, data = str(event.get("type") or ""), event.get("data")
        if not isinstance(data, dict):
            return []
        sid = data.get("sessionID")
        if self.session_id and sid and sid != self.session_id:
            return []
        key = f"{data.get('assistantMessageID')}:{data.get('ordinal')}"
        if kind in ("session.reasoning.started", "session.reasoning.delta"):
            return self._phase("thinking")
        if kind == "session.text.started":
            return self._phase("answering")
        if kind == "session.text.delta":
            out = self._phase("answering")
            text = data.get("delta")
            if isinstance(text, str) and text and key not in self._said:
                self._streamed[key] = self._streamed.get(key, "") + text
                out.append({"type": "assistant_delta", "content": text})
            return out
        if kind == "session.text.ended":
            text = data.get("text")
            if isinstance(text, str) and text and key not in self._said:
                self._said.add(key)
                return self._settle(key, text)
            return []
        if kind == "session.tool.input.started":
            self._tool_names[str(data.get("id") or "")] = str(data.get("name") or "tool")
            return []
        if kind in ("session.tool.called", "session.tool.success", "session.tool.failed"):
            return self._tool_event(kind, data)
        if kind == "session.step.ended":
            tokens = data.get("tokens")
            cache_value = tokens.get("cache") if isinstance(tokens, dict) else None
            cache = cache_value if isinstance(cache_value, dict) else {}
            if isinstance(tokens, dict):
                counts = [tokens.get("input"), tokens.get("output"), cache.get("read", 0), cache.get("write", 0)]
                if all(type(n) is int and n >= 0 for n in counts) and any(counts):
                    prompt, completion = counts[0] + counts[2] + counts[3], counts[1]
                    mid = str(data.get("assistantMessageID") or "")
                    if self._usage.get(mid) != (prompt, completion):
                        self._usage[mid] = (prompt, completion)
                        return [{"type": "usage", "prompt_tokens": prompt, "completion_tokens": completion}]
            return []
        if kind == "session.execution.started" and not self._busy:
            self._busy = True
            return [{"type": "status", "message": "Working…"}]
        if kind in ("session.execution.succeeded", "session.execution.interrupted"):
            return self._finish()
        if kind == "session.execution.failed":
            err = _obj(data.get("error"))
            self.done = True
            return [{"type": "error", "message": str(err.get("message") or "OpenCode reported an error")}]
        if kind == "session.retry.scheduled":
            # Without this the run looks frozen while OpenCode waits out its backoff.
            err = _obj(data.get("error"))
            attempt = data.get("attempt")
            reason = " ".join(str(err.get("message") or "").split())[:160]
            return [{"type": "status", "message": "The model request failed; OpenCode is retrying"
                     + (f" (attempt {attempt})" if type(attempt) is int else "") + (f": {reason}" if reason else ".")}]
        if kind == "session.compaction.started":
            return [{"type": "status", "message": "OpenCode is summarizing the conversation to make room…"}]
        return []

    def _tool_event(self, kind: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        call_id = str(data.get("id") or "")
        name = self._tool_names.get(call_id) or str(data.get("name") or "tool")
        out: list[dict[str, Any]] = []
        if call_id not in self._called:
            self._called.add(call_id)
            arguments = data.get("input") if kind == "session.tool.called" else None
            out.append({"type": "tool_call", "id": call_id, "name": name,
                        "arguments": arguments if isinstance(arguments, dict) else {}})
        if kind != "session.tool.called" and call_id not in self._finished:
            self._finished.add(call_id)
            if kind == "session.tool.success":
                content = data.get("content")
                text = "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text") \
                    if isinstance(content, list) else str(content or "")
                out.append({"type": "tool_result", "id": call_id, "name": name, "content": text, "error": False})
            else:
                err = data.get("error")
                message = err.get("message") if isinstance(err, dict) else err
                out.append({"type": "tool_result", "id": call_id, "name": name, "content": str(message or ""), "error": True})
        return out


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _v2_permission(req: dict[str, Any]) -> dict[str, Any]:
    """An OpenCode 2 permission request in the 1.x shape the Work view reads."""
    action = str(req.get("action") or "tool")
    return {"id": req.get("id"), "sessionID": req.get("sessionID"),
            "permission": V2_PERMISSION_NAMES.get(action, action),
            "patterns": [str(r) for r in req.get("resources") or []],
            "metadata": _obj(req.get("metadata")),
            "tool": {"callID": str(_obj(req.get("source")).get("id") or "")}}


def _v2_question(form: dict[str, Any]) -> dict[str, Any] | None:
    """An OpenCode 2 question form in the 1.x question shape, or None for any other kind of form.

    Each question is one field: its title is the header, its description the question."""
    metadata = _obj(form.get("metadata"))
    fields = form.get("fields")
    if metadata.get("kind") != "question" or not isinstance(fields, list):
        return None
    questions, keys, multiple = [], [], []
    for field in fields:
        if not isinstance(field, dict) or not field.get("key"):
            continue
        keys.append(str(field["key"]))
        multiple.append(field.get("type") == "multiselect")
        questions.append({
            "question": field.get("description") or field.get("title") or "",
            "header": field.get("title") or "",
            "options": [{"label": o.get("label") or o.get("value"), "description": o.get("description") or ""}
                        for o in field.get("options") or [] if isinstance(o, dict)],
            "multiple": field.get("type") == "multiselect",
            "custom": field.get("custom", True),
        })
    tool = _obj(metadata.get("tool"))
    return {"id": form.get("id"), "sessionID": form.get("sessionID"), "questions": questions,
            "tool": {"callID": str(tool.get("id") or "")}, "keys": keys, "multiple": multiple}


class _V1Api:
    """The OpenCode 1.x HTTP API: ``/session``, ``/event``, ``/permission`` and ``/question``."""

    events_path = "/event"

    def __init__(self, client: httpx.AsyncClient, root: Path) -> None:
        self.client = client
        client.params = httpx.QueryParams({"directory": str(root)})

    def translator(self, session_id: str) -> OpenCodeTranslator:
        return OpenCodeTranslator(session_id)

    async def create_session(self, title: str, model: str) -> str:
        created = await self.client.post("/session", json={"title": title})
        created.raise_for_status()
        return str(created.json()["id"])

    async def prompt(self, session_id: str, text: str, model: str) -> None:
        sent = await self.client.post(f"/session/{session_id}/prompt_async", json={
            "model": {"providerID": PROVIDER_ID, "modelID": model}, "parts": [{"type": "text", "text": text}]})
        sent.raise_for_status()

    async def command(self, session_id: str, name: str, arguments: str, model: str) -> httpx.Response:
        # OpenCode answers this request only when the command has finished.
        return await self.client.post(f"/session/{session_id}/command", timeout=None,
                                      json={"command": name, "arguments": arguments, "model": f"{PROVIDER_ID}/{model}"})

    def inline(self, event: dict[str, Any], session_id: str) -> dict[str, Any] | None:
        """A permission or question request carried on the event stream, ready for the run loop."""
        req = event.get("properties") or {}
        if not isinstance(req, dict) or req.get("sessionID") != session_id or not req.get("id"):
            return None
        if event.get("type") == "permission.asked":
            return {"type": "_permission", "request": req}
        if event.get("type") == "question.asked":
            return {"type": "_question", "request": req}
        return None

    async def pending(self, session_id: str, questions: bool) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for path, kind in [("/permission", "_permission")] + ([("/question", "_question")] if questions else []):
            try:
                listed = (await self.client.get(path)).json()
            except (httpx.HTTPError, ValueError):
                continue
            for req in listed if isinstance(listed, list) else []:
                if isinstance(req, dict) and req.get("sessionID") == session_id and req.get("id"):
                    found.append({"type": kind, "request": req})
        return found

    async def reply_permission(self, session_id: str, req: dict[str, Any], approved: bool) -> None:
        await self.client.post(f"/permission/{req['id']}/reply", json={"reply": "once" if approved else "reject"})

    async def reply_question(self, session_id: str, req: dict[str, Any], answers: list[list[str]] | None) -> None:
        if answers is None:
            await self.client.post(f"/question/{req['id']}/reject")
        else:
            await self.client.post(f"/question/{req['id']}/reply", json={"answers": answers})

    async def abort(self, session_id: str) -> None:
        await self.client.post(f"/session/{session_id}/abort")

    async def delete(self, session_id: str) -> None:
        await self.client.delete(f"/session/{session_id}")

    async def commands(self) -> list[dict[str, Any]]:
        response = await self.client.get("/command")
        response.raise_for_status()
        found = response.json()
        return found if isinstance(found, list) else []

    async def saved_allows(self) -> list[str]:
        return []

    async def provider_blocked(self) -> bool:
        return False


class _V2Api(_V1Api):
    """The OpenCode 2 HTTP API under ``/api``: the folder is a ``location``, questions are forms."""

    events_path = "/api/event"

    def __init__(self, client: httpx.AsyncClient, root: Path) -> None:
        self.client = client
        client.params = httpx.QueryParams({"location[directory]": str(root)})
        self.root = root

    def translator(self, session_id: str) -> OpenCodeTranslator:
        return OpenCodeV2Translator(session_id)

    async def create_session(self, title: str, model: str) -> str:
        # OpenCode 2 has no title field: it names the session from the first message itself.
        created = await self.client.post("/api/session", json={
            "model": {"id": model, "providerID": PROVIDER_ID}, "location": {"directory": str(self.root)}})
        created.raise_for_status()
        return str(created.json()["data"]["id"])

    async def prompt(self, session_id: str, text: str, model: str) -> None:
        (await self.client.post(f"/api/session/{session_id}/prompt", json={"text": text})).raise_for_status()

    async def command(self, session_id: str, name: str, arguments: str, model: str) -> httpx.Response:
        sent = await self.client.post(f"/api/session/{session_id}/command", timeout=None, json={"name": name, "text": arguments})
        if sent.status_code == 404 and "CommandNotFound" in sent.text:
            # Skills are listed beside commands (as 1.x did) but run as a prompt that names the skill.
            sent = await self.client.post(f"/api/session/{session_id}/prompt",
                                          json={"text": arguments or f"Use the {name} skill.", "skills": [{"id": name}]})
        return sent

    def inline(self, event: dict[str, Any], session_id: str) -> dict[str, Any] | None:
        data = event.get("data") or {}
        if not isinstance(data, dict):
            return None
        if event.get("type") == "permission.asked" and data.get("sessionID") == session_id and data.get("id"):
            return {"type": "_permission", "request": _v2_permission(data)}
        form = data.get("form")
        if event.get("type") == "form.created" and isinstance(form, dict) and form.get("sessionID") == session_id and form.get("id"):
            question = _v2_question(form)
            return {"type": "_question", "request": question} if question else None
        return None

    async def pending(self, session_id: str, questions: bool) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        sources = [(f"/api/session/{session_id}/permission", "_permission")]
        if questions:
            sources.append((f"/api/session/{session_id}/form", "_question"))
        for path, kind in sources:
            try:
                listed = (await self.client.get(path)).json().get("data")
            except (httpx.HTTPError, ValueError, AttributeError):
                continue
            for item in listed if isinstance(listed, list) else []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                req = _v2_permission(item) if kind == "_permission" else _v2_question(item)
                if req and req.get("sessionID") == session_id:
                    found.append({"type": kind, "request": req})
        return found

    async def reply_permission(self, session_id: str, req: dict[str, Any], approved: bool) -> None:
        await self.client.post(f"/api/session/{session_id}/permission/{req['id']}/reply",
                               json={"decision": "once" if approved else "reject"})

    async def reply_question(self, session_id: str, req: dict[str, Any], answers: list[list[str]] | None) -> None:
        if answers is None:
            await self.client.delete(f"/api/session/{session_id}/form/{req['id']}")
            return
        answer: dict[str, Any] = {}
        for key, many, given in zip(req.get("keys") or [], req.get("multiple") or [], answers, strict=False):
            answer[key] = list(given) if many else (given[0] if given else "")
        await self.client.post(f"/api/session/{session_id}/form/{req['id']}/reply", json={"answer": answer})

    async def abort(self, session_id: str) -> None:
        await self.client.post(f"/api/session/{session_id}/interrupt")

    async def delete(self, session_id: str) -> None:
        await self.client.delete(f"/api/session/{session_id}")

    async def commands(self) -> list[dict[str, Any]]:
        """Commands and skills in the 1.x ``/command`` shape. The list fills in shortly after startup."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + COMMAND_LOAD_SECONDS
        while True:
            response = await self.client.get("/api/command")
            response.raise_for_status()
            commands = response.json().get("data") or []
            if commands or loop.time() >= deadline:
                break
            await asyncio.sleep(0.25)
        skills = (await self.client.get("/api/skill")).json().get("data") or []
        # OpenCode 2 no longer says whether a command takes arguments, so every command may.
        return [{"name": c.get("name"), "description": c.get("description") or "", "source": "command", "hints": ["$ARGUMENTS"]}
                for c in commands if isinstance(c, dict)] + \
               [{"name": s.get("id"), "description": s.get("description") or "", "source": "skill", "hints": []}
                for s in skills if isinstance(s, dict)]

    async def saved_allows(self) -> list[str]:
        """What the user saved with "Always allow" for this project, as "permission pattern".

        OpenCode 2 keeps these in the opencode.db it shares with the user's own OpenCode and checks them
        after the config's rules, so each one beats an "ask" from Work."""
        try:
            saved = (await self.client.get("/api/permission/saved")).json().get("data")
        except (httpx.HTTPError, ValueError, AttributeError):
            return []
        return [f"{V2_PERMISSION_NAMES.get(str(s['action']), s['action'])} {s.get('resource') or '*'}"
                for s in saved if isinstance(s, dict) and s.get("action")] if isinstance(saved, list) else []

    async def provider_blocked(self) -> bool:
        """True when the user's own config removes Work's provider.

        OpenCode 2 ranks policies from user and project config above the inline config (so
        ``enabled_providers`` without arc-llama in it blocks it), and a run would then fail with no clear reason."""
        with contextlib.suppress(httpx.HTTPError, ValueError, AttributeError):
            listed = (await self.client.get("/api/provider")).json().get("data")
            if isinstance(listed, list):
                return not any(isinstance(p, dict) and p.get("id") == PROVIDER_ID for p in listed)
        return False


async def _api(client: httpx.AsyncClient, root: Path) -> _V1Api:
    """The API this server speaks: OpenCode 2 answers ``/api/info`` with JSON, 1.x with its web page."""
    with contextlib.suppress(httpx.HTTPError, ValueError):
        info = await client.get("/api/info")
        if info.status_code == 200 and "json" in info.headers.get("content-type", "") and isinstance(info.json(), dict):
            return _V2Api(client, root)
    return _V1Api(client, root)


class _Server:
    """One private ``opencode serve`` process."""

    def __init__(self, binary: str, root: Path, env: dict[str, str]) -> None:
        self.binary, self.root, self.env = binary, root, env
        self.proc: asyncio.subprocess.Process | None = None
        self.url = ""

    async def start(self) -> str:
        if not self.root.is_dir():
            raise OpenCodeError(f"The project folder does not exist: {self.root}")
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
    question_callback: Callable[[str, list[dict[str, Any]]], Awaitable[list[list[str]] | None]] | None = None,
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
    instead of a plain prompt, so user commands, MCP prompts and skills behave as they do in OpenCode.

    With ``question_callback`` set, each question OpenCode's ``question`` tool asks is yielded as
    ``question_required`` and answered with what the callback returns (None dismisses it). Without one
    the tool is denied, so the model cannot ask something nobody will see."""
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
                         mcp_servers=mcp_server_names(root), profile=tool_profile,
                         questions=question_callback is not None)
        ),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
    }
    # Some OpenCode clients only get the question tool with this set; build_config denies it when unanswerable.
    if question_callback is not None:
        env["OPENCODE_ENABLE_QUESTION_TOOL"] = "1"
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

    auth = httpx.BasicAuth("opencode", password)
    pump: asyncio.Task[None] | None = None
    poller: asyncio.Task[None] | None = None
    command_task: asyncio.Task[None] | None = None
    try:
        async with httpx.AsyncClient(base_url=url, auth=auth, timeout=30.0) as client:
            api = await _api(client, root)
            if await api.provider_blocked():
                yield {"type": "error", "message": "Your own OpenCode settings turn off the arc-llama provider, so Work can't "
                       "reach your local model. Add \"arc-llama\" to enabled_providers in your opencode.json, or remove the "
                       "enabled_providers, disabled_providers or provider.use policy that blocks it, then try again."}
                return
            # With auto-approve Work only asks about commands, so a saved edit rule changes nothing.
            skipped = [s for s in await api.saved_allows() if not auto_confirm or s.startswith("bash ")]
            if skipped:
                shown = ", ".join(skipped[:5]) + (f" and {len(skipped) - 5} more" if len(skipped) > 5 else "")
                yield {"type": "warning", "message": f"OpenCode won't ask before these, because they were saved with "
                       f"Always allow in your own OpenCode for this project: {shown}. Remove them in OpenCode to have Work ask again."}

            async def pump_events() -> None:
                try:
                    async with client.stream("GET", api.events_path, timeout=None) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data:"):
                                with contextlib.suppress(ValueError):
                                    await queue.put(json.loads(line[5:]))
                except (httpx.HTTPError, asyncio.CancelledError):
                    pass
                finally:
                    await queue.put(None)

            pump = asyncio.create_task(pump_events())
            session_id = await api.create_session((session_title or task).strip().split("\n")[0][:80] or "arc-llama", model)
            yield {"type": "harness_session", "session_id": "oc-" + session_id}
            translator = api.translator(session_id)

            handled: set[str] = set()      # permission and question ids already put to the user (events and polling can both report one)

            async def poll_permissions() -> None:
                seen: set[str] = set()
                while True:
                    await asyncio.sleep(PERMISSION_POLL_SECONDS)
                    for found in await api.pending(session_id, question_callback is not None):
                        if found["request"]["id"] not in seen:
                            seen.add(found["request"]["id"])
                            await queue.put(found)

            poller = asyncio.create_task(poll_permissions())
            if command:
                # The command can take as long as the run, so send it alongside the event stream and
                # only look at the result if it failed to start.
                async def run_command() -> None:
                    try:
                        sent = await api.command(session_id, command, arguments, model)
                        if sent.status_code >= 400:
                            detail = " ".join(sent.text.split())[:300]
                            await queue.put({"type": "_failed", "message": f"OpenCode could not run /{command} (HTTP {sent.status_code}){': ' + detail if detail else ''}."})
                    except httpx.HTTPError as exc:
                        await queue.put({"type": "_failed", "message": f"OpenCode could not run /{command}: {exc}"})

                command_task = asyncio.create_task(run_command())
            else:
                await api.prompt(session_id, task, model)

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
                # The stream is the reliable source: OpenCode 1.19's list endpoint can answer 400 for a
                # pending web fetch (its own response validation fails), so polling alone can hang.
                event = api.inline(event, session_id) or event
                if event.get("type") == "_question":
                    req = event["request"]
                    if req.get("id") in handled or question_callback is None:
                        continue
                    handled.add(req["id"])
                    questions = clean_questions(req.get("questions"))
                    call_id = str((req.get("tool") or {}).get("callID") or "")
                    answers = None
                    if questions:
                        yield {"type": "question_required", "id": req["id"], "call_id": call_id, "questions": questions}
                        asked_at = loop.time()
                        try:
                            answers = await asyncio.wait_for(question_callback(req["id"], questions), timeout)
                        except asyncio.TimeoutError:
                            answers = None
                        # Time the user spends answering is not run time.
                        deadline += loop.time() - asked_at
                    await api.reply_question(session_id, req, answers)
                    continue
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
                    asked_at = loop.time()
                    approved = bool(await confirm_callback(req["id"], tool, ask_arguments)) if confirm_callback else auto_confirm
                    deadline += loop.time() - asked_at
                    await api.reply_permission(session_id, req, approved)
                    continue
                for out in translator.feed(event):
                    yield out
            with contextlib.suppress(httpx.HTTPError):
                await api.abort(session_id)
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
        async with httpx.AsyncClient(base_url=url, auth=httpx.BasicAuth("opencode", password), timeout=30.0) as client:
            found = await (await _api(client, root)).commands()
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
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
        async with httpx.AsyncClient(base_url=url, auth=httpx.BasicAuth("opencode", password), timeout=30.0) as client:
            api = await _api(client, root)
            session_id = await api.create_session("arc-llama tool scan", model)
            try:
                await api.prompt(session_id, "hi", model)
                loop = asyncio.get_running_loop()
                deadline = loop.time() + timeout
                while not captured and loop.time() < deadline:
                    await asyncio.sleep(0.5)
            finally:
                with contextlib.suppress(httpx.HTTPError):
                    await api.abort(session_id)
                with contextlib.suppress(httpx.HTTPError):
                    await api.delete(session_id)        # leave no trace in the user's history
    except (httpx.HTTPError, OpenCodeError, KeyError, ValueError) as exc:
        raise OpenCodeError(f"Could not read OpenCode's tools: {exc}") from exc
    finally:
        listener.shutdown()
        listener.server_close()
        thread.join()
        server.terminate_now()
        with contextlib.suppress(BaseException):
            await asyncio.shield(server.stop())
    if not captured:
        raise OpenCodeError("OpenCode did not report its tools in time.")
    return captured[-1]


def group_tools(names: list[str], servers: list[str]) -> list[dict[str, str]]:
    """Label each tool as built-in, from one MCP server, or from a plugin."""
    builtin = {"bash", "read", "glob", "grep", "list", "task", "todowrite", "todoread", "lsp", "patch",
               "shell", "execute", "subagent",                      # OpenCode 2's names
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
