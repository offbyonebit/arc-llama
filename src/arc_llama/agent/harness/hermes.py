"""Hermes Agent's ACP stdio transport adapted to Arc Llama Work events.

Each run has an isolated Hermes home and routes primary and auxiliary inference
through Arc Llama. Hermes' default ACP mode asks before file edits and commands
its own checks flag; it does not offer Arc Llama's auto-approve policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio

from arc_llama.agent.harness.inference import InferenceObserver

STARTUP_TIMEOUT = 45.0
MAX_FRAME = 4 * 1024 * 1024


def find_hermes_binary(configured: str | None = None) -> str | None:
    for candidate in (configured, os.environ.get("ARC_LLAMA_HERMES_BIN"), "hermes"):
        if candidate and (found := shutil.which(candidate)):
            return found
    return None


def build_hermes_config(model: str, base_url: str) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1"
    local = {
        "provider": "custom",
        "model": model,
        "base_url": endpoint,
        "api_key": "arc-llama-local",
    }
    return {
        "model": {**local, "default": model, "api_mode": "chat_completions"},
        "fallback_model": [],
        "providers": {},
        "custom_providers": [],
        "mcp_servers": {},
        "approvals": {"mode": "manual", "timeout": 3600},
        "compression": {"enabled": False},
        "auxiliary": {
            name: dict(local)
            for name in (
                "vision",
                "web_extract",
                "compression",
                "skills_hub",
                "approval",
                "mcp",
                "title_generation",
                "triage_specifier",
                "kanban_decomposer",
                "profile_describer",
                "curator",
            )
        },
        "terminal": {"backend": "local", "timeout": 180},
        "agent": {"max_turns": 30},
        "display": {"show_thinking": False},
    }


def hermes_environment(home: Path) -> dict[str, str]:
    # Start with an allow-list. In particular do not inherit provider keys,
    # proxies, PYTHONPATH, HERMES_YOLO_MODE, or an alternate Hermes config.
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "USER", "SYSTEMROOT")
        if key in os.environ
    }
    env.update(
        {
            "HOME": str(home),
            "HERMES_HOME": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "XDG_DATA_HOME": str(home / "data"),
            "PYTHONUNBUFFERED": "1",
            "HERMES_INTERACTIVE": "1",
            "HERMES_YOLO_MODE": "0",
            "NO_PROXY": "127.0.0.1,localhost,::1",
        }
    )
    return env


class HermesError(RuntimeError):
    """The installed Hermes runtime could not complete its ACP exchange."""


def _tool(update: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = update.get("rawInput")
    args = raw if isinstance(raw, dict) else {}
    if isinstance(args.get("arguments"), dict) and isinstance(args.get("tool"), str):
        return args["tool"], args["arguments"]
    kind = update.get("kind")
    if kind == "execute" or "command" in args:
        return "terminal", args
    if kind == "edit":
        return ("patch" if "old_string" in args else "write_file"), args
    return {"read": "read_file", "search": "search_files", "fetch": "web_fetch"}.get(
        str(kind), "hermes_tool"
    ), args


def _tool_text(update: dict[str, Any]) -> str:
    pieces = []
    for block in update.get("content") or []:
        if not isinstance(block, dict):
            continue
        content = block.get("content")
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            pieces.append(content["text"])
        elif block.get("type") == "diff":
            pieces.append(str(block.get("newText") or ""))
    if not pieces and update.get("rawOutput") is not None:
        raw = update["rawOutput"]
        pieces.append(raw if isinstance(raw, str) else json.dumps(raw))
    return "\n".join(pieces)


class _ACP:
    def __init__(self, proc: asyncio.subprocess.Process, queue: asyncio.Queue) -> None:
        self.proc, self.queue = proc, queue
        self.next_id = 0
        self.session = ""
        self.tools: dict[str, tuple[str, dict[str, Any]]] = {}

    async def send(self, message: dict[str, Any]) -> None:
        if self.proc.stdin is None:
            raise HermesError("Hermes ACP input closed")
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())
        await self.proc.stdin.drain()

    async def request(self, method: str, params: dict[str, Any]) -> int:
        self.next_id += 1
        await self.send({"id": self.next_id, "method": method, "params": params})
        return self.next_id

    async def read(self) -> None:
        assert self.proc.stdout is not None
        try:
            while raw := await self.proc.stdout.readline():
                try:
                    message = json.loads(raw)
                except ValueError as exc:
                    raise HermesError("Hermes sent malformed ACP JSON") from exc
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    raise HermesError("Hermes sent an invalid ACP frame")
                await self.queue.put({"type": "_rpc", "message": message})
        except (ValueError, HermesError) as exc:
            await self.queue.put({"type": "_failure", "message": str(exc)})
        finally:
            await self.queue.put({"type": "_eof"})

    async def stderr(self) -> None:
        assert self.proc.stderr is not None
        # Hermes may log task text. Drain without storing or publishing it.
        while await self.proc.stderr.read(8192):
            pass

    def updates(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        if params.get("sessionId") != self.session:
            return []
        update = params.get("update")
        if not isinstance(update, dict):
            raise HermesError("Hermes sent an invalid session update")
        kind = update.get("sessionUpdate")
        if kind in ("agent_message_chunk", "agent_thought_chunk"):
            content = update.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else None
            if not isinstance(text, str) or not text:
                return []
            events: list[dict[str, Any]] = [
                {
                    "type": "activity",
                    "phase": "thinking" if kind == "agent_thought_chunk" else "answering",
                }
            ]
            if kind == "agent_message_chunk":
                events.append({"type": "assistant_delta", "content": text})
            return events
        if kind in ("tool_call", "tool_call_update"):
            tool_id = update.get("toolCallId")
            if not isinstance(tool_id, str) or not tool_id:
                raise HermesError("Hermes tool event has no call id")
            events = []
            if tool_id not in self.tools:
                name, args = _tool(update)
                self.tools[tool_id] = name, args
                events.append({"type": "tool_call", "id": tool_id, "name": name, "arguments": args})
            if update.get("status") in ("completed", "failed"):
                name, _ = self.tools[tool_id]
                text = _tool_text(update)
                events.append(
                    {
                        "type": "tool_result",
                        "id": tool_id,
                        "name": name,
                        "content": text,
                        "error": update["status"] == "failed",
                    }
                )
            return events
        return []

    def permission(
        self, params: dict[str, Any], root: Path
    ) -> tuple[dict[str, Any], str | None, str | None, bool]:
        call = params.get("toolCall")
        options = params.get("options")
        if (
            params.get("sessionId") != self.session
            or not isinstance(call, dict)
            or not isinstance(options, list)
        ):
            raise HermesError("Hermes sent an invalid permission request")
        name, args = _tool(call)
        call_id = str(call.get("toolCallId") or "permission")
        for known_id, (known_name, known_args) in reversed(list(self.tools.items())):
            if known_name == name and (
                known_args.get("path") == args.get("path")
                if name in ("write_file", "patch")
                else known_args.get("command") == args.get("command")
            ):
                call_id = known_id
                break
        allowed = next(
            (
                o.get("optionId")
                for o in options
                if isinstance(o, dict)
                and o.get("kind") == "allow_once"
                and isinstance(o.get("optionId"), str)
            ),
            None,
        )
        denied = next(
            (
                o.get("optionId")
                for o in options
                if isinstance(o, dict)
                and o.get("kind") == "reject_once"
                and isinstance(o.get("optionId"), str)
            ),
            None,
        )
        inside = True
        if name in ("write_file", "patch"):
            path = args.get("path")
            inside = (
                isinstance(path, str)
                and bool(path)
                and (root / path).resolve().is_relative_to(root)
            )
        return (
            {"type": "confirm_required", "id": call_id, "tool": name, "arguments": args},
            allowed,
            denied,
            inside,
        )


async def _stop(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None and proc.stdin is not None:
        proc.stdin.close()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), 0.75)
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        elif proc.returncode is None:
            proc.terminate()
    if proc.returncode is not None:
        return
    try:
        await asyncio.wait_for(proc.wait(), 3)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        await proc.wait()


async def run_hermes(
    *,
    task: str,
    model: str,
    base_url: str,
    root: Path,
    confirm_callback: Callable[[str, str, dict], Awaitable[bool]],
    auto_confirm: bool = False,
    local_only: bool = True,
    binary: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if auto_confirm:
        yield {
            "type": "error",
            "message": "Hermes uses native approvals; auto-approve is not supported.",
        }
        yield {"type": "done"}
        return
    if not local_only or urlparse(base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        yield {"type": "error", "message": "Hermes Work requires local Arc Llama inference."}
        yield {"type": "done"}
        return
    executable = find_hermes_binary(binary)
    if executable is None:
        yield {
            "type": "error",
            "message": "Hermes Agent is not installed. Install Hermes with its ACP extra, or set ARC_LLAMA_HERMES_BIN.",
        }
        yield {"type": "done"}
        return
    root = root.resolve()
    queue: asyncio.Queue = asyncio.Queue()
    observer = InferenceObserver(base_url, task, queue)
    proc = None
    background: list[asyncio.Task] = []
    rpc = None
    temporary = tempfile.TemporaryDirectory(prefix="arc-llama-hermes-")
    try:
        home = Path(temporary.name)
        (home / "config.yaml").write_text(json.dumps(build_hermes_config(model, observer.url)))
        proc = await asyncio.create_subprocess_exec(
            executable,
            "acp",
            cwd=root,
            env=hermes_environment(home),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
            limit=MAX_FRAME,
        )
        rpc = _ACP(proc, queue)
        background = [asyncio.create_task(rpc.read()), asyncio.create_task(rpc.stderr())]
        yield {"type": "status", "message": "Starting Hermes…"}
        expected = await rpc.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {"name": "arc-llama", "version": "1"},
            },
        )
        stage = "initialize"
        deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
        while True:
            if stage == "session/prompt":
                packet = await queue.get()
            else:
                packet = await asyncio.wait_for(
                    queue.get(), max(0.01, deadline - asyncio.get_running_loop().time())
                )
            if packet["type"] == "_inference":
                yield packet["event"]
                continue
            if packet["type"] in ("_eof", "_failure"):
                raise HermesError(
                    packet.get(
                        "message",
                        "Hermes ACP exited before completing the task. Check that Hermes ACP dependencies are installed.",
                    )
                )
            message = packet["message"]
            method = message.get("method")
            params = message.get("params") or {}
            if not isinstance(params, dict):
                raise HermesError("Hermes ACP params must be an object")
            if method == "session/update":
                for event in rpc.updates(params):
                    yield event
            elif method and "id" in message:
                if method != "session/request_permission" or stage != "session/prompt":
                    await rpc.send(
                        {
                            "id": message["id"],
                            "error": {"code": -32601, "message": "Unsupported client method"},
                        }
                    )
                    continue
                event, allowed, denied, inside = rpc.permission(params, root)
                approved = False
                if inside and allowed is not None:
                    yield event
                    try:
                        approved = await asyncio.wait_for(
                            confirm_callback(event["id"], event["tool"], event["arguments"]), 50
                        )
                    except TimeoutError as exc:
                        raise HermesError(
                            "Hermes approval timed out. Start the task again to retry."
                        ) from exc
                choice = allowed if approved else denied
                outcome = (
                    {"outcome": "selected", "optionId": choice}
                    if choice is not None
                    else {"outcome": "cancelled"}
                )
                await rpc.send({"id": message["id"], "result": {"outcome": outcome}})
            elif "id" in message:
                if message["id"] != expected:
                    raise HermesError("Hermes ACP returned an unexpected response id")
                if message.get("error"):
                    raise HermesError(
                        f"Hermes rejected {stage}. Check its ACP installation and the selected local model."
                    )
                result = message.get("result")
                if not isinstance(result, dict):
                    raise HermesError("Hermes ACP returned an invalid response")
                if stage == "initialize":
                    if result.get("protocolVersion") != 1:
                        raise HermesError("Hermes ACP protocol version is not supported")
                    stage = "session/new"
                    expected = await rpc.request(stage, {"cwd": str(root), "mcpServers": []})
                elif stage == "session/new":
                    sid = result.get("sessionId")
                    if not isinstance(sid, str) or not sid:
                        raise HermesError("Hermes did not create an ACP session")
                    rpc.session = sid
                    yield {"type": "harness_session", "session_id": "hermes-" + sid}
                    stage = "session/set_mode"
                    expected = await rpc.request(stage, {"sessionId": sid, "modeId": "default"})
                elif stage == "session/set_mode":
                    stage = "session/prompt"
                    expected = await rpc.request(
                        stage,
                        {"sessionId": rpc.session, "prompt": [{"type": "text", "text": task}]},
                    )
                    yield {"type": "status", "message": "Hermes is working…"}
                else:
                    if result.get("stopReason") != "end_turn":
                        raise HermesError(
                            f"Hermes stopped without finishing ({result.get('stopReason', 'unknown reason')})."
                        )
                    break
    except (OSError, TimeoutError, HermesError, BrokenPipeError) as exc:
        yield {
            "type": "error",
            "message": "Hermes did not become ready in time."
            if isinstance(exc, TimeoutError)
            else str(exc),
        }
    finally:
        with anyio.CancelScope(shield=True):
            if proc is not None:
                if rpc is not None and rpc.session:
                    with contextlib.suppress(OSError, ConnectionError):
                        await rpc.send(
                            {"method": "session/cancel", "params": {"sessionId": rpc.session}}
                        )
                await asyncio.shield(_stop(proc))
            for job in background:
                job.cancel()
            if background:
                await asyncio.shield(asyncio.gather(*background, return_exceptions=True))
            await asyncio.shield(observer.close())
            temporary.cleanup()
    yield {"type": "done"}
