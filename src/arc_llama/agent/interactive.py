"""Interactive, multi-turn agent loop for a terminal REPL.

Unlike ``run_agent``, which executes a single task to completion, this module
maintains a conversation across user inputs so the agent behaves like Claude
Code or similar coding assistants.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.prompts import (
    PLANNING_INSTRUCTIONS,
    SYSTEM_PROMPT,
    USER_INSTRUCTIONS,
    format_tool_result,
)
from arc_llama.agent.tools import TOOLS, ToolContext, ToolResult
from arc_llama.chat_store import ChatStore

log = logging.getLogger("arc_llama.agent.interactive")


class InteractiveAgent:
    """Persistent terminal agent that can chat and use tools over many turns."""

    def __init__(
        self,
        model: str,
        base_url: str,
        root: Path,
        *,
        auto_confirm: bool = False,
        plan_mode: bool = False,
        max_turns: int = 30,
        timeout: float = 600.0,
        chat_store: ChatStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.root = Path(root).resolve()
        self.auto_confirm = auto_confirm
        self.plan_mode = plan_mode
        self.max_turns = max_turns
        self.timeout = timeout
        self.chat_store = chat_store
        self.checkpoint_store = checkpoint_store
        self.run_id = run_id
        self.extra = dict(extra or {})

        self.conversation: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTIONS},
        ]
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout, base_url=self.base_url)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _tool_context(self) -> ToolContext:
        extra = dict(self.extra)
        extra["chat_store"] = self.chat_store
        return ToolContext(
            root=self.root,
            client=self._client,  # type: ignore[arg-type]
            extra=extra,
            checkpoint_store=self.checkpoint_store,
            run_id=self.run_id,
        )

    async def chat(
        self,
        user_text: str,
        *,
        confirm_callback: Callable[[str, str, dict[str, Any]], Awaitable[bool]] | None = None,
        plan_callback: Callable[[str], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process one user turn and yield agent events.

        The conversation history is preserved across calls so the agent can
        refer to earlier turns.
        """
        if self.plan_mode and not self.auto_confirm and plan_callback is None:
            yield {
                "type": "error",
                "message": "plan_mode requires a plan_callback or auto_confirm=True",
            }
            return

        client = await self._get_client()
        ctx = self._tool_context()
        ctx.client = client

        if self.plan_mode:
            yield {"type": "status", "message": "Generating plan..."}
            plan_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{PLANNING_INSTRUCTIONS}\nTask: {user_text}"},
            ]
            try:
                plan_response = await client.post(
                    "/v1/chat/completions",
                    json={"model": self.model, "messages": plan_messages},
                )
                plan_response.raise_for_status()
                plan_data = plan_response.json()
                plan_content = (
                    plan_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
            except httpx.HTTPError as e:
                yield {"type": "error", "message": f"LLM request failed while planning: {e}"}
                return
            except json.JSONDecodeError as e:
                yield {"type": "error", "message": f"Invalid JSON from LLM while planning: {e}"}
                return

            if not plan_content:
                yield {"type": "error", "message": "Model returned an empty plan"}
                return

            yield {"type": "plan", "content": plan_content}

            if not self.auto_confirm:
                assert plan_callback is not None  # noqa: S101
                try:
                    approved = await plan_callback(plan_content)
                except Exception as e:
                    yield {"type": "error", "message": f"Plan approval failed: {e}"}
                    return

                if not approved:
                    yield {"type": "status", "message": "Plan denied by user."}
                    yield {"type": "done"}
                    return

            self.conversation.insert(
                1,
                {"role": "system", "content": f"Approved plan:\n{plan_content}"},
            )

        self.conversation.append({"role": "user", "content": user_text})
        yield {
            "type": "status",
            "message": f"Running agent for model '{self.model}' in {self.root}",
        }

        for turn in range(self.max_turns):
            yield {"type": "status", "message": f"Turn {turn + 1}/{self.max_turns}"}

            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": self.conversation,
                        "tools": TOOLS.definitions,
                        "tool_choice": "auto",
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                yield {"type": "error", "message": f"LLM request failed: {e}"}
                return
            except json.JSONDecodeError as e:
                yield {"type": "error", "message": f"Invalid JSON from LLM: {e}"}
                return

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []

            self.conversation.append(message)

            if content:
                yield {"type": "assistant", "content": content}

            if not tool_calls:
                yield {"type": "done"}
                return

            for call_index, tc in enumerate(tool_calls):
                name = tc.get("function", {}).get("name")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {}

                call_id = tc.get("id") or f"turn-{turn}-call-{call_index}"

                yield {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}

                if TOOLS.requires_confirmation(name) and not self.auto_confirm:
                    yield {
                        "type": "confirm_required",
                        "id": call_id,
                        "tool": name,
                        "arguments": arguments,
                    }
                    assert confirm_callback is not None  # noqa: S101
                    try:
                        approved = await confirm_callback(call_id, name, arguments)
                    except Exception as e:
                        yield {"type": "error", "message": f"Confirmation failed: {e}"}
                        return

                    if not approved:
                        result = ToolResult(content=f"User denied {name}.", error=True)
                    else:
                        result = await TOOLS.execute(name, arguments, ctx)
                else:
                    result = await TOOLS.execute(name, arguments, ctx)

                if result.checkpoint_id:
                    yield {"type": "checkpoint", "id": result.checkpoint_id, "run_id": self.run_id}

                yield {
                    "type": "tool_result",
                    "id": call_id,
                    "name": name,
                    "content": result.content,
                    "error": result.error,
                }

                result_text = format_tool_result(name, arguments, result.content, result.error)
                self.conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", call_id),
                        "name": name,
                        "content": result_text,
                    }
                )

        yield {
            "type": "error",
            "message": f"Agent reached the maximum number of turns ({self.max_turns}).",
        }
