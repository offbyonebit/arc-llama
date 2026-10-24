"""Agent loop for the arc-llama coding assistant.

The loop talks to the local OpenAI-compatible server, executes tools in a
sandboxed project root, and yields SSE events so a UI can stream progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from arc_llama.agent.checkpoints import CheckpointStore
from arc_llama.agent.prompts import (
    DOCUMENT_INSTRUCTIONS,
    PLANNING_INSTRUCTIONS,
    SYSTEM_PROMPT,
    USER_INSTRUCTIONS,
    format_tool_result,
)
from arc_llama.agent.tools import TOOLS, ToolContext, ToolResult
from arc_llama.chat_store import ChatStore
from arc_llama.stream_metrics import generation_rate

log = logging.getLogger("arc_llama.agent.loop")


async def run_agent(
    task: str,
    model: str,
    base_url: str,
    root: Path,
    *,
    auto_confirm: bool = False,
    confirm_callback: Callable[[str, str, dict[str, Any]], Awaitable[bool]] | None = None,
    plan_mode: bool = False,
    plan_callback: Callable[[str], Awaitable[bool]] | None = None,
    run_id: str | None = None,
    checkpoint_store: CheckpointStore | None = None,
    max_turns: int = 30,
    report_context: bool = False,
    timeout: float = 600.0,
    chat_store: ChatStore | None = None,
    extra: dict[str, Any] | None = None,
    work_type: str = "code",
) -> AsyncIterator[dict[str, Any]]:
    """Run the agent loop and yield SSE-shaped events.

    Events:
        {"type": "status", "message": str}
        {"type": "plan", "content": str}
        {"type": "assistant", "content": str}
        {"type": "usage", "prompt_tokens": int, "completion_tokens": int}
        {"type": "performance", "generation_tok_s": float}
        {"type": "tool_call", "id": str, "name": str, "arguments": dict}
        {"type": "tool_result", "id": str, "name": str, "content": str, "error": bool}
        {"type": "confirm_required", "id": str, "tool": str, "arguments": dict}
        {"type": "error", "message": str}
        {"type": "done"}

    Args:
        confirm_callback: Optional async callable invoked as
            `confirm_callback(call_id, tool_name, arguments)` when a
            write_file/run_command call needs approval. Should return True
            to execute the tool, False to deny it.
        plan_callback: Optional async callable invoked as
            `plan_callback(plan_text)` when plan_mode is enabled and
            auto_confirm is False.
    """
    root = root.resolve()
    if work_type == "documents":
        from arc_llama.agent.documents import documents_registry

        tools = documents_registry()
        instructions = DOCUMENT_INSTRUCTIONS
    else:
        tools = TOOLS
        instructions = USER_INSTRUCTIONS
    if plan_mode and not auto_confirm and plan_callback is None:
        yield {
            "type": "error",
            "message": "plan_mode requires a plan_callback or auto_confirm=True",
        }
        return

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instructions}{task}"},
    ]

    yield {"type": "status", "message": f"Starting agent for model '{model}' in {root}"}

    async with httpx.AsyncClient(timeout=timeout, base_url=base_url.rstrip("/")) as client:
        tool_extra = dict(extra or {})
        tool_extra["chat_store"] = chat_store
        ctx = ToolContext(
            root=root,
            client=client,
            extra=tool_extra,
            checkpoint_store=checkpoint_store,
            run_id=run_id,
        )

        if plan_mode:
            yield {"type": "status", "message": "Generating plan..."}
            plan_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{PLANNING_INSTRUCTIONS}\nTask: {task}"},
            ]
            try:
                plan_response = await client.post(
                    "/v1/chat/completions",
                    json={"model": model, "messages": plan_messages},
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

            if not auto_confirm:
                assert plan_callback is not None  # noqa: S101
                try:
                    approved = await asyncio.wait_for(
                        plan_callback(plan_content),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    yield {"type": "error", "message": "Plan approval timed out."}
                    return
                except Exception as e:
                    yield {"type": "error", "message": f"Plan approval failed: {e}"}
                    return

                if not approved:
                    yield {"type": "status", "message": "Plan denied by user."}
                    yield {"type": "done"}
                    return

            # Inject the approved plan into the execution conversation.
            conversation.insert(
                1,
                {"role": "system", "content": f"Approved plan:\n{plan_content}"},
            )

        for turn in range(max_turns):
            yield {"type": "status", "message": f"Turn {turn + 1}/{max_turns}"}

            yield {"type": "activity", "phase": "generating"}
            if report_context:
                try:
                    counted = await client.post("/v1/chat/budget", json={"model": model, "messages": conversation, "tools": tools.definitions, "max_tokens": 1}, timeout=4.0)
                    counted.raise_for_status()
                    budget = counted.json()
                    if type(budget.get("prompt_tokens")) is int and budget["prompt_tokens"] >= 0:
                        yield {"type": "context", "prompt_tokens": budget["prompt_tokens"], "context_tokens": budget.get("context_tokens"), "exact": budget.get("exact") is True, "source": "agent_request"}
                except (httpx.HTTPError, ValueError, TypeError):
                    pass
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": conversation,
                        "tools": tools.definitions,
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

            rate = generation_rate(data)
            if rate is not None:
                yield {"type": "performance", "generation_tok_s": rate}

            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if (type(prompt_tokens) is int and prompt_tokens >= 0
                    and type(completion_tokens) is int and completion_tokens >= 0):
                yield {"type": "usage", "prompt_tokens": prompt_tokens,
                       "completion_tokens": completion_tokens}

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []

            conversation.append(message)

            if content:
                yield {"type": "assistant", "content": content}

            if not tool_calls:
                yield {"type": "done"}
                return

            # Process tool calls
            for call_index, tc in enumerate(tool_calls):
                name = tc.get("function", {}).get("name")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {}

                call_id = tc.get("id") or f"turn-{turn}-call-{call_index}"

                yield {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}

                if tools.requires_confirmation(name) and not auto_confirm:
                    yield {
                        "type": "confirm_required",
                        "id": call_id,
                        "tool": name,
                        "arguments": arguments,
                    }
                    assert confirm_callback is not None  # noqa: S101
                    try:
                        approved = await asyncio.wait_for(
                            confirm_callback(call_id, name, arguments),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        yield {
                            "type": "error",
                            "message": "Confirmation timed out.",
                        }
                        return
                    except Exception as e:
                        yield {
                            "type": "error",
                            "message": f"Confirmation failed: {e}",
                        }
                        return

                    if not approved:
                        result = ToolResult(
                            content=f"User denied {name}.",
                            error=True,
                        )
                    else:
                        result = await tools.execute(name, arguments, ctx)
                else:
                    result = await tools.execute(name, arguments, ctx)

                if result.checkpoint_id:
                    yield {
                        "type": "checkpoint",
                        "id": result.checkpoint_id,
                        "run_id": run_id,
                    }

                yield {
                    "type": "tool_result",
                    "id": call_id,
                    "name": name,
                    "content": result.content,
                    "error": result.error,
                }

                result_text = format_tool_result(name, arguments, result.content, result.error)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", call_id),
                        "name": name,
                        "content": result_text,
                    }
                )
                if result.images:
                    # Tool messages carry text only, so a page picture goes in as the next user turn.
                    conversation.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Image from {name}:"}]
                            + [{"type": "image_url", "image_url": {"url": url}} for url in result.images],
                        }
                    )

        yield {
            "type": "error",
            "message": f"Agent reached the maximum number of turns ({max_turns}).",
        }
