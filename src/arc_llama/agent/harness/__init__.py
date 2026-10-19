"""Agent harness adapters.

A harness is an external agent runtime (OpenCode, ...) that arc-llama drives
over its own protocol while supplying the inference. Every adapter yields the
same SSE-shaped events as :func:`arc_llama.agent.loop.run_agent`, so a UI
written against that vocabulary works with any harness.
"""
import os

from arc_llama.agent.harness.hermes import find_hermes_binary, run_hermes
from arc_llama.agent.harness.opencode import (
    OpenCodeError,
    OpenCodeTranslator,
    build_config,
    filter_commands,
    find_binary,
    list_capabilities,
    list_commands,
    mcp_server_names,
    run_opencode,
)

HARNESSES = ("builtin", "opencode", "hermes")


MAX_HISTORY_ITEMS = 24
MAX_HISTORY_ITEM_CHARS = 3000
MAX_HISTORY_TOTAL_CHARS = 10000


def clean_history(history: object) -> list[dict[str, str]] | None:
    """Validate the conversation a client sends. Returns None if the shape is wrong."""
    if history is None:
        return []
    if not isinstance(history, list) or len(history) > 200:
        return None
    out: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant") or not isinstance(item.get("content"), str):
            return None
        out.append({"role": item["role"], "content": item["content"][:MAX_HISTORY_ITEM_CHARS]})
    return out[-MAX_HISTORY_ITEMS:]


def compose_prompt(task: str, history: list[dict[str, str]], root: str | None = None) -> str:
    """The message the agent receives: earlier turns as context, then the new request.

    Oldest turns are dropped first when the conversation is long. The user's own
    transcript keeps the plain `task`; only the agent sees this composed text."""
    where = (
        f"The project folder is {root}. Create and edit files inside it, using paths relative to it, "
        "and do not write anywhere else.\n\n"
        if root
        else ""
    )
    if not history:
        return where + task
    kept: list[str] = []
    budget = MAX_HISTORY_TOTAL_CHARS
    for item in reversed(history):
        line = ("User: " if item["role"] == "user" else "Assistant: ") + item["content"]
        if budget - len(line) < 0 and kept:
            break
        kept.append(line)
        budget -= len(line)
    earlier = "\n\n".join(reversed(kept))
    return (
        f"{where}You are continuing a conversation with the user, working in the same project folder. "
        "Here is what has been said so far:\n\n"
        f"{earlier}\n\n"
        "The user's new message (act on this one):\n"
        f"{task}"
    )


def experimental_enabled() -> bool:
    """External harnesses ship behind the same switch as the built-in agent."""
    return os.environ.get("ARC_LLAMA_EXPERIMENTAL_AGENT", "").lower() in ("1", "true", "yes")

__all__ = [
    "HARNESSES",
    "clean_history",
    "compose_prompt",
    "experimental_enabled",
    "OpenCodeError",
    "OpenCodeTranslator",
    "build_config",
    "filter_commands",
    "find_binary",
    "list_capabilities",
    "list_commands",
    "mcp_server_names",
    "run_opencode",
    "find_hermes_binary",
    "run_hermes",
]
