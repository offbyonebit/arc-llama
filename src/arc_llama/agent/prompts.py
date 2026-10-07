"""System prompts and formatting for the arc-llama coding agent."""
from __future__ import annotations

# Keep the system message minimal. Some local models (notably Qwen3-Coder through
# llama-server's chat template) switch to an XML function-calling format when the
# system message mentions files, tools, or coding assistance, which breaks the
# OpenAI-compatible tool_calls handling in the agent loop. The detailed behavioural
# instructions are therefore sent as the first user message instead.
SYSTEM_PROMPT = "You are a helpful assistant."

PLANNING_INSTRUCTIONS = (
    "You are working inside the user's project directory.\n"
    "\n"
    "Before making any changes, produce a clear implementation plan for the "
    "user's request. List the files you need to read or modify, the commands "
    "you intend to run, and the order of operations. Do not write any files or "
    "run any commands yet. Keep the plan concise but specific enough to review.\n"
)

USER_INSTRUCTIONS = (
    "You are working inside the user's project directory.\n"
    "\n"
    "You have access to functions that let you read, write, search, and execute "
    "commands inside the project root. Use them step by step to answer the user's "
    "request.\n"
    "\n"
    "Guidelines:\n"
    "- Before changing files, read them first.\n"
    "- Prefer small, targeted edits over large rewrites.\n"
    "- After writing files, run commands (linters, tests, type-checkers) to verify "
    "your work when appropriate.\n"
    "- If a command fails, diagnose the error and retry if it makes sense.\n"
    "- Do not assume file contents; use read_file or list_directory to inspect the "
    "project.\n"
    "- Keep the user informed of what you are doing.\n"
    "- When finished, summarize what you changed.\n"
    "\n"
    "You may call multiple functions in parallel when they are independent.\n"
    "\n"
    "You can also reference the user's saved chat history with the list_chats, "
    "read_chat, and search_chats functions. Use these when the user refers to something "
    "they discussed earlier or asks \"what did I do on ...\". User-installed skills "
    "may add additional functions; use them when they are relevant to the task.\n"
    "\n"
    "Task: "
)


def format_tool_result(name: str, arguments: dict, result: str, error: bool) -> str:
    """Format a tool result for inclusion in the conversation history."""
    status = "error" if error else "ok"
    return f'<tool_result name="{name}" status="{status}">\n{result}\n</tool_result>'
