"""Prove tool profiles against the real OpenCode installed on this machine.

For a project folder it asks OpenCode which tools it would send the model, first with no profile
and then under a profile, and fails if anything a profile switched off still reaches the model, or
if something it kept went missing. Needs OpenCode and starts the user's MCP servers, so it is not
part of the unit tests; run it with `scripts/check.sh --live`."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from arc_llama.agent.harness.opencode import (
    HIDDEN_TOOLS,
    build_config,
    find_binary,
    group_tools,
    mcp_server_names,
    probe_tools,
)

# OpenCode only offers these while a connected MCP server exposes resources, so a profile can drop them.
CONDITIONAL_TOOLS = ("list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()


async def main() -> int:
    if find_binary() is None:
        print("opencode not found: skipping")
        return 0
    servers = mcp_server_names(ROOT)
    everything = group_tools(await probe_tools(ROOT, build_config("http://127.0.0.1:1", "m", mcp_servers=servers)), servers)
    names = {t["id"] for t in everything}
    print(f"OpenCode offers {len(names)} tools; MCP servers: {servers or 'none'}")
    problems: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    keep = servers[:1]
    droppable = [t["id"] for t in everything if t["group"] == "plugin"][:1] + [t["id"] for t in everything if t["group"] == "builtin" and t["id"] == "bash"]
    profile = {"mcp": keep, "skills": [], "web": False, "tools_off": droppable}
    under = set(await probe_tools(ROOT, build_config("http://127.0.0.1:1", "m", mcp_servers=servers, profile=profile)))
    print(f"Under a profile keeping {keep or 'no servers'} and dropping {droppable or 'nothing'}: {len(under)} tools")

    for tool in everything:
        gone = tool["group"] in servers and tool["group"] not in keep
        gone = gone or tool["id"] in droppable or tool["id"] in ("webfetch", "websearch", "codesearch")
        if gone:
            require(tool["id"] not in under, f"{tool['id']} was switched off but still reaches the model")
        elif tool["id"] not in HIDDEN_TOOLS and tool["id"] not in CONDITIONAL_TOOLS:
            require(tool["id"] in under, f"{tool['id']} should have stayed available but is missing")
    require(len(under) < len(names) or not (servers or droppable), "the profile did not reduce the tool list")

    empty = set(await probe_tools(ROOT, build_config("http://127.0.0.1:1", "m", mcp_servers=servers, profile={"mcp": [], "skills": [], "web": False})))
    require(not [t for t in everything if t["group"] in servers and t["id"] in empty], "a profile with no servers still sent MCP tools")

    for message in problems:
        print("PROBLEM:", message)
    print("profile enforcement: " + ("FAILED" if problems else "ok"))
    return 1 if problems else 0


sys.exit(asyncio.run(main()))
