"""MCP client support for extending the agent with external tool servers.

This module is optional: it only imports the official ``mcp`` SDK when an MCP
server is configured. Install the ``mcp`` extra to use it.
"""
from __future__ import annotations

import logging
from typing import Any

from arc_llama.agent.tools import TOOLS, Tool, ToolResult
from arc_llama.config import MCPServerConfig

log = logging.getLogger("arc_llama.agent.mcp_client")


try:
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
    from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

    _MCP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency may be missing
    _MCP_AVAILABLE = False
    _MCP_IMPORT_ERROR = exc


class _MCPToolProxy:
    """Callable that forwards a tool call to an MCP session."""

    def __init__(self, session: Any, original_name: str) -> None:
        self.session = session
        self.original_name = original_name

    async def __call__(self, arguments: dict[str, Any], ctx: Any) -> ToolResult:
        result = await self.session.call_tool(self.original_name, arguments=arguments)
        parts = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(item))
        return ToolResult("\n".join(parts))


class MCPClientManager:
    """Loads configured MCP servers and registers their tools."""

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers
        self._clients: list[Any] = []
        self._sessions: list[Any] = []
        self._registered_tools: list[str] = []

    async def start(self) -> None:
        if not self.servers:
            return
        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "MCP support requires the 'mcp' extra. "
                "Install with: pip install 'arc-llama[mcp]'"
            ) from _MCP_IMPORT_ERROR

        try:
            for cfg in self.servers:
                if not cfg.name or not cfg.command:
                    log.warning("Skipping invalid MCP server config: %s", cfg)
                    continue
                params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env or None,
                )
                client_ctx = stdio_client(params)
                read, write = await client_ctx.__aenter__()
                self._clients.append(client_ctx)
                session = ClientSession(read, write)
                await session.__aenter__()
                self._sessions.append(session)
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    prefixed = f"mcp_{cfg.name}_{tool.name}"
                    self._registered_tools.append(prefixed)
                    TOOLS.register(
                        Tool(
                            name=prefixed,
                            description=f"[MCP:{cfg.name}] {tool.description}",
                            parameters=tool.inputSchema,
                            handler=_MCPToolProxy(session, tool.name),
                            requires_confirmation=True,
                            is_async=True,
                        )
                    )
                    log.info("Registered MCP tool %s from server %s", prefixed, cfg.name)
        except BaseException:
            # SDK stdio/session contexts must unwind in the same task that
            # entered them, including when initialize or tool discovery fails.
            await self.stop()
            raise

    async def stop(self) -> None:
        for name in self._registered_tools:
            TOOLS.unregister(name)
        self._registered_tools.clear()
        for session in reversed(self._sessions):
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                log.warning("Error closing MCP session: %s", e)
        self._sessions.clear()
        for client in reversed(self._clients):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                log.warning("Error closing MCP client: %s", e)
        self._clients.clear()
