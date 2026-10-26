"""MCP client support for extending the agent with external tool servers.

This module is optional: it only imports the official ``mcp`` SDK when an MCP
server is configured. Install the ``mcp`` extra to use it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
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


REQUEST_TIMEOUT = timedelta(seconds=60)
"""Startup, discovery and other requests; a server that never answers fails instead of hanging."""
CALL_TIMEOUT = timedelta(seconds=300)
"""One tool call. A timeout does not mean the tool did nothing, so the call is never retried."""
MAX_TOOL_PAGES = 50


def _describe_content(item: Any) -> str:
    """One MCP content block as text the model can use, without dumping binary payloads."""
    kind = getattr(item, "type", "")
    text = getattr(item, "text", None)
    if text is not None:
        return str(text)
    if kind in ("image", "audio"):
        data = getattr(item, "data", "") or ""
        return f"[{kind} {getattr(item, 'mimeType', '') or 'unknown type'}, about {len(data) * 3 // 4} bytes]"
    resource = getattr(item, "resource", None)
    if resource is not None:
        inner = getattr(resource, "text", None)
        uri = getattr(resource, "uri", "")
        return str(inner) if inner is not None else f"[resource {uri} ({getattr(resource, 'mimeType', '') or 'binary'})]"
    if kind == "resource_link":
        return f"[resource link {getattr(item, 'uri', '')}]"
    return f"[{kind or 'unknown'} content]"


def _structured_in_text(structured: Any, content: list[Any]) -> bool:
    """True when a text block already carries *structured*, so it isn't sent twice.

    FastMCP wraps a tool's non-object return value as ``{"result": value}`` in
    ``structuredContent`` while the text block holds the bare value, so the
    wrapped value counts as represented too.
    """
    candidates = [structured]
    if isinstance(structured, dict) and list(structured) == ["result"]:
        candidates.append(structured["result"])
    canonical = {json.dumps(c, sort_keys=True, default=str) for c in candidates}
    for item in content:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        if any(isinstance(c, str) and c == text for c in candidates):
            return True
        try:
            if json.dumps(json.loads(text), sort_keys=True, default=str) in canonical:
                return True
        except (ValueError, TypeError):
            continue
    return False


class _MCPToolProxy:
    """Callable that forwards a tool call to an MCP session."""

    def __init__(self, session: Any, original_name: str, is_current: Callable[[], bool] | None = None) -> None:
        self.session = session
        self.original_name = original_name
        self.is_current = is_current

    async def __call__(self, arguments: dict[str, Any], ctx: Any) -> ToolResult:
        if self.is_current is not None and not self.is_current():
            return ToolResult("MCP tool catalog changed; rediscover the tool before calling it.", error=True)
        result = await self.session.call_tool(
            self.original_name, arguments=arguments, read_timeout_seconds=CALL_TIMEOUT
        )
        parts = [_describe_content(item) for item in (result.content or [])]
        structured = getattr(result, "structuredContent", None)
        if structured is not None and not _structured_in_text(structured, result.content or []):
            parts.append(json.dumps(structured, indent=2, default=str))
        images = tuple(
            f"data:{item.mimeType};base64,{item.data}"
            for item in (result.content or [])
            if getattr(item, "type", "") == "image" and getattr(item, "data", None)
        )
        # A tool that reports an error stays an error for the agent.
        return ToolResult("\n".join(parts), error=bool(getattr(result, "isError", False)), images=images)


async def _list_all_tools(session: Any) -> list[Any]:
    """Every page of tools/list, stopping on a repeated cursor or the page cap."""
    tools: list[Any] = []
    seen: set[str] = set()
    cursor: str | None = None
    for _ in range(MAX_TOOL_PAGES):
        page = await (session.list_tools(cursor) if cursor else session.list_tools())
        tools.extend(page.tools)
        cursor = getattr(page, "nextCursor", None)
        if not cursor:
            return tools
        if cursor in seen:
            log.warning("MCP server repeated tool-list cursor %r; using the tools listed so far", cursor)
            return tools
        seen.add(cursor)
    log.warning("MCP server listed more than %d pages of tools; using the first %d", MAX_TOOL_PAGES, MAX_TOOL_PAGES)
    return tools


class MCPClientManager:
    """Loads configured MCP servers and registers their tools."""

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers
        self._clients: list[Any] = []
        self._sessions: list[Any] = []
        self._registered_tools: list[str] = []
        self._session_tools: dict[Any, dict[str, Tool]] = {}
        self._versions: dict[Any, int] = {}
        self._refresh_tasks: dict[Any, asyncio.Task[None]] = {}
        self._closing = False

    def _forget_tools(self, session: Any) -> None:
        for name, owned in self._session_tools.pop(session, {}).items():
            if TOOLS.get(name) is owned:
                TOOLS.unregister(name)
            if name in self._registered_tools:
                self._registered_tools.remove(name)

    def _register_tools(self, cfg: MCPServerConfig, session: Any, tools: list[Any], version: int) -> None:
        self._forget_tools(session)
        owned: dict[str, Tool] = {}
        self._session_tools[session] = owned
        for tool in tools:
            prefixed = f"mcp_{cfg.name}_{tool.name}"
            if not tool.name or TOOLS.get(prefixed) is not None:
                log.warning("Skipping MCP tool %r from %s: empty or duplicate name", tool.name, cfg.name)
                continue
            def current(s: Any = session, v: int = version) -> bool:
                return not self._closing and self._versions.get(s) == v
            proxy = Tool(
                name=prefixed,
                description=f"[MCP:{cfg.name}] {tool.description}",
                parameters=tool.inputSchema,
                handler=_MCPToolProxy(session, tool.name, current),
                requires_confirmation=True,
                is_async=True,
            )
            owned[prefixed] = proxy
            self._registered_tools.append(prefixed)
            TOOLS.register(proxy)

    def _message_handler(self, cfg: MCPServerConfig, holder: dict[str, Any]) -> Callable[[Any], Awaitable[None]]:
        async def handle(message: Any) -> None:
            if self._closing or getattr(getattr(message, "root", None), "method", None) != "notifications/tools/list_changed":
                return
            session = holder["session"]
            self._versions[session] = self._versions.get(session, 0) + 1
            self._forget_tools(session)
            if session not in self._refresh_tasks:
                # The SDK dispatches notifications from its receive loop. A
                # separate request task lets that loop receive tools/list's reply.
                self._refresh_tasks[session] = asyncio.create_task(self._refresh(cfg, session))
        return handle

    async def _refresh(self, cfg: MCPServerConfig, session: Any) -> None:
        try:
            for _ in range(3):
                version = self._versions[session]
                tools = await asyncio.wait_for(_list_all_tools(session), timeout=REQUEST_TIMEOUT.total_seconds())
                if self._closing:
                    return
                if version == self._versions[session]:
                    self._register_tools(cfg, session, tools, version)
                    return
            log.warning("MCP tool catalog for %s kept changing; tools remain unavailable", cfg.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Could not refresh MCP tools for %s; tools remain unavailable: %s", cfg.name, exc)
        finally:
            self._refresh_tasks.pop(session, None)

    async def start(self) -> None:
        self._closing = False
        if not self.servers:
            return
        if not _MCP_AVAILABLE:
            raise RuntimeError(
                "MCP support requires the 'mcp' extra. Install with: pip install 'arc-llama[mcp]'"
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
                holder: dict[str, Any] = {}
                session = ClientSession(read, write, read_timeout_seconds=REQUEST_TIMEOUT,
                                        message_handler=self._message_handler(cfg, holder))
                holder["session"] = session
                self._versions[session] = 0
                await session.__aenter__()
                self._sessions.append(session)
                await session.initialize()
                version = self._versions[session]
                listed = await asyncio.wait_for(_list_all_tools(session), timeout=REQUEST_TIMEOUT.total_seconds())
                if version == self._versions[session]:
                    self._register_tools(cfg, session, listed, version)
        except BaseException:
            # SDK stdio/session contexts must unwind in the same task that
            # entered them, including when initialize or tool discovery fails.
            await self.stop()
            raise

    async def stop(self) -> None:
        self._closing = True
        tasks = list(self._refresh_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._refresh_tasks.clear()
        for session in list(self._session_tools):
            self._forget_tools(session)
        self._versions.clear()
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
