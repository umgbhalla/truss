"""Async singleton manager providing pooled MCP sessions.

The real MCP integration will establish transport layers (e.g. stdio, WebSocket
or HTTP) to *remote* tool servers.  For now we expose a minimal API surface so
that revised activity implementations can be exercised by the existing unit
suite while the broader migration is completed.

Public helpers
--------------
- :func:`default_manager` – obtain the shared :class:`MCPClientManager`.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Dict

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


from truss.data_models import MCPClientConfig, MCPServerConfig


__all__ = [ 
    "MCPClientManager",
    "default_manager",
]


class MCPClientManager: 
    _instance: "MCPClientManager | None" = None

    def __new__(cls):  
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_singleton() 
        return cls._instance

   
    def _init_singleton(self) -> None: 
        self._exit_stack = AsyncExitStack()
        self._sessions: Dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()
        self._config: MCPClientConfig = MCPClientConfig()
        
    async def get_session(self, mcp_server_config: MCPServerConfig) -> ClientSession: 
        """Return an existing or new session for *server_name*.

        The real implementation will lazily establish the transport and cache
        the resulting client for reuse across activities/workflows.
        """
        assert mcp_server_config is not None
        server_name = mcp_server_config.name
        async with self._lock:
            if server_name not in self._sessions:
                stdio_server_params = StdioServerParameters(
                    command=mcp_server_config.command,
                    args=mcp_server_config.args,
                    env=mcp_server_config.env,
                    description=mcp_server_config.description,
                )

                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(stdio_server_params)
                )
                stdio, write = stdio_transport
                session = await self._exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

                await session.initialize()
                self._sessions[server_name] = session
                
            return self._sessions[server_name]

    async def aclose(self) -> None: 
        await self._exit_stack.aclose()
        self._sessions.clear()


default_manager = MCPClientManager()
