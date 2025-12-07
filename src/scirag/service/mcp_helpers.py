"""MCP client utilities for consistent tool calling.

This module provides helper functions to eliminate duplicated MCP client patterns
across the codebase. It handles:
- Consistent result extraction from MCP tool calls
- Event loop management for async operations in sync contexts
- Server status checking
"""

import asyncio
import json
import logging
from typing import Any

from fastmcp import Client as MCPClient

logger = logging.getLogger(__name__)


async def call_mcp_tool(
    server_url: str, tool_name: str, params: dict[str, Any] | None = None
) -> Any:
    """Call an MCP tool and extract the result.

    Handles the common pattern of connecting to an MCP server, calling a tool,
    and extracting the result from the CallToolResult object.

    Args:
        server_url: The MCP server URL (e.g., "http://localhost:8001/sse")
        tool_name: Name of the tool to call
        params: Parameters to pass to the tool (default: empty dict)

    Returns:
        The extracted result from the tool call. If the result contains JSON
        in a TextContent list, it will be parsed and returned as a Python object.

    Raises:
        Exception: If the MCP server connection or tool call fails
    """
    if params is None:
        params = {}

    client = MCPClient(server_url)
    async with client:
        result = await client.call_tool(tool_name, params)
        return extract_mcp_result(result)


def extract_mcp_result(result: Any) -> Any:
    """Extract the content from an MCP CallToolResult.

    Args:
        result: The raw result from an MCP tool call

    Returns:
        The extracted content, parsed from JSON if applicable
    """
    if hasattr(result, "content") and result.content:
        if isinstance(result.content, list) and result.content:
            # TextContent list - parse JSON from first item
            text = result.content[0].text
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
        return result.content
    return result


async def check_mcp_server(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Check if an MCP server is reachable and get its info.

    Args:
        url: The MCP server URL to check
        timeout: Connection timeout in seconds (default 5.0)

    Returns:
        Dict containing:
        - url: The server URL
        - status: "connected" or "failed"
        - tools: List of tool names (if connected)
        - server_name: Server name from protocol (if available)
        - error: Error message (if failed)
    """
    try:
        client = MCPClient(url)
        async with asyncio.timeout(timeout):
            async with client:
                # Try to list tools to verify connection
                tools = await client.list_tools()
                tool_names = [tool.name for tool in tools] if tools else []

                # Get server name from initialize_result if available
                server_name = None
                if client.initialize_result and client.initialize_result.serverInfo:
                    server_name = client.initialize_result.serverInfo.name

                return {
                    "url": url,
                    "status": "connected",
                    "tools": tool_names,
                    "server_name": server_name,
                }
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Timeout connecting to MCP server {url}")
        return {
            "url": url,
            "status": "failed",
            "error": f"Connection timeout ({timeout}s)",
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to connect to MCP server {url}: {e}")
        return {
            "url": url,
            "status": "failed",
            "error": str(e),
        }


def run_async(coro: Any) -> Any:
    """Run an async coroutine in a new event loop.

    This is useful for calling async functions from synchronous Flask routes.
    Creates a new event loop, runs the coroutine, and properly cleans up.

    Args:
        coro: An awaitable coroutine to execute

    Returns:
        The result of the coroutine

    Note:
        For CLI commands, prefer using asyncio.run() directly.
        This helper is mainly for Flask routes that need sync compatibility.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
