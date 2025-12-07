"""Base classes and protocols for LLM services."""

import logging
from typing import Any, Protocol

from fastmcp import Client as MCPClient

logger = logging.getLogger(__name__)


class LLMService(Protocol):
    """Protocol defining the interface for LLM services.

    This protocol ensures type safety and allows for multiple LLM provider
    implementations while maintaining a consistent interface.
    """

    async def generate_response(
        self, messages: list[dict], mcp_servers: list[str] | None = None
    ) -> str:
        """Generate a response from the LLM based on the provided messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]
            mcp_servers: Optional list of MCP server URLs for tool use.
                        If provided, the LLM can call tools from these servers.

        Returns:
            str: The generated response content from the LLM.
        """
        ...

    def generate_embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            model: Optional embedding model name. If None, uses a default for the service.

        Returns:
            list[list[float]]: List of embedding vectors
        """
        ...


class MCPToolMixin:
    """Mixin class providing MCP tool discovery and execution functionality.

    This mixin provides shared functionality for discovering and calling tools
    from MCP servers. It can be used by any LLM service that wants to support
    MCP tool calling.
    """

    async def discover_mcp_tools(
        self, mcp_servers: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[str, Any]]]:
        """Discover tools from MCP servers.

        Args:
            mcp_servers: List of MCP server URLs

        Returns:
            Tuple of (tools_list, tool_registry) where:
            - tools_list: List of tools in a generic format
            - tool_registry: Dict mapping tool name to (server_url, mcp_tool) tuple
        """
        tools_list: list[dict[str, Any]] = []
        tool_registry: dict[str, tuple[str, Any]] = {}

        for server_url in mcp_servers:
            try:
                mcp_client = MCPClient(server_url)
                async with mcp_client:
                    tools = await mcp_client.list_tools()
                    if tools:
                        for tool in tools:
                            # Store tool in a generic format
                            tool_info = {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema
                                or {"type": "object", "properties": {}},
                            }
                            tools_list.append(tool_info)
                            tool_registry[tool.name] = (server_url, tool)
                            logger.debug(f"  Registered tool: {tool.name} from {server_url}")
                        logger.info(f"🔧 Discovered {len(tools)} tools from {server_url}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to discover tools from {server_url}: {e}")

        return tools_list, tool_registry

    async def call_mcp_tool(
        self, server_url: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Call a tool on an MCP server.

        Args:
            server_url: The MCP server URL
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            str: The tool result as a string
        """
        try:
            mcp_client = MCPClient(server_url)
            async with mcp_client:
                result = await mcp_client.call_tool(tool_name, arguments)
                # Extract content from CallToolResult
                if hasattr(result, "content") and result.content:
                    if isinstance(result.content, list) and result.content:
                        return result.content[0].text
                    return str(result.content)
                return str(result)
        except Exception as e:
            logger.error(f"❌ Failed to call tool {tool_name}: {e}")
            return f"Error calling tool: {e}"
