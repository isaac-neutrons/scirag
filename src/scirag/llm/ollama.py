"""Ollama LLM service implementation."""

import logging
from typing import Any

import ollama

from scirag.constants import MAX_TOOL_ITERATIONS, get_embedding_model
from scirag.llm.base import MCPToolMixin

logger = logging.getLogger(__name__)


class OllamaService(MCPToolMixin):
    """Ollama LLM service implementation.

    This service uses the Ollama API to generate responses from local LLM models.
    Supports MCP tool servers for agentic workflows.
    """

    def __init__(self, host: str, model: str) -> None:
        """Initialize the Ollama service.

        Args:
            host: The Ollama server host URL (e.g., "http://localhost:11434")
            model: The model name to use (e.g., "llama3")
        """
        self.host = host
        self.model = model
        logger.info(f"🤖 Initializing OllamaService: host={host}, model={model}")
        # Configure the Ollama client with the specified host
        self.client = ollama.Client(host=host)

    def _convert_tools_to_ollama_format(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert generic tools to Ollama format.

        Args:
            tools: List of tools in generic format

        Returns:
            List of tools in Ollama format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

    async def generate_response(
        self, messages: list[dict], mcp_servers: list[str] | None = None
    ) -> str:
        """Generate a response using Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            mcp_servers: Optional list of MCP server URLs for tool use.
                        If provided, tools from these MCP servers will be available to Ollama.

        Returns:
            str: The generated response content from the model.
        """
        logger.info(f"🗣️  Generating response with {self.model}")
        logger.debug(f"Messages: {len(messages)} messages")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_preview = msg.get("content", "")[:100]
            logger.debug(f"  Message {i + 1} ({role}): {content_preview}...")

        try:
            # Discover tools from MCP servers if provided
            ollama_tools: list[dict[str, Any]] = []
            tool_registry: dict[str, tuple[str, Any]] = {}

            if mcp_servers:
                logger.info(f"🔧 Discovering tools from {len(mcp_servers)} MCP servers...")
                tools_list, tool_registry = await self.discover_mcp_tools(mcp_servers)
                ollama_tools = self._convert_tools_to_ollama_format(tools_list)
                logger.info(f"🔧 Total tools available: {len(ollama_tools)}")

            # Make the initial chat request
            chat_kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if ollama_tools:
                chat_kwargs["tools"] = ollama_tools

            response = self.client.chat(**chat_kwargs)

            # Handle tool calls in a loop
            iteration = 0
            working_messages = list(messages)

            while response.message.tool_calls and iteration < MAX_TOOL_ITERATIONS:
                iteration += 1
                tool_count = len(response.message.tool_calls)
                logger.info(f"🔧 Processing {tool_count} tool calls (iteration {iteration})")

                # Add the assistant's message with tool calls to the conversation
                working_messages.append(response.message.model_dump())

                # Process each tool call
                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    logger.info(f"  📞 Calling tool: {tool_name}")
                    logger.debug(f"     Arguments: {tool_args}")

                    if tool_name in tool_registry:
                        server_url, _ = tool_registry[tool_name]
                        tool_result = await self.call_mcp_tool(server_url, tool_name, tool_args)
                        logger.debug(f"     Result: {tool_result[:200]}...")
                    else:
                        tool_result = f"Error: Unknown tool '{tool_name}'"
                        logger.warning(f"  ⚠️ Unknown tool: {tool_name}")

                    # Add tool result to messages
                    working_messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_name": tool_name,
                    })

                # Make another chat request with tool results
                chat_kwargs["messages"] = working_messages
                response = self.client.chat(**chat_kwargs)

            if iteration >= MAX_TOOL_ITERATIONS:
                logger.warning("⚠️ Reached maximum tool call iterations")

            # Extract the final message content
            content = response.message.content or ""
            logger.info(f"✅ Response generated: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}", exc_info=True)
            raise

    def generate_embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts using Ollama.

        Args:
            texts: List of text strings to embed
            model: Optional embedding model name. If None, uses EMBEDDING_MODEL env var
                   or service-specific default.

        Returns:
            list[list[float]]: List of embedding vectors
        """
        embedding_model = model or get_embedding_model("ollama")
        embeddings = []

        for text in texts:
            response = self.client.embed(model=embedding_model, input=text)
            embeddings.append(response["embeddings"][0])

        logger.info(f"✅ Generated {len(embeddings)} embeddings with {embedding_model}")
        return embeddings
