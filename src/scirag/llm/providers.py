"""LLM service abstraction layer for scirag."""

import logging
import os
from typing import Any, Protocol

import ollama
from dotenv import load_dotenv
from fastmcp import Client as MCPClient
from google import genai

from scirag.constants import (
    DEFAULT_OLLAMA_HOST,
    MAX_TOOL_ITERATIONS,
    get_embedding_model,
)

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "DEBUG")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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


class GeminiService(MCPToolMixin):
    """Google Gemini LLM service implementation.

    This service uses the Google Gemini API to generate responses from Google's LLM models.
    The API key is automatically retrieved from the GEMINI_API_KEY environment variable.
    Supports MCP tool servers for agentic workflows.
    """

    def __init__(self, model: str) -> None:
        """Initialize the Gemini service.

        Args:
            model: The model name to use (e.g., "gemini-2.5-flash")
        """
        self.model = model
        logger.info(f"🤖 Initializing GeminiService: model={model}")
        # The client gets the API key from the GEMINI_API_KEY environment variable
        self.client = genai.Client()

    def _convert_tools_to_gemini_format(
        self, tools: list[dict[str, Any]]
    ) -> list[genai.types.Tool]:
        """Convert generic tools to Gemini format.

        Args:
            tools: List of tools in generic format

        Returns:
            List of Gemini Tool objects
        """
        function_declarations = []
        for tool in tools:
            # Create function declaration for Gemini
            func_decl = genai.types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
            )
            function_declarations.append(func_decl)

        if function_declarations:
            return [genai.types.Tool(function_declarations=function_declarations)]
        return []

    async def generate_response(
        self, messages: list[dict], mcp_servers: list[str] | None = None
    ) -> str:
        """Generate a response using Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     The Gemini API expects a specific format, so we convert the messages.
            mcp_servers: Optional list of MCP server URLs for tool use.
                        If provided, tools from these MCP servers will be available to Gemini.

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
            gemini_tools: list[genai.types.Tool] = []
            tool_registry: dict[str, tuple[str, Any]] = {}

            if mcp_servers:
                logger.info(f"🔧 Discovering tools from {len(mcp_servers)} MCP servers...")
                tools_list, tool_registry = await self.discover_mcp_tools(mcp_servers)
                gemini_tools = self._convert_tools_to_gemini_format(tools_list)
                logger.info(f"🔧 Total tools available: {len(tools_list)}")

            # Convert messages to Gemini format
            contents = "\n".join([msg.get("content", "") for msg in messages])

            # Build generation config
            generate_kwargs: dict[str, Any] = {
                "model": self.model,
                "contents": contents,
            }
            if gemini_tools:
                generate_kwargs["config"] = genai.types.GenerateContentConfig(
                    tools=gemini_tools,
                )

            response = self.client.models.generate_content(**generate_kwargs)

            # Handle tool calls in a loop
            iteration = 0

            while response.candidates and iteration < MAX_TOOL_ITERATIONS:
                candidate = response.candidates[0]

                # Check if there are function calls in the response
                function_calls = []
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)

                if not function_calls:
                    break  # No more tool calls, exit loop

                iteration += 1
                call_count = len(function_calls)
                logger.info(f"🔧 Processing {call_count} tool calls (iteration {iteration})")

                # Process each function call
                function_responses = []
                for func_call in function_calls:
                    tool_name = func_call.name
                    tool_args = dict(func_call.args) if func_call.args else {}

                    logger.info(f"  📞 Calling tool: {tool_name}")
                    logger.debug(f"     Arguments: {tool_args}")

                    if tool_name in tool_registry:
                        server_url, _ = tool_registry[tool_name]
                        tool_result = await self.call_mcp_tool(server_url, tool_name, tool_args)
                        logger.debug(f"     Result: {tool_result[:200]}...")
                    else:
                        tool_result = f"Error: Unknown tool '{tool_name}'"
                        logger.warning(f"  ⚠️ Unknown tool: {tool_name}")

                    # Build function response
                    function_responses.append(
                        genai.types.Part.from_function_response(
                            name=tool_name,
                            response={"result": tool_result},
                        )
                    )

                # Send function responses back to Gemini
                # Build conversation with tool results
                conversation_contents = [
                    contents,
                    candidate.content,
                    genai.types.Content(parts=function_responses),
                ]

                generate_kwargs["contents"] = conversation_contents
                response = self.client.models.generate_content(**generate_kwargs)

            if iteration >= MAX_TOOL_ITERATIONS:
                logger.warning("⚠️ Reached maximum tool call iterations")

            # Extract the text content from the final response
            content = response.text
            logger.info(f"✅ Response generated: {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}", exc_info=True)
            raise

    def generate_embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts using Gemini.

        Args:
            texts: List of text strings to embed
            model: Optional embedding model name. If None, uses EMBEDDING_MODEL env var
                   or service-specific default.

        Returns:
            list[list[float]]: List of embedding vectors
        """
        embedding_model = model or get_embedding_model("gemini")
        embeddings = []

        for text in texts:
            try:
                response = self.client.models.embed_content(model=embedding_model, contents=[text])
                embeddings.append(response.embeddings[0].values)
            except Exception as e:
                logger.error(f"❌ Gemini embedding error for text: {e}", exc_info=True)
                raise

        logger.info(f"✅ Generated {len(embeddings)} embeddings with {embedding_model}")
        return embeddings


def get_llm_service(config: dict | None = None) -> LLMService:
    """Factory function to create an LLM service instance.

    Args:
        config: Optional configuration dictionary. If None, uses environment variables.
                Expected keys:
                - 'service': Service type (default: from LLM_SERVICE env, or "ollama")
                - 'host': Ollama host URL (default: from OLLAMA_HOST env)
                - 'model': Model name (default: from LLM_MODEL env)

    Returns:
        LLMService: An instance implementing the LLMService protocol.
    """
    if config is None:
        config = {}

    # Read service type from config, then env, then default to ollama
    service_type = config.get("service", os.getenv("LLM_SERVICE", "ollama"))

    if service_type == "ollama":
        host = config.get("host", os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
        model = config.get("model", os.getenv("LLM_MODEL", "llama3"))
        return OllamaService(host=host, model=model)

    if service_type == "gemini":
        model = config.get("model", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
        return GeminiService(model=model)

    raise ValueError(f"Unsupported service type: {service_type}")
