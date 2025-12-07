"""Google Gemini LLM service implementation."""

import logging
from typing import Any

from google import genai

from scirag.constants import MAX_TOOL_ITERATIONS, get_embedding_model
from scirag.llm.base import MCPToolMixin

logger = logging.getLogger(__name__)


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
