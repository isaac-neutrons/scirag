"""LLM service abstraction layer for scirag."""

import logging
import os
from typing import Protocol

import ollama
from dotenv import load_dotenv

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

    async def generate_response(self, messages: list[dict]) -> str:
        """Generate a response from the LLM based on the provided messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            str: The generated response content from the LLM.

        Raises:
            Exception: If the LLM service fails to generate a response.
        """
        ...


class OllamaService:
    """Ollama LLM service implementation.

    This service uses the Ollama API to generate responses from local LLM models.
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
        logger.debug("✅ Ollama client configured")

    async def generate_response(self, messages: list[dict]) -> str:
        """Generate a response using Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            str: The generated response content from the model.
        """
        logger.info(f"🗣️  Generating response with {self.model}")
        logger.debug(f"Messages: {len(messages)} messages")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_preview = msg.get("content", "")[:100]
            logger.debug(f"  Message {i + 1} ({role}): {content_preview}...")

        # Use the synchronous chat method from ollama
        # In a real async context, you might want to use asyncio.to_thread
        # to avoid blocking the event loop
        try:
            logger.debug(f"Calling Ollama API at {self.host}...")
            response = self.client.chat(model=self.model, messages=messages)
            logger.debug("✅ Ollama API call successful")

            # Extract the message content from the response
            content = response["message"]["content"]
            logger.info(f"✅ Response generated: {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}", exc_info=True)
            raise


def get_llm_service(config: dict | None = None) -> LLMService:
    """Factory function to create an LLM service instance.

    Args:
        config: Optional configuration dictionary. If None, uses environment variables.
                Expected keys:
                - 'service': Service type (default: "ollama")
                - 'host': Ollama host URL (default: from OLLAMA_HOST env)
                - 'model': Model name (default: from OLLAMA_MODEL env)

    Returns:
        LLMService: An instance implementing the LLMService protocol.

    Raises:
        ValueError: If an unsupported service type is specified.
    """
    if config is None:
        config = {}

    service_type = config.get("service", "ollama")

    if service_type == "ollama":
        host = config.get("host", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model = config.get("model", os.getenv("OLLAMA_MODEL", "llama3"))
        return OllamaService(host=host, model=model)

    raise ValueError(f"Unsupported service type: {service_type}")
