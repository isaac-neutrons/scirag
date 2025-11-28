"""LLM service abstraction layer for scirag."""

import logging
import os
from typing import Protocol

import ollama
from dotenv import load_dotenv
from google import genai

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
            response = self.client.chat(model=self.model, messages=messages)

            # Extract the message content from the response
            content = response["message"]["content"]
            logger.info(f"✅ Response generated: {len(content)} characters")
            return content
        except Exception as e:
            logger.error(f"❌ Ollama API error: {e}", exc_info=True)
            raise

    def generate_embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for a list of texts using Ollama.

        Args:
            texts: List of text strings to embed
            model: Optional embedding model name. If None, uses a default model.

        Returns:
            list[list[float]]: List of embedding vectors
        """
        embedding_model = model or "nomic-embed-text"
        embeddings = []

        for text in texts:
            response = self.client.embed(model=embedding_model, input=text)
            embeddings.append(response["embeddings"][0])

        logger.info(f"✅ Generated {len(embeddings)} embeddings with {embedding_model}")
        return embeddings


class GeminiService:
    """Google Gemini LLM service implementation.

    This service uses the Google Gemini API to generate responses from Google's LLM models.
    The API key is automatically retrieved from the GEMINI_API_KEY environment variable.
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

    async def generate_response(self, messages: list[dict]) -> str:
        """Generate a response using Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     The Gemini API expects a specific format, so we convert the messages.

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
            # Convert messages to a format Gemini can use
            # For simplicity, we'll concatenate all messages into a single prompt
            # A more sophisticated implementation might preserve the conversation structure
            contents = "\n".join([msg.get("content", "") for msg in messages])

            response = self.client.models.generate_content(model=self.model, contents=contents)

            # Extract the text content from the response
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
            model: Optional embedding model name. If None, uses 'text-embedding-004'.

        Returns:
            list[list[float]]: List of embedding vectors
        """
        embedding_model = model or "text-embedding-004"
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
                - 'service': Service type (default: "ollama")
                - 'host': Ollama host URL (default: from OLLAMA_HOST env)
                - 'model': Model name (default: from LLM_MODEL env)

    Returns:
        LLMService: An instance implementing the LLMService protocol.
    """
    if config is None:
        config = {}

    service_type = config.get("service", "ollama")

    if service_type == "ollama":
        host = config.get("host", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        model = config.get("model", os.getenv("LLM_MODEL", "llama3"))
        return OllamaService(host=host, model=model)

    if service_type == "gemini":
        model = config.get("model", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
        return GeminiService(model=model)

    raise ValueError(f"Unsupported service type: {service_type}")
