"""Factory function for creating LLM service instances."""

import logging
import os

from dotenv import load_dotenv

from scirag.constants import DEFAULT_OLLAMA_HOST
from scirag.llm.base import LLMService
from scirag.llm.gemini import GeminiService
from scirag.llm.ollama import OllamaService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
