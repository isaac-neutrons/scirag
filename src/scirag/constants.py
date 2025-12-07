"""Application-wide constants and defaults for SciRAG.

This module provides a single source of truth for configuration defaults,
magic numbers, and other constants used throughout the application.
"""

import os

# =============================================================================
# File Upload Limits
# =============================================================================
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# =============================================================================
# LLM Settings
# =============================================================================
MAX_TOOL_ITERATIONS = 10  # Maximum iterations for tool calling loops
DEFAULT_TOP_K = 5  # Default number of results for vector search

# =============================================================================
# Display Settings
# =============================================================================
CONTENT_PREVIEW_LENGTH = 200  # Characters to show in content previews

# =============================================================================
# Default URLs and Hosts
# =============================================================================
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LOCAL_MCP_URL = "http://localhost:8001/sse"
DEFAULT_RAVENDB_URL = "http://localhost:8080"
DEFAULT_RAVENDB_DATABASE = "scirag"

# =============================================================================
# Embedding Model Defaults
# =============================================================================
EMBEDDING_DEFAULTS = {
    "ollama": "nomic-embed-text",
    "gemini": "text-embedding-004",
}

# Default embedding dimensions (for RavenDB vector index)
DEFAULT_EMBEDDING_DIMENSIONS = 768


def get_embedding_model(service: str | None = None) -> str:
    """Get the default embedding model for a given LLM service.

    Checks the EMBEDDING_MODEL environment variable first, then falls back
    to service-specific defaults.

    Args:
        service: The LLM service name ("ollama" or "gemini").
                If None, uses LLM_SERVICE env var or defaults to "ollama".

    Returns:
        str: The embedding model name to use.
    """
    # Environment variable takes precedence
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        return env_model

    # Determine service if not provided
    if service is None:
        service = os.getenv("LLM_SERVICE", "ollama")

    # Return service-specific default
    return EMBEDDING_DEFAULTS.get(service, EMBEDDING_DEFAULTS["ollama"])
