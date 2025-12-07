"""LLM service abstraction layer for scirag.

This package provides a unified interface for multiple LLM providers:
- OllamaService: Local LLM via Ollama
- GeminiService: Google Gemini API

All services implement the LLMService protocol and support MCP tool calling.

Usage:
    from scirag.llm import get_llm_service, LLMService

    # Create service from environment config
    service = get_llm_service()

    # Or with explicit config
    service = get_llm_service({"service": "gemini", "model": "gemini-2.5-flash"})
"""

# Re-export public API for backward compatibility
from scirag.llm.base import LLMService, MCPToolMixin
from scirag.llm.factory import get_llm_service
from scirag.llm.gemini import GeminiService
from scirag.llm.ollama import OllamaService

__all__ = [
    "LLMService",
    "MCPToolMixin",
    "OllamaService",
    "GeminiService",
    "get_llm_service",
]
