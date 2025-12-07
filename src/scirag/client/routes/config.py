"""Shared configuration for route modules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RouteConfig:
    """Configuration container for Flask route dependencies.

    This replaces global variables with a proper configuration object
    that can be passed around and tested more easily.
    """

    llm_service: Any = None
    local_mcp_server_url: str | None = None
    mcp_tool_servers: list[str] = field(default_factory=list)
    upload_folder: Path | None = None
    allowed_extensions: set[str] = field(default_factory=lambda: {"pdf"})


# Single shared config instance
_config = RouteConfig()


def get_config() -> RouteConfig:
    """Get the shared route configuration.

    Returns:
        RouteConfig instance with current settings
    """
    return _config


def init_config(
    llm_service: Any = None,
    local_mcp_server_url: str | None = None,
    mcp_tool_servers: list[str] | None = None,
    upload_folder: Path | None = None,
) -> None:
    """Initialize the shared route configuration.

    Args:
        llm_service: LLM service instance
        local_mcp_server_url: Local MCP server URL
        mcp_tool_servers: List of MCP tool server URLs
        upload_folder: Path to upload folder
    """
    if llm_service is not None:
        _config.llm_service = llm_service
    if local_mcp_server_url is not None:
        _config.local_mcp_server_url = local_mcp_server_url
    if mcp_tool_servers is not None:
        _config.mcp_tool_servers = mcp_tool_servers
    if upload_folder is not None:
        _config.upload_folder = upload_folder
