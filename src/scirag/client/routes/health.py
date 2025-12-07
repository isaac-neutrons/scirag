"""Health check and status API routes."""

import logging

from flask import Blueprint, jsonify

from scirag.client.routes.config import get_config
from scirag.service.mcp_helpers import check_mcp_server, run_async

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns:
        JSON with service status
    """
    config = get_config()
    return jsonify(
        {
            "status": "healthy",
            "llm_service": "initialized" if config.llm_service else "not initialized",
        }
    )


@health_bp.route("/api/mcp-status", methods=["GET"])
def get_mcp_status():
    """Get status of all configured MCP servers.

    Returns:
        JSON response with connected and failed MCP servers
    """
    config = get_config()
    logger.info("🔌 Checking MCP server status...")

    connected_servers = []
    failed_servers = []

    # Check local MCP server
    if config.local_mcp_server_url:
        result = run_async(check_mcp_server(config.local_mcp_server_url))
        result["name"] = result.get("server_name") or "Local Document Server"
        result["type"] = "local"
        if result["status"] == "connected":
            connected_servers.append(result)
        else:
            failed_servers.append(result)

    # Check tool MCP servers
    for url in config.mcp_tool_servers:
        result = run_async(check_mcp_server(url))
        fallback_name = f"Tool Server ({url.split('/')[-2] if '/' in url else url})"
        result["name"] = result.get("server_name") or fallback_name
        result["type"] = "tool"
        if result["status"] == "connected":
            connected_servers.append(result)
        else:
            failed_servers.append(result)

    logger.info(f"✅ Connected: {len(connected_servers)}, Failed: {len(failed_servers)}")
    return jsonify(
        {
            "connected": connected_servers,
            "failed": failed_servers,
            "total_configured": 1 + len(config.mcp_tool_servers),
        }
    )
