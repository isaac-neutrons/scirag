"""Health check and status API routes."""

import logging

from flask import Blueprint, jsonify

from scirag.service.mcp_helpers import check_mcp_server, run_async

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)

# These will be set by app.py during initialization
llm_service = None
local_mcp_server_url = None
mcp_tool_servers = []


def init_health_routes(llm, mcp_url, tool_servers):
    """Initialize health routes with service dependencies.

    Args:
        llm: LLM service instance
        mcp_url: Local MCP server URL
        tool_servers: List of MCP tool server URLs
    """
    global llm_service, local_mcp_server_url, mcp_tool_servers
    llm_service = llm
    local_mcp_server_url = mcp_url
    mcp_tool_servers = tool_servers


@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint.

    Returns:
        JSON with service status
    """
    return jsonify(
        {
            "status": "healthy",
            "llm_service": "initialized" if llm_service else "not initialized",
        }
    )


@health_bp.route("/api/mcp-status", methods=["GET"])
def get_mcp_status():
    """Get status of all configured MCP servers.

    Returns:
        JSON response with connected and failed MCP servers
    """
    logger.info("🔌 Checking MCP server status...")

    connected_servers = []
    failed_servers = []

    # Check local MCP server
    if local_mcp_server_url:
        result = run_async(check_mcp_server(local_mcp_server_url))
        result["name"] = result.get("server_name") or "Local Document Server"
        result["type"] = "local"
        if result["status"] == "connected":
            connected_servers.append(result)
        else:
            failed_servers.append(result)

    # Check tool MCP servers
    for url in mcp_tool_servers:
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
            "total_configured": len(mcp_tool_servers) + (1 if local_mcp_server_url else 0),
        }
    )
