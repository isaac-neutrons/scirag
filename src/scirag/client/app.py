"""Flask web application for RAG-based chat interface.

This module provides a REST API endpoint for answering user queries using
Retrieval-Augmented Generation (RAG). It orchestrates document retrieval
via an MCP server and response generation via an LLM service.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from scirag.client.routes import (
    chat_bp,
    health_bp,
    init_config,
    pages_bp,
    upload_bp,
)
from scirag.constants import DEFAULT_LOCAL_MCP_URL, DEFAULT_OLLAMA_HOST, MAX_UPLOAD_SIZE_BYTES
from scirag.llm import get_llm_service

# Configure logging
log_level = os.getenv("LOG_LEVEL", "DEBUG")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.debug("Environment variables loaded")

# Create Flask app
app = Flask(__name__)
logger.debug("Flask app created")

# Configure upload settings
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "/tmp/scirag_uploads"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Register blueprints
app.register_blueprint(pages_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(health_bp)


def initialize_services():
    """Initialize LLM service and MCP server URL on startup."""
    logger.info("🔧 Initializing services...")

    # Initialize LLM service
    llm_config = {
        "service": os.getenv("LLM_SERVICE", "ollama"),
        "host": os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        "model": os.getenv("LLM_MODEL", "llama3"),
    }
    logger.debug(f"LLM config: {llm_config}")
    llm_service = get_llm_service(llm_config)
    logger.info("✅ LLM service initialized successfully")

    # Store local MCP server URL
    local_mcp_server_url = os.getenv("LOCAL_MCP_SERVER_URL", DEFAULT_LOCAL_MCP_URL)
    logger.info(f"✅ Local MCP server URL configured: {local_mcp_server_url}")

    # Load MCP tool servers for LLM tool use
    mcp_tool_servers_env = os.getenv("MCP_TOOL_SERVERS", "")
    if mcp_tool_servers_env:
        mcp_tool_servers = [url.strip() for url in mcp_tool_servers_env.split(",") if url.strip()]
        logger.info(f"✅ MCP tool servers configured: {mcp_tool_servers}")
    else:
        mcp_tool_servers = []
        logger.info("ℹ️ No MCP tool servers configured for LLM tool use")

    # Initialize route configuration
    init_config(
        llm_service=llm_service,
        local_mcp_server_url=local_mcp_server_url,
        mcp_tool_servers=mcp_tool_servers,
        upload_folder=UPLOAD_FOLDER,
    )


def create_app():
    """Factory function for creating the Flask application.

    This function is used by WSGI servers like gunicorn to create the app.
    It initializes services before returning the app instance.

    Returns:
        Flask: The configured Flask application instance
    """
    initialize_services()
    return app


def main() -> None:
    """Entry point for the Flask application command-line interface."""
    print("🚀 Starting SciRAG Flask application...")

    # Initialize services
    print("📦 Initializing LLM services...")
    initialize_services()
    print("✅ Services initialized successfully")

    # Run Flask app
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_ENV", "development") == "development"

    print(f"🌐 Starting Flask server on http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print("📝 Press CTRL+C to quit")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
