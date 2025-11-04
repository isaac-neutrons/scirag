"""Flask web application for RAG-based chat interface.

This module provides a REST API endpoint for answering user queries using
Retrieval-Augmented Generation (RAG). It orchestrates document retrieval
via an MCP server and response generation via an LLM service.
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastmcp import Client
from flask import Flask, jsonify, render_template, request

from scirag.service.llm_services import get_llm_service

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

# Global service instances
llm_service = None
mcp_client = None


def initialize_services():
    """Initialize LLM service and MCP client on startup."""
    global llm_service, mcp_client
    logger.info("🔧 Initializing services...")

    # Initialize LLM service
    llm_config = {
        "service": "ollama",
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3"),
    }
    logger.debug(f"LLM config: {llm_config}")
    llm_service = get_llm_service(llm_config)
    logger.info("✅ LLM service initialized successfully")

    # Initialize MCP client with server URL
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
    mcp_client = Client(mcp_server_url)
    logger.info(f"✅ MCP client initialized successfully (server: {mcp_server_url})")


@app.route("/")
def index():
    """Serve the chat interface web page.

    Returns:
        HTML page with interactive chat interface
    """
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests using RAG pipeline with MCP server.

    Expects JSON body with 'query' field. Returns JSON with 'response' field.

    Request:
        {
            "query": "What is quantum entanglement?",
            "top_k": 5  # Optional, default 5
        }

    Response:
        {
            "response": "Based on the documents, quantum entanglement is...",
            "sources": [
                {"source": "paper.pdf", "chunk_index": 0, "content": "..."},
                ...
            ]
        }

    Returns:
        JSON response with answer and sources
    """
    logger.info("📨 Received chat request")
    try:
        # Get query from request
        data = request.get_json()
        if not data or "query" not in data:
            logger.warning("❌ Missing 'query' field in request")
            return jsonify({"error": "Missing 'query' field in request"}), 400

        user_query = data["query"]
        top_k = data.get("top_k", 5)
        logger.info(f"🔍 Query: '{user_query[:100]}...'")

        # Run async operations in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Call Retrieval Tool via FastMCP client
            logger.info("📡 Calling MCP retrieval tool...")

            async def get_chunks():
                async with mcp_client:
                    result = await mcp_client.call_tool(
                        "retrieve_document_chunks", {"query": user_query, "top_k": top_k}
                    )
                    # Extract content from CallToolResult
                    # result.content is a list of TextContent objects
                    if hasattr(result, "content") and result.content:
                        # If content is a list of TextContent, get the first one's text
                        if isinstance(result.content, list):
                            # The text field contains the JSON data
                            return json.loads(result.content[0].text)
                        return result.content
                    return result

            retrieved_chunks = loop.run_until_complete(get_chunks())
            logger.info(f"✅ Retrieved {len(retrieved_chunks)} chunks")

            # 2. Format context from retrieved chunks
            context = format_context(retrieved_chunks)

            # 3. Construct prompt with context
            system_prompt = (
                "You are an expert assistant. Your task is to answer the user's question based "
                "ONLY on the context provided below. Do not use any outside knowledge. "
                "If the answer cannot be found in the context, state that clearly. "
                "Cite the source filename for the information you use."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n---\n{context}\n---\n\nQuestion: {user_query}",
                },
            ]

            # 4. Call LLM Service to generate response
            logger.info("🤖 Generating response from LLM...")
            llm_response = loop.run_until_complete(llm_service.generate_response(messages))
            logger.info("✅ Response generated")

            # 5. Return response with sources
            logger.info("✅ Chat request completed successfully")
            return jsonify({"response": llm_response, "sources": retrieved_chunks})

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context string for LLM.

    Args:
        chunks: List of document chunks with metadata

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."

    context_str = ""
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")
        context_str += f"Source: {source}\nContent: {content}\n\n"
    return context_str.strip()


@app.route("/health", methods=["GET"])
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
