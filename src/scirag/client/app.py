"""Flask web application for RAG-based chat interface.

This module provides a REST API endpoint for answering user queries using
Retrieval-Augmented Generation (RAG). It orchestrates document retrieval
via an MCP server and response generation via an LLM service.
"""

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from scirag.service.database import get_sources
from scirag.service.llm_services import get_llm_service
from scirag.service.mcp_server import retrieve_document_chunks_impl

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


def initialize_services():
    """Initialize LLM service on startup."""
    global llm_service
    logger.info("🔧 Initializing LLM services...")

    # Initialize LLM service
    llm_config = {
        "service": "ollama",
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3"),
    }
    logger.debug(f"LLM config: {llm_config}")
    llm_service = get_llm_service(llm_config)
    logger.info("✅ LLM service initialized successfully")


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved document chunks into a context string.

    Args:
        chunks: List of document chunks with 'source', 'content', 'chunk_index' keys

    Returns:
        Formatted context string with sources and content
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")
        chunk_idx = chunk.get("chunk_index", 0)
        context_parts.append(
            f"[Source {i}: {source}, Chunk {chunk_idx}]\n{content}\n"
        )

    return "\n".join(context_parts)


@app.route("/")
def index():
    """Serve the chat interface web page.

    Returns:
        HTML page with interactive chat interface
    """
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests using RAG pipeline.

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
                {"source": "paper.pdf", "chunk_index": 0},
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

        query = data["query"]
        top_k = data.get("top_k", 5)
        logger.info(f"🔍 Query: '{query[:100]}...' (top_k={top_k})")

        # Retrieve relevant document chunks via direct function call
        # Run async function in sync context
        logger.debug("🔄 Creating event loop for document retrieval...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("📚 Retrieving document chunks...")
            chunks = loop.run_until_complete(
                retrieve_document_chunks_impl(query, top_k)
            )
            logger.info(f"✅ Retrieved {len(chunks)} document chunks")
            logger.debug(f"Chunks: {[c.get('source', 'Unknown') for c in chunks]}")
        finally:
            loop.close()

        # Format context from retrieved chunks
        logger.debug("📝 Formatting context from chunks...")
        context = format_context(chunks)
        logger.debug(f"Context length: {len(context)} characters")

        # Create system prompt
        logger.debug("🎯 Creating system prompt...")
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based ONLY "
            "on the provided context from the knowledge base.\n\n"
            "If the context doesn't contain enough information to answer the question, "
            "say so clearly. Do not make up information or use knowledge outside the "
            "provided context.\n\n"
            "Always cite the sources you use in your answer."
        )

        # Construct messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        logger.debug(f"Constructed {len(messages)} messages for LLM")

        # Generate response using LLM service (sync call)
        logger.info("🤖 Generating LLM response...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(llm_service.generate_response(messages))
            logger.info(f"✅ LLM response generated ({len(response)} chars)")
            logger.debug(f"Response preview: {response[:200]}...")
        finally:
            loop.close()

        # Extract source information
        sources = [
            {
                "source": chunk.get("source", "Unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            }
            for chunk in chunks
        ]
        logger.debug(f"Extracted {len(sources)} sources")

        logger.info("✅ Chat request completed successfully")
        return jsonify({"response": response, "sources": sources})

    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/sources", methods=["GET"])
def sources():
    """Get list of unique document sources with metadata.

    Returns:
        JSON with list of sources, each containing:
        - source: filename
        - metadata: document metadata dict
        - chunk_count: number of chunks
    """
    logger.info("📊 API: Fetching sources list")

    try:
        sources_list = get_sources()
        logger.info(f"✅ API: Returning {len(sources_list)} sources")
        return jsonify({"sources": sources_list})

    except ConnectionError as e:
        logger.error(f"❌ API: Connection error: {e}", exc_info=True)
        return jsonify({"error": f"Cannot connect to database: {str(e)}"}), 503

    except Exception as e:
        logger.error(f"❌ API: Error fetching sources: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


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
