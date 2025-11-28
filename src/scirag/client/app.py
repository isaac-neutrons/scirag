"""Flask web application for RAG-based chat interface.

This module provides a REST API endpoint for answering user queries using
Retrieval-Augmented Generation (RAG). It orchestrates document retrieval
via an MCP server and response generation via an LLM service.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from scirag.client.ingest import extract_chunks_from_pdf
from scirag.llm.providers import get_llm_service

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
ALLOWED_EXTENSIONS = {"pdf"}
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Global service instances
llm_service = None
local_mcp_server_url = None
mcp_tool_servers = []


def initialize_services():
    """Initialize LLM service and MCP server URL on startup."""
    global llm_service, local_mcp_server_url, mcp_tool_servers
    logger.info("🔧 Initializing services...")

    # Initialize LLM service
    llm_config = {
        "service": os.getenv("LLM_SERVICE", "ollama"),
        "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "model": os.getenv("LLM_MODEL", "llama3"),
    }
    logger.debug(f"LLM config: {llm_config}")
    llm_service = get_llm_service(llm_config)
    logger.info("✅ LLM service initialized successfully")

    # Store local MCP server URL (client created per-request)
    local_mcp_server_url = os.getenv("LOCAL_MCP_SERVER_URL", "http://localhost:8001/sse")
    logger.info(f"✅ Local MCP server URL configured: {local_mcp_server_url}")

    # Load MCP tool servers for LLM tool use
    mcp_tool_servers_env = os.getenv("MCP_TOOL_SERVERS", "")
    if mcp_tool_servers_env:
        mcp_tool_servers = [url.strip() for url in mcp_tool_servers_env.split(",") if url.strip()]
        logger.info(f"✅ MCP tool servers configured: {mcp_tool_servers}")
    else:
        mcp_tool_servers = []
        logger.info("ℹ️ No MCP tool servers configured for LLM tool use")


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed.

    Args:
        filename: The filename to check

    Returns:
        True if extension is allowed, False otherwise
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the chat interface web page.

    Returns:
        HTML page with interactive chat interface
    """
    return render_template("chat.html")


@app.route("/upload")
def upload_page():
    """Serve the document upload page.

    Returns:
        HTML page with drag-and-drop upload interface
    """
    return render_template("upload.html")


@app.route("/api/collections", methods=["GET"])
def list_collections_endpoint():
    """Get list of existing collection names.

    Returns:
        JSON response with list of collection names
    """
    logger.info("📂 Fetching collection names")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def get_collections_via_mcp():
                client = Client(local_mcp_server_url)
                async with client:
                    result = await client.call_tool("list_collections", {})
                    if hasattr(result, "content") and result.content:
                        if isinstance(result.content, list):
                            return json.loads(result.content[0].text)
                        return result.content
                    return result

            collections = loop.run_until_complete(get_collections_via_mcp())
        finally:
            loop.close()

        logger.info(f"✅ Found {len(collections)} collections")
        return jsonify({"success": True, "collections": collections})
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch collections: {e}")
        # Return empty list if database doesn't exist or other error
        return jsonify({"success": True, "collections": []})


@app.route("/api/upload", methods=["POST"])
def upload_documents():
    """Handle document upload and ingestion into vectorstore.

    Expects multipart form data with:
        - files: One or more PDF files
        - collection: Name of the collection to store documents in

    Response:
        {
            "success": true,
            "message": "Successfully ingested X documents",
            "details": [
                {"filename": "doc.pdf", "chunks": 10, "status": "success"},
                ...
            ]
        }

    Returns:
        JSON response with upload status
    """
    logger.info("📤 Received document upload request")

    try:
        # Check if files were provided
        if "files" not in request.files:
            logger.warning("❌ No files in request")
            return jsonify({"success": False, "error": "No files provided"}), 400

        files = request.files.getlist("files")
        collection = request.form.get("collection", "default")

        if not files or all(f.filename == "" for f in files):
            logger.warning("❌ No files selected")
            return jsonify({"success": False, "error": "No files selected"}), 400

        logger.info(f"📁 Collection: {collection}")
        logger.info(f"📄 Files received: {len(files)}")

        results = []
        success_count = 0

        for file in files:
            if file.filename == "":
                continue

            if not allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "File type not allowed. Only PDF files are accepted."
                })
                continue

            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                filepath = UPLOAD_FOLDER / filename
                file.save(filepath)
                logger.info(f"💾 Saved file: {filepath}")

                # Extract chunks from PDF (without embeddings)
                chunks = extract_chunks_from_pdf(filepath, collection)
                logger.info(f"📊 Extracted {len(chunks)} chunks from {filename}")

                # Store chunks via MCP tool (handles embedding generation)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    async def store_via_mcp():
                        client = Client(local_mcp_server_url)
                        async with client:
                            result = await client.call_tool(
                                "store_document_chunks",
                                {"chunks": chunks, "collection": collection}
                            )
                            if hasattr(result, "content") and result.content:
                                if isinstance(result.content, list):
                                    return json.loads(result.content[0].text)
                                return result.content
                            return result

                    store_result = loop.run_until_complete(store_via_mcp())
                finally:
                    loop.close()

                if store_result.get("success"):
                    logger.info(
                        f"✅ Stored {store_result.get('chunks_stored', 0)} chunks "
                        f"for {filename} in collection '{collection}'"
                    )
                    results.append({
                        "filename": filename,
                        "chunks": store_result.get("chunks_stored", len(chunks)),
                        "status": "success"
                    })
                    success_count += 1
                else:
                    error_msg = store_result.get("message", "Unknown error")
                    logger.error(f"❌ Failed to store chunks: {error_msg}")
                    results.append({
                        "filename": filename,
                        "status": "error",
                        "error": error_msg
                    })

                # Clean up temporary file
                filepath.unlink()

            except Exception as e:
                logger.error(f"❌ Error processing {file.filename}: {e}", exc_info=True)
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })

        return jsonify({
            "success": success_count > 0,
            "message": f"Successfully ingested {success_count} of {len(files)} documents",
            "collection": collection,
            "details": results
        })

    except Exception as e:
        logger.error(f"❌ Error in upload handler: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


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
        collection = data.get("collection", None)  # None = search all collections
        logger.info(f"🔍 Query: '{user_query[:100]}...'")
        logger.info(f"📁 Collection filter: {collection or 'All collections'}")

        # Run async operations in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 1. Call Retrieval Tool via FastMCP client
            logger.info("📡 Calling MCP retrieval tool...")

            async def get_chunks():
                client = Client(local_mcp_server_url)
                async with client:
                    tool_params = {"query": user_query, "top_k": top_k}
                    if collection:
                        tool_params["collection"] = collection
                    result = await client.call_tool(
                        "retrieve_document_chunks", tool_params
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
            llm_response = loop.run_until_complete(
                llm_service.generate_response(messages, mcp_servers=mcp_tool_servers)
            )
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
