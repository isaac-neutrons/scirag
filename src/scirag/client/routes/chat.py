"""Chat API routes using RAG pipeline."""

import logging

from flask import Blueprint, jsonify, request

from scirag.service.mcp_helpers import call_mcp_tool, run_async

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

# These will be set by app.py during initialization
llm_service = None
local_mcp_server_url = None
mcp_tool_servers = []


def init_chat_routes(llm, mcp_url, tool_servers):
    """Initialize chat routes with service dependencies.

    Args:
        llm: LLM service instance
        mcp_url: Local MCP server URL
        tool_servers: List of MCP tool server URLs
    """
    global llm_service, local_mcp_server_url, mcp_tool_servers
    llm_service = llm
    local_mcp_server_url = mcp_url
    mcp_tool_servers = tool_servers


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


@chat_bp.route("/api/chat", methods=["POST"])
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

        # 1. Call Retrieval Tool via MCP helper
        logger.info("📡 Calling MCP retrieval tool...")
        tool_params = {"query": user_query, "top_k": top_k}
        if collection:
            tool_params["collection"] = collection

        retrieved_chunks = run_async(
            call_mcp_tool(local_mcp_server_url, "retrieve_document_chunks", tool_params)
        )
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
        llm_response = run_async(
            llm_service.generate_response(messages, mcp_servers=mcp_tool_servers)
        )
        logger.info("✅ Response generated")

        # 5. Return response with sources
        logger.info("✅ Chat request completed successfully")
        return jsonify({"response": llm_response, "sources": retrieved_chunks})

    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
