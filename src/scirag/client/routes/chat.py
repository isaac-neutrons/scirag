"""Chat API routes using RAG pipeline."""

import logging

from flask import Blueprint, jsonify, request

from scirag.client.routes.config import get_config
from scirag.service.mcp_helpers import call_mcp_tool, run_async

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)


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

    Expects JSON body with 'query' field and optional 'messages' for conversation history.
    Returns JSON with 'response' field.

    Request:
        {
            "query": "What is quantum entanglement?",
            "messages": [  # Optional conversation history
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"}
            ],
            "session_id": "uuid",  # Optional session identifier
            "top_k": 5,  # Optional, default 5
            "collection": "papers"  # Optional collection filter
        }

    Response:
        {
            "response": "Based on the documents, quantum entanglement is...",
            "sources": [
                {"source": "paper.pdf", "chunk_index": 0, "content": "..."},
                ...
            ],
            "session_id": "uuid"
        }

    Returns:
        JSON response with answer, sources, and session ID
    """
    config = get_config()
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
        session_id = data.get("session_id", None)
        conversation_history = data.get("messages", [])

        logger.info(f"🔍 Query: '{user_query[:100]}...'")
        logger.info(f"📁 Collection filter: {collection or 'All collections'}")
        logger.info(f"💬 Conversation history: {len(conversation_history)} messages")

        # 1. Call Retrieval Tool via MCP helper
        logger.info("📡 Calling MCP retrieval tool...")
        tool_params = {"query": user_query, "top_k": top_k}
        if collection:
            tool_params["collection"] = collection

        retrieved_chunks = run_async(
            call_mcp_tool(config.local_mcp_server_url, "retrieve_document_chunks", tool_params)
        )
        logger.info(f"✅ Retrieved {len(retrieved_chunks)} chunks")

        # 2. Format context from retrieved chunks
        context = format_context(retrieved_chunks)

        # 3. Construct prompt with context and conversation history
        system_prompt = (
            "You are an expert assistant. Your task is to answer the user's question based "
            "ONLY on the context provided below. Do not use any outside knowledge. "
            "If the answer cannot be found in the context, state that clearly. "
            "Cite the source filename for the information you use."
        )

        # Build messages: system prompt + context + conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add context as the first user message
        messages.append(
            {
                "role": "user",
                "content": f"Here is the context from the document database:\n---\n{context}\n---",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "I understand. I'll answer your questions based only on this context.",
            }
        )

        # Add conversation history (excluding the current query which is already in history)
        # The history includes the current message, so we use all of it
        for msg in conversation_history:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # 4. Call LLM Service to generate response
        logger.info(f"🤖 Generating response from LLM with {len(messages)} messages...")
        llm_response = run_async(
            config.llm_service.generate_response(messages, mcp_servers=config.mcp_tool_servers)
        )
        logger.info("✅ Response generated")

        # 5. Return response with sources and session ID
        logger.info("✅ Chat request completed successfully")
        response_data = {"response": llm_response, "sources": retrieved_chunks}
        if session_id:
            response_data["session_id"] = session_id
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
