"""FastMCP server for document retrieval using vector search."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from scirag.service.database import search_documents

# Configure logging
log_level = os.getenv("LOG_LEVEL", "DEBUG")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.debug("Environment variables loaded for MCP server")

# Create FastMCP instance
mcp = FastMCP("SciRAG Document Retrieval")


@mcp.tool()
async def retrieve_document_chunks(
    query: str, top_k: int = 5, collection: str | None = None
) -> list[dict[str, Any]]:
    """
    Searches the document knowledge base for text chunks that are semantically
    similar to the user's query. Returns the top_k most relevant chunks.
    Use this tool to find information to answer a user's question.

    Args:
        query: The search query text
        top_k: Number of top results to return (default: 5)
        collection: Optional collection name to filter by (None = search all)
    """
    logger.debug(
        f"MCP Tool: Parameters - query='{query[:100]}...', top_k={top_k}, collection={collection}"
    )

    try:
        results = search_documents(query=query, top_k=top_k, collection=collection)
        logger.info(f"✅ MCP Tool: Returning {len(results)} formatted results to MCP client")
        return results
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e


def main() -> None:
    """Entry point for the MCP server command-line interface."""
    logger.info("🚀 Starting SciRAG MCP Server...")
    mcp.run(transport="sse", host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
