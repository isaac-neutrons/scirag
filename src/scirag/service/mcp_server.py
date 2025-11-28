"""FastMCP server for document retrieval using vector search."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from scirag.service.database import get_collections, search_documents, store_chunks_with_embeddings

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
        f"MCP Tool: Parameters - query='{query[:100]}...', "
        f"top_k={top_k}, collection={collection}"
    )

    try:
        results = search_documents(query=query, top_k=top_k, collection=collection)
        logger.info(
            f"✅ MCP Tool: Returning {len(results)} formatted results to MCP client"
        )
        return results
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e


@mcp.tool()
async def store_document_chunks(
    chunks: list[dict[str, Any]], collection: str = "DocumentChunks"
) -> dict[str, Any]:
    """
    Stores document chunks in the vectorstore database. Generates embeddings
    for each chunk's text and stores them with their metadata.

    Use this tool to add new documents to the knowledge base after they have
    been read and chunked by the client.

    Args:
        chunks: List of chunk dictionaries, each containing:
            - text: The text content of the chunk (required)
            - source_filename: Original document filename (required)
            - chunk_index: Index of this chunk in the document (required)
            - metadata: Optional dict with additional metadata
        collection: Collection name to store chunks in (default: "DocumentChunks")

    Returns:
        dict with:
            - success: bool indicating if storage was successful
            - chunks_stored: number of chunks successfully stored
            - collection: the collection name used
            - message: descriptive message about the operation
    """
    logger.info(
        f"📥 MCP Tool store_document_chunks: "
        f"Storing {len(chunks)} chunks in collection '{collection}'"
    )

    try:
        # Validate chunks have required fields
        for i, chunk in enumerate(chunks):
            if "text" not in chunk:
                raise ValueError(f"Chunk {i} missing required field 'text'")
            if "source_filename" not in chunk:
                raise ValueError(f"Chunk {i} missing required field 'source_filename'")
            if "chunk_index" not in chunk:
                raise ValueError(f"Chunk {i} missing required field 'chunk_index'")

        # Store chunks with embeddings
        chunks_stored = store_chunks_with_embeddings(
            chunks=chunks, collection=collection
        )

        logger.info(
            f"✅ MCP Tool: Successfully stored {chunks_stored} chunks "
            f"in collection '{collection}'"
        )

        return {
            "success": True,
            "chunks_stored": chunks_stored,
            "collection": collection,
            "message": f"Successfully stored {chunks_stored} chunks",
        }

    except ValueError as e:
        error_msg = f"Validation error: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}")
        return {
            "success": False,
            "chunks_stored": 0,
            "collection": collection,
            "message": error_msg,
        }
    except Exception as e:
        error_msg = f"Storage error: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        return {
            "success": False,
            "chunks_stored": 0,
            "collection": collection,
            "message": error_msg,
        }


def main() -> None:
    """Entry point for the MCP server command-line interface."""
    logger.info("🚀 Starting SciRAG MCP Server...")
    mcp.run(transport="sse", host="0.0.0.0", port=8001)


@mcp.tool()
async def list_collections() -> list[str]:
    """
    Lists all available document collections in the vectorstore.

    Use this tool to discover what collections exist before querying
    or storing documents.

    Returns:
        List of collection names as strings
    """
    logger.info("📂 MCP Tool list_collections: Fetching available collections")

    try:
        collections = get_collections()
        logger.info(f"✅ MCP Tool: Found {len(collections)} collections")
        return collections
    except Exception as e:
        error_msg = f"Error fetching collections: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        return []


if __name__ == "__main__":
    main()
