"""FastMCP server for document retrieval using vector search."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP

from scirag.service.database import get_sources, search_documents

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


async def retrieve_document_chunks_impl(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Implementation of document chunk retrieval with vector search.

    This function performs semantic search over ingested document chunks using
    vector embeddings and RavenDB's vector search functionality.

    Args:
        query: The search query text
        top_k: Number of top results to return (default: 5)

    Returns:
        list[dict]: List of relevant chunks, each containing:
            - source: Original document filename
            - content: Text content of the chunk
            - chunk_index: Position of chunk in document
            - score: Similarity score (0.0 if not available)
            - metadata: Dict containing document metadata

    Example:
        >>> results = await retrieve_document_chunks_impl("quantum entanglement", top_k=3)
        >>> for result in results:
        ...     print(f"Source: {result['source']}")
        ...     print(f"Content: {result['content'][:100]}...")
    """
    logger.info(f"🔍 MCP: Retrieving document chunks for query: '{query[:100]}...'")
    logger.debug(f"MCP: top_k={top_k}")

    try:
        logger.info("📊 MCP: Calling search_documents from database module...")
        results = search_documents(query=query, top_k=top_k)
        logger.info(f"✅ MCP: Retrieved {len(results)} results from vector search")
        return results
    except ConnectionError as e:
        logger.error(f"❌ MCP: Connection error: {e}", exc_info=True)
        raise
    except ValueError as e:
        logger.error(f"❌ MCP: Value error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ MCP: Unexpected error during search: {e}", exc_info=True)
        raise


@mcp.tool()
async def retrieve_document_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Searches the document knowledge base for text chunks that are semantically
    similar to the user's query. Returns the top_k most relevant chunks.
    Use this tool to find information to answer a user's question.
    """
    logger.info("🔌 MCP Tool: retrieve_document_chunks called via MCP protocol")
    logger.debug(f"MCP Tool: Parameters - query='{query[:100]}...', top_k={top_k}")

    try:
        logger.debug("MCP Tool: Calling retrieve_document_chunks_impl...")
        results = await retrieve_document_chunks_impl(query, top_k)
        logger.info(
            f"✅ MCP Tool: Returning {len(results)} "
            "formatted results to MCP client"
        )
        return results
    except ConnectionError as e:
        error_msg = f"Connection error: {e} - Check if Ollama and RavenDB are running"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e
    except AttributeError as e:
        error_msg = f"Attribute error: {e} - This may indicate a version mismatch"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e


@mcp.tool()
async def list_document_sources() -> list[dict[str, Any]]:
    """
    Lists all unique document sources in the knowledge base with their metadata.
    Returns information about each document including filename, metadata, and chunk count.
    Use this tool to see what documents are available in the system.
    """
    logger.info("🔌 MCP Tool: list_document_sources called via MCP protocol")

    try:
        logger.debug("MCP Tool: Calling get_sources from database...")
        sources = get_sources()
        logger.info(f"✅ MCP Tool: Returning {len(sources)} sources to MCP client")
        return sources
    except ConnectionError as e:
        error_msg = f"Connection error: {e} - Check if RavenDB is running"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        logger.error(f"❌ MCP Tool: {error_msg}", exc_info=True)
        raise ValueError(error_msg) from e


def main() -> None:
    """Entry point for the MCP server command-line interface."""
    logger.info("🚀 Starting SciRAG MCP Server...")
    logger.info("📡 MCP Server: Listening on STDIO transport")
    logger.info("🔧 MCP Server: Ready to receive tool calls")
    logger.info("🔌 MCP Server: Tools available:")
    logger.info("   - retrieve_document_chunks: Search for relevant document chunks")
    logger.info("   - list_document_sources: List all available documents")

    try:
        logger.info("✅ MCP Server: Initialization complete")
        # Use SSE transport for HTTP connectivity
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        logger.info("⚠️  MCP Server: Received shutdown signal")
        logger.info("👋 MCP Server: Shutting down gracefully")
    except Exception as e:
        logger.error(f"❌ MCP Server: Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
