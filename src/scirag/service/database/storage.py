"""Storage operations for storing document chunks with embeddings."""

import os
from typing import Any

from ravendb import DocumentStore

from scirag.constants import get_embedding_model
from scirag.llm import get_llm_service
from scirag.service.database.config import RavenDBConfig
from scirag.service.database.models import DocumentChunk
from scirag.service.database.operations import ensure_index_exists


def store_chunks_with_embeddings(
    chunks: list[dict[str, Any]],
    collection: str = "DocumentChunks",
    embedding_model: str | None = None,
    url: str | None = None,
    database: str | None = None,
) -> int:
    """Generate embeddings for chunks and store them in RavenDB.

    This function takes raw chunk data (without embeddings), generates embeddings
    for each chunk's text, and stores them in the vectorstore database.

    Args:
        chunks: List of chunk dictionaries, each containing:
            - text: The text content of the chunk
            - source_filename: Original document filename
            - chunk_index: Index of this chunk in the document
            - metadata: Optional dict with additional metadata (file_size, etc.)
        collection: Collection name to store chunks in (default: "DocumentChunks")
        embedding_model: Model to use for embeddings
            (defaults to EMBEDDING_MODEL env or service-specific default)
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        int: Number of chunks successfully stored
    """
    if not chunks:
        return 0

    # Get defaults
    if embedding_model is None:
        embedding_model = get_embedding_model()
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # Generate embeddings for all chunk texts
    llm_service = os.getenv("LLM_SERVICE", "ollama")
    service = get_llm_service({"service": llm_service})
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = service.generate_embeddings(texts, embedding_model)

    # Create document store and ensure index exists
    store = DocumentStore([url], database)
    store.initialize()

    try:
        ensure_index_exists(store)

        # Store chunks in a single session
        with store.open_session() as session:
            for i, chunk in enumerate(chunks):
                source_filename = chunk.get("source_filename", "unknown")
                chunk_index = chunk.get("chunk_index", i)
                doc_id = f"{source_filename}_chunk_{chunk_index}"

                # Create DocumentChunk entity for RavenDB
                doc = DocumentChunk(
                    Id=doc_id,
                    source_filename=source_filename,
                    chunk_index=chunk_index,
                    text=chunk.get("text", ""),
                    embedding=embeddings[i],
                    metadata=chunk.get("metadata", {}),
                    collection=collection,
                )

                # Store the document
                session.store(doc, doc_id)

                # Set the collection in document metadata
                metadata = session.advanced.get_metadata_for(doc)
                metadata["@collection"] = collection

            session.save_changes()

        return len(chunks)

    finally:
        store.close()
