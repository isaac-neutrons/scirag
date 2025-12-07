"""Database configuration and connection management for RavenDB.

This package provides a unified interface for RavenDB operations:
- Configuration management (RavenDBConfig)
- Document store creation and index management
- CRUD operations (create, delete, count)
- Vector search operations
- Chunk storage with embedding generation

Usage:
    from scirag.service.database import (
        RavenDBConfig,
        create_document_store,
        search_documents,
        store_chunks_with_embeddings,
    )
"""

# Re-export public API
from scirag.service.database.config import RavenDBConfig
from scirag.service.database.operations import (
    count_documents,
    create_database,
    create_document_store,
    database_exists,
    delete_database,
    ensure_index_exists,
    get_collections,
    search_documents,
)
from scirag.service.database.storage import store_chunks_with_embeddings
from scirag.service.database.utils import cosine_similarity

__all__ = [
    # Config
    "RavenDBConfig",
    # Operations
    "create_document_store",
    "ensure_index_exists",
    "database_exists",
    "create_database",
    "delete_database",
    "count_documents",
    "get_collections",
    "search_documents",
    # Storage
    "store_chunks_with_embeddings",
    # Utils
    "cosine_similarity",
]
