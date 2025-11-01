"""Database configuration and connection management for RavenDB."""

import os

from dotenv import load_dotenv
from pyravendb.store.document_store import DocumentStore

# Load environment variables
load_dotenv()


class RavenDBConfig:
    """Configuration class for RavenDB connection details."""

    @staticmethod
    def get_url() -> str:
        """Get the RavenDB server URL from environment variables.

        Returns:
            str: RavenDB server URL (default: http://localhost:8080)
        """
        return os.getenv("RAVENDB_URL", "http://localhost:8080")

    @staticmethod
    def get_database_name() -> str:
        """Get the RavenDB database name from environment variables.

        Returns:
            str: Database name (default: scirag)
        """
        return os.getenv("RAVENDB_DATABASE", "scirag")


def create_document_store(
    url: str | None = None, database: str | None = None
) -> DocumentStore:
    """Create and initialize a DocumentStore instance.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        DocumentStore: Initialized DocumentStore instance

    Example:
        >>> store = create_document_store()
        >>> with store.open_session() as session:
        ...     # Use session for database operations
        ...     pass
        >>> store.close()
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    store = DocumentStore(url, database)
    store.initialize()
    return store


def ensure_index_exists(store: DocumentStore) -> None:
    """Ensure the vector search index exists in RavenDB.

    Creates a static index named 'DocumentChunks/ByEmbedding' for vector search
    on the embedding field of DocumentChunks collection.

    Args:
        store: Initialized DocumentStore instance

    Example:
        >>> store = get_document_store()
        >>> ensure_index_exists(store)
    """
    # Note: This is a simplified implementation.
    # In a real RavenDB setup, you would use the RavenDB management API
    # to create indexes. For now, we'll create a basic structure.
    _ = "DocumentChunks/ByEmbedding"  # Index name for future implementation

    # Index creation would typically be done through RavenDB Studio or
    # using the management API. This function serves as a placeholder
    # for ensuring the index exists before using it.

    # In production, you would implement something like:
    # - Check if index exists using store operations
    # - Create index definition with vector field
    # - Deploy index to RavenDB

    pass  # Placeholder for now, will be implemented with actual RavenDB API

