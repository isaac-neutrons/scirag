"""Database configuration and connection management for RavenDB."""

import os
from typing import Any

import ollama
from dotenv import load_dotenv
from ravendb import DocumentStore

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


def create_document_store(url: str | None = None, database: str | None = None) -> DocumentStore:
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

    store = DocumentStore([url], database)
    store.initialize()
    return store


def ensure_index_exists(store: DocumentStore) -> None:
    """Ensure the vector search index exists in RavenDB.

    Creates a static index named 'DocumentChunks/ByEmbedding' with vector search
    capabilities for the embedding field.

    This function uses the official ravendb package (v7.1.2+) which supports
    native vector search with:
    - Cosine similarity search on embeddings
    - Efficient KNN (K-Nearest Neighbors) queries
    - Vector distance calculations

    Args:
        store: Initialized DocumentStore instance

    Raises:
        Exception: If index creation fails

    Example:
        >>> store = create_document_store()
        >>> ensure_index_exists(store)
        >>> store.close()
    """
    from ravendb.documents.indexes.definitions import IndexDefinition, IndexFieldOptions
    from ravendb.documents.indexes.vector.options import VectorOptions
    from ravendb.documents.operations.indexes import GetIndexNamesOperation

    index_name = "DocumentChunks/ByEmbedding"

    try:
        # Check if index already exists
        existing_indexes = store.maintenance.send(
            GetIndexNamesOperation(0, 100)
        )
        if index_name in existing_indexes:
            return  # Index already exists
    except Exception:
        # If we can't check, try to create anyway
        pass

    # Create index definition with vector search support
    index_definition = IndexDefinition()
    index_definition.name = index_name
    
    # Use CreateField to explicitly create the vector field in the map
    # This tells RavenDB this is a vector field for KNN search
    index_definition.maps = {
        """from chunk in docs.DocumentChunks
        select new {
            source_filename = chunk.source_filename,
            chunk_index = chunk.chunk_index,
            text = chunk.text,
            embedding = CreateField("embedding", chunk.embedding, new CreateFieldOptions { Storage = FieldStorage.Yes, Indexing = FieldIndexing.No })
        }"""
    }

    # Configure the embedding field as a vector field with 768 dimensions
    from ravendb.documents.indexes.definitions import FieldIndexing, FieldStorage

    vector_options = VectorOptions(dimensions=768)

    index_definition.fields = {
        "embedding": IndexFieldOptions(
            storage=FieldStorage.YES,
            indexing=FieldIndexing.NO,
            vector=vector_options
        )
    }

    # Create the index
    from ravendb.documents.operations.indexes import PutIndexesOperation

    store.maintenance.send(PutIndexesOperation(index_definition))


def database_exists(url: str | None = None, database: str | None = None) -> bool:
    """Check if a database exists in RavenDB.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        bool: True if database exists, False otherwise

    Example:
        >>> if not database_exists():
        ...     create_database()
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # Try to create a store and check if we can access the database
    try:
        store = DocumentStore([url], database)
        store.initialize()
        # Try to access the database by getting stats
        with store.open_session() as session:
            # Attempt a simple query to verify database exists
            list(session.query().take(0))
        store.close()
        return True
    except Exception:
        # Database doesn't exist or connection failed
        return False


def create_database(url: str | None = None, database: str | None = None) -> None:
    """Create a new database in RavenDB.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Raises:
        Exception: If database creation fails

    Example:
        >>> create_database()
    """
    import requests

    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # Use RavenDB HTTP API to create database
    api_url = f"{url}/admin/databases"
    payload = {"DatabaseName": database, "Settings": {}, "Disabled": False}

    response = requests.put(api_url, json=payload)
    response.raise_for_status()


def delete_database(url: str | None = None, database: str | None = None) -> None:
    """Delete a database from RavenDB.

    WARNING: This operation is irreversible and will delete all data in the database.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Raises:
        Exception: If database deletion fails

    Example:
        >>> delete_database()
    """
    from ravendb.serverwide.operations.common import DeleteDatabaseOperation

    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # Use RavenDB's DeleteDatabaseOperation with hard_delete=True
    store = DocumentStore([url], database)
    try:
        store.initialize()
        operation = DeleteDatabaseOperation(database_name=database, hard_delete=True)
        store.maintenance.server.send(operation)
    finally:
        store.close()


def count_documents(url: str | None = None, database: str | None = None) -> int:
    """Count the number of DocumentChunk documents in the database.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        int: Number of DocumentChunk documents in the database

    Raises:
        Exception: If database connection fails or query fails

    Example:
        >>> count = count_documents()
        >>> print(f"Database contains {count} document chunks")
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    store = DocumentStore([url], database)
    store.initialize()

    try:
        with store.open_session() as session:
            # Use raw RQL query to get count from metadata
            # Query for documents in the DocumentChunks collection
            rql_query = "from DocumentChunks"
            results = list(session.advanced.raw_query(rql_query, object_type=dict))
            return len(results)
    finally:
        store.close()


def search_documents(
    query: str,
    top_k: int = 5,
    embedding_model: str | None = None,
    url: str | None = None,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar documents using vector search.

    Args:
        query: The text query to search for
        top_k: Number of top results to return (default: 5)
        embedding_model: Ollama embedding model name
            (defaults to OLLAMA_EMBEDDING_MODEL env or 'nomic-embed-text')
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        list[dict]: List of documents with their content and metadata.
            Each dict contains:
            - source: Source filename
            - content: Text content of the chunk
            - chunk_index: Index of the chunk in the source document
            - score: Similarity score (0.0 if not available from RavenDB)
            - metadata: Dict containing document metadata (file_size, creation_date, etc.)

    Raises:
        ConnectionError: If cannot connect to Ollama or RavenDB
        ValueError: If invalid response from Ollama

    Example:
        >>> results = search_documents("quantum mechanics", top_k=3)
        >>> for result in results:
        ...     print(f"{result['source']}: {result['content'][:50]}...")
    """
    # Get defaults
    if embedding_model is None:
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # 1. Generate query embedding
    try:
        response = ollama.embed(model=embedding_model, input=query)
        query_embedding = response["embeddings"][0]
    except ConnectionError as e:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        error_msg = f"Cannot connect to Ollama at {ollama_host} - Is Ollama running?"
        raise ConnectionError(error_msg) from e
    except KeyError as e:
        error_msg = f"Invalid response from Ollama: {e} - Check if model '{embedding_model}' exists"
        raise ValueError(error_msg) from e

    # 2. Query RavenDB with vector search
    store = DocumentStore([url], database)
    store.initialize()

    try:
        with store.open_session() as session:
            # Use vector search to find similar documents
            # Query the collection (not the index directly) to let RavenDB auto-select the vector index
            # The vector_search() performs KNN search using the embedding field configured in the index
            results = list(
                session.query_collection("DocumentChunks", object_type=dict)
                .vector_search("embedding", query_embedding)
                .order_by_score()
                .take(top_k)
            )

        # 3. Format the results
        formatted_results = [
            {
                "source": result.get("source_filename", "Unknown"),
                "content": result.get("text", ""),
                "chunk_index": result.get("chunk_index", 0),
                "score": result.get("@metadata", {}).get("@index-score", 0.0),
                "metadata": result.get("metadata", {}),
            }
            for result in results
        ]

        return formatted_results

    finally:
        store.close()


def get_sources(
    url: str | None = None,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Get list of unique document sources with their metadata.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        list[dict]: List of unique sources, each containing:
            - source: Source filename
            - metadata: Dict containing document metadata (file_size, creation_date, etc.)
            - chunk_count: Number of chunks for this source

    Raises:
        ConnectionError: If cannot connect to RavenDB

    Example:
        >>> sources = get_sources()
        >>> for source in sources:
        ...     print(f"{source['source']}: {source['chunk_count']} chunks")
    """
    # Get defaults
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    store = create_document_store(url, database)

    try:
        with store.open_session() as session:
            # Query all documents to get unique sources
            # We'll group by source_filename in Python since RavenDB queries are limited
            all_chunks = list(
                session.query_collection("DocumentChunks", object_type=dict)
            )

        # Group by source and collect metadata
        sources_dict = {}
        for chunk in all_chunks:
            source = chunk.get("source_filename", "Unknown")
            if source not in sources_dict:
                sources_dict[source] = {
                    "source": source,
                    "metadata": chunk.get("metadata", {}),
                    "chunk_count": 0
                }
            sources_dict[source]["chunk_count"] += 1

        # Convert to list and sort by source filename
        sources_list = sorted(sources_dict.values(), key=lambda x: x["source"])

        return sources_list

    except Exception as e:
        ravendb_url = url
        error_msg = f"Error fetching sources from RavenDB at {ravendb_url}: {e}"
        raise ConnectionError(error_msg) from e
    finally:
        store.close()
