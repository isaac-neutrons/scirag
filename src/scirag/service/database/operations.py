"""Database operations for RavenDB - CRUD, indexing, and queries."""

import os
from typing import Any

import requests
from ravendb import DocumentStore
from ravendb.documents.indexes.definitions import (
    FieldIndexing,
    FieldStorage,
    IndexDefinition,
    IndexFieldOptions,
)
from ravendb.documents.indexes.vector.options import VectorOptions
from ravendb.documents.operations.indexes import GetIndexNamesOperation, PutIndexesOperation
from ravendb.serverwide.operations.common import DeleteDatabaseOperation

from scirag.constants import DEFAULT_EMBEDDING_DIMENSIONS, get_embedding_model
from scirag.llm import get_llm_service
from scirag.service.database.config import RavenDBConfig
from scirag.service.database.utils import cosine_similarity


def create_document_store(url: str | None = None, database: str | None = None) -> DocumentStore:
    """Create and initialize a DocumentStore instance.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        DocumentStore: Initialized DocumentStore instance
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

    Args:
        store: Initialized DocumentStore instance
    """
    index_name = "DocumentChunks/ByEmbedding"

    # Check if index already exists
    existing_indexes = store.maintenance.send(GetIndexNamesOperation(0, 100))
    if index_name in existing_indexes:
        return  # Index already exists

    # Create index definition with vector search support
    index_definition = IndexDefinition()
    index_definition.name = index_name

    index_definition.maps = {
        """from chunk in docs
        where chunk.embedding != null
        select new {
            source_filename = chunk.source_filename,
            chunk_index = chunk.chunk_index,
            text = chunk.text,
            collection = chunk.collection,
            embedding = CreateField("embedding", chunk.embedding, new CreateFieldOptions { Storage = FieldStorage.Yes, Indexing = FieldIndexing.No })
        }"""
    }

    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS)))
    vector_options = VectorOptions(dimensions=dimensions)

    index_definition.fields = {
        "embedding": IndexFieldOptions(
            storage=FieldStorage.YES, indexing=FieldIndexing.NO, vector=vector_options
        )
    }

    store.maintenance.send(PutIndexesOperation(index_definition))


def database_exists(url: str | None = None, database: str | None = None) -> bool:
    """Check if a database exists in RavenDB.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        bool: True if database exists, False otherwise
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    try:
        store = DocumentStore([url], database)
        store.initialize()
        with store.open_session() as session:
            list(session.query().take(0))
        store.close()
        return True
    except Exception:
        return False


def create_database(url: str | None = None, database: str | None = None) -> None:
    """Create a new database in RavenDB.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

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
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

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
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    store = DocumentStore([url], database)
    store.initialize()

    try:
        with store.open_session() as session:
            rql_query = "from DocumentChunks"
            results = list(session.advanced.raw_query(rql_query, object_type=dict))
            return len(results)
    finally:
        store.close()


def get_collections(url: str | None = None, database: str | None = None) -> list[str]:
    """Get a list of unique collection names from the database.

    Uses RavenDB's REST API to fetch collection statistics and extract collection names.

    Args:
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        list[str]: Sorted list of unique collection names
    """
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    try:
        response = requests.get(f"{url}/databases/{database}/collections/stats", timeout=10)
        response.raise_for_status()

        data = response.json()
        collections = list(data.get("Collections", {}).keys())
        return sorted(collections)

    except requests.RequestException:
        return []


def search_documents(
    query: str,
    top_k: int = 5,
    collection: str | None = None,
    embedding_model: str | None = None,
    url: str | None = None,
    database: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar documents using vector search.

    Args:
        query: The text query to search for
        top_k: Number of top results to return (default: 5)
        collection: Collection name to filter by (None or empty = search all collections)
        embedding_model: embedding model name
            (defaults to EMBEDDING_MODEL env or service-specific default)
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        list[dict]: List of documents with their content and metadata.
    """
    # Get defaults
    if embedding_model is None:
        embedding_model = get_embedding_model()
    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    # Generate query embedding
    llm_service = os.getenv("LLM_SERVICE", "ollama")
    service = get_llm_service({"service": llm_service})
    query_embedding = service.generate_embeddings([query], embedding_model)[0]

    # Query RavenDB with vector search
    store = DocumentStore([url], database)
    store.initialize()

    try:
        with store.open_session() as session:
            # Determine which collection(s) to search
            if collection and collection.strip():
                collections_to_search = [collection.strip()]
            else:
                all_collections = get_collections(url, database)
                collections_to_search = all_collections if all_collections else ["DocumentChunks"]

            # Query each collection and collect results
            all_results = []
            for coll in collections_to_search:
                try:
                    query_base = session.query_collection(coll, object_type=dict)
                    coll_results = list(
                        query_base.vector_search("embedding", query_embedding)
                        .order_by_score()
                        .take(top_k)
                    )
                    all_results.extend(coll_results)
                except Exception:
                    continue

            # Compute scores for results
            def get_score(result: dict) -> float:
                metadata = result.get("@metadata", {})
                index_score = metadata.get("@index-score")
                if index_score is not None:
                    return float(index_score)
                result_embedding = result.get("embedding", [])
                if result_embedding:
                    return cosine_similarity(query_embedding, result_embedding)
                return 0.0

            # Sort all results by score and take top_k
            all_results.sort(key=get_score, reverse=True)
            results = all_results[:top_k]

        # Format the results
        formatted_results = []
        for result in results:
            metadata = result.get("@metadata", {})
            index_score = metadata.get("@index-score")
            if index_score is not None:
                score = float(index_score)
            else:
                result_embedding = result.get("embedding", [])
                score = cosine_similarity(query_embedding, result_embedding)

            formatted_results.append(
                {
                    "source": result.get("source_filename", "Unknown"),
                    "content": result.get("text", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "score": score,
                    "metadata": result.get("metadata", {}),
                    "collection": metadata.get(
                        "@collection", result.get("collection", "DocumentChunks")
                    ),
                }
            )

        return formatted_results

    finally:
        store.close()
