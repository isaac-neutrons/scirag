"""Database configuration and connection management for RavenDB."""

import math
import os
from typing import Any

import requests
from dotenv import load_dotenv
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

from scirag.llm.providers import get_llm_service

# Load environment variables
load_dotenv()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


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
    """

    index_name = "DocumentChunks/ByEmbedding"

    # Check if index already exists
    existing_indexes = store.maintenance.send(GetIndexNamesOperation(0, 100))
    if index_name in existing_indexes:
        return  # Index already exists

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

    vector_options = VectorOptions(dimensions=768)

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
    """

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
    """

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
            (defaults to EMBEDDING_MODEL env or 'nomic-embed-text')
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
    """
    # Get defaults
    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
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
            # Use vector search to find similar documents
            # Query the collection to let RavenDB auto-select the vector index
            # vector_search() performs KNN search using the embedding field

            # Determine which collection(s) to search
            if collection and collection.strip():
                # Search specific collection
                collections_to_search = [collection.strip()]
            else:
                # Search all collections - get the list of collections
                all_collections = get_collections(url, database)
                # If no collections exist, default to DocumentChunks
                collections_to_search = all_collections if all_collections else ["DocumentChunks"]

            # Query each collection and collect results
            all_results = []
            for coll in collections_to_search:
                try:
                    query_base = session.query_collection(coll, object_type=dict)
                    coll_results = list(
                        query_base
                        .vector_search("embedding", query_embedding)
                        .order_by_score()
                        .take(top_k)
                    )
                    all_results.extend(coll_results)
                except Exception:
                    # Skip collections that fail (e.g., no embedding field)
                    continue

            # Compute scores for results - use @index-score if available,
            # otherwise compute cosine similarity
            def get_score(result: dict) -> float:
                metadata = result.get("@metadata", {})
                index_score = metadata.get("@index-score")
                if index_score is not None:
                    return float(index_score)
                # Fallback: compute cosine similarity with query embedding
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
            # Get score - prefer @index-score, fallback to computed similarity
            metadata = result.get("@metadata", {})
            index_score = metadata.get("@index-score")
            if index_score is not None:
                score = float(index_score)
            else:
                result_embedding = result.get("embedding", [])
                score = cosine_similarity(query_embedding, result_embedding)

            formatted_results.append({
                "source": result.get("source_filename", "Unknown"),
                "content": result.get("text", ""),
                "chunk_index": result.get("chunk_index", 0),
                "score": score,
                "metadata": result.get("metadata", {}),
                "collection": metadata.get(
                    "@collection", result.get("collection", "DocumentChunks")
                ),
            })

        return formatted_results

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
    import requests

    if url is None:
        url = RavenDBConfig.get_url()
    if database is None:
        database = RavenDBConfig.get_database_name()

    try:
        # Use RavenDB REST API to get collection statistics
        response = requests.get(
            f"{url}/databases/{database}/collections/stats", timeout=10
        )
        response.raise_for_status()

        data = response.json()
        collections = list(data.get("Collections", {}).keys())
        return sorted(collections)

    except requests.RequestException:
        # If API call fails, return empty list
        return []


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
            (defaults to EMBEDDING_MODEL env or 'nomic-embed-text')
        url: RavenDB server URL (defaults to value from RavenDBConfig.get_url())
        database: Database name (defaults to value from RavenDBConfig.get_database_name())

    Returns:
        int: Number of chunks successfully stored
    """
    if not chunks:
        return 0

    # Get defaults
    if embedding_model is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
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

                # Create document dict matching DocumentChunk structure
                doc = {
                    "id": doc_id,
                    "source_filename": source_filename,
                    "chunk_index": chunk_index,
                    "text": chunk.get("text", ""),
                    "embedding": embeddings[i],
                    "metadata": chunk.get("metadata", {}),
                    "collection": collection,
                }

                # Store the document
                session.store(doc, doc_id)

                # Set the collection in document metadata
                metadata = session.advanced.get_metadata_for(doc)
                metadata["@collection"] = collection

            session.save_changes()

        return len(chunks)

    finally:
        store.close()
