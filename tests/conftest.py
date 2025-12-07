"""Pytest configuration and shared fixtures for the test suite."""

import json
import requests
from pathlib import Path
from typing import Generator

import pytest


# Test fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_EMBEDDINGS = FIXTURES_DIR / "sample_embeddings.json"
TEST_CHUNKS = FIXTURES_DIR / "test_chunks.json"


# Service availability checks
def ollama_available() -> bool:
    """Check if Ollama server is running and accessible.

    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, Exception):
        return False


def ravendb_available() -> bool:
    """Check if RavenDB server is running and accessible.

    Returns:
        True if RavenDB is available, False otherwise
    """
    try:
        response = requests.get("http://localhost:8080/databases", timeout=2)
        return response.status_code == 200 or response.status_code == 401  # Auth required is OK
    except (requests.RequestException, Exception):
        return False


# Path fixtures
@pytest.fixture
def sample_pdf() -> Path:
    """Provide path to sample PDF fixture.

    Returns:
        Path to the sample PDF file
    """
    return SAMPLE_PDF


@pytest.fixture
def sample_embeddings() -> dict:
    """Load sample embeddings from fixture file.

    Returns:
        Dictionary containing sample embeddings data
    """
    with open(SAMPLE_EMBEDDINGS) as f:
        return json.load(f)


@pytest.fixture
def test_chunks() -> dict:
    """Load test chunks from fixture file.

    Returns:
        Dictionary containing test document chunks
    """
    with open(TEST_CHUNKS) as f:
        return json.load(f)


# Service fixtures with skip markers
@pytest.fixture
def ollama_service():
    """Provide OllamaService instance, skip if Ollama not available.

    Yields:
        OllamaService instance configured for localhost

    Raises:
        pytest.skip: If Ollama server is not running
    """
    if not ollama_available():
        pytest.skip("Ollama server not running on localhost:11434")

    from scirag.llm import OllamaService

    return OllamaService(host="http://localhost:11434", model="llama3")


@pytest.fixture
def ravendb_store():
    """Provide RavenDB DocumentStore, skip if RavenDB not available.

    Yields:
        Initialized DocumentStore instance

    Raises:
        pytest.skip: If RavenDB server is not running
    """
    if not ravendb_available():
        pytest.skip("RavenDB server not running on localhost:8080")

    from scirag.service.database import create_document_store

    store = create_document_store()
    yield store
    store.close()


@pytest.fixture
def temp_test_db(tmp_path) -> Generator[str, None, None]:
    """Create a temporary test database name.

    Args:
        tmp_path: pytest tmp_path fixture

    Yields:
        Temporary database name
    """
    db_name = f"test_scirag_{tmp_path.name}"
    yield db_name
    # Cleanup could be added here if needed


# Helper fixtures
@pytest.fixture
def mock_embedding():
    """Provide a simple mock embedding vector.

    Returns:
        List of floats representing an embedding vector
    """
    return [0.1, 0.2, 0.3, 0.15, -0.1, 0.05, 0.25, -0.05]


# Test data generators
@pytest.fixture
def create_test_chunk():
    """Factory fixture to create test document chunks.

    Returns:
        Function that creates a test chunk with custom parameters
    """

    def _create_chunk(
        chunk_id: str = "test_chunk_0",
        text: str = "Test chunk text",
        source: str = "test.pdf",
        chunk_index: int = 0,
        collection: str = "test-collection",
    ) -> dict:
        return {
            "id": chunk_id,
            "text": text,
            "source_filename": source,
            "chunk_index": chunk_index,
            "collection": collection,
            "metadata": {
                "file_size": 1024,
                "page_count": 1,
                "title": "Test Document",
                "created_date": "2025-12-02",
                "modified_date": "2025-12-02",
            },
        }

    return _create_chunk
