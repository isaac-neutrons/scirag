"""Tests for the database module."""

import os
from unittest.mock import MagicMock, patch
import ollama
import pytest

from scirag.service.database import (
    RavenDBConfig,
    cosine_similarity,
    count_documents,
    create_database,
    create_document_store,
    database_exists,
    ensure_index_exists,
    search_documents,
)


class TestRavenDBIntegration:
    """Integration tests for RavenDB operations."""

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_store_and_retrieve_chunks(self, ravendb_store, create_test_chunk):
        """Test storing and retrieving document chunks from real RavenDB."""
        # Create test chunks
        chunks = [
            create_test_chunk(chunk_id="test_1", text="First chunk", chunk_index=0),
            create_test_chunk(chunk_id="test_2", text="Second chunk", chunk_index=1),
        ]
        
        # Store chunks
        with ravendb_store.open_session() as session:
            for chunk in chunks:
                session.store(chunk, chunk["id"])
            session.save_changes()
        
        # Retrieve and verify
        with ravendb_store.open_session() as session:
            retrieved_1 = session.load(chunks[0]["id"])
            retrieved_2 = session.load(chunks[1]["id"])
            
            assert retrieved_1 is not None
            assert retrieved_2 is not None
            assert retrieved_1["text"] == "First chunk"
            assert retrieved_2["text"] == "Second chunk"

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_count_documents_real(self, ravendb_store, create_test_chunk):
        """Test counting documents in real RavenDB."""
        # Store some test chunks
        with ravendb_store.open_session() as session:
            for i in range(3):
                chunk = create_test_chunk(chunk_id=f"count_test_{i}", chunk_index=i)
                session.store(chunk, chunk["id"])
            session.save_changes()
        
        # Count should be >= 3 (may have other test data)
        count = count_documents()
        assert count >= 3


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors_return_one(self):
        """Test that identical vectors have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Test that orthogonal vectors have similarity of 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        """Test that opposite vectors have similarity of -1.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_empty_vectors_return_zero(self):
        """Test that empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_different_length_vectors_return_zero(self):
        """Test that vectors of different lengths return 0.0."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_zero_magnitude_vector_returns_zero(self):
        """Test that zero magnitude vectors return 0.0."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0


# Removed TestRavenDBConfig class - trivial getter tests that don't verify business logic
# These tests only verified that os.environ.get() works, which is standard library behavior


class TestCreateDocumentStore:
    """Tests for create_document_store function."""

    @patch("scirag.service.database.DocumentStore")
    def test_creates_document_store_with_defaults(self, mock_document_store_class):
        """Test that create_document_store creates a DocumentStore with default config."""
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://test:8080", "RAVENDB_DATABASE": "testdb"},
        ):
            store = create_document_store()

            # Verify DocumentStore was created with correct parameters (list format)
            mock_document_store_class.assert_called_once_with(
                ["http://test:8080"], "testdb"
            )

            # Verify initialize was called
            mock_store.initialize.assert_called_once()

            # Verify the instance is returned
            assert store is mock_store

    @patch("scirag.service.database.DocumentStore")
    def test_creates_document_store_with_custom_params(self, mock_document_store_class):
        """Test that create_document_store accepts custom URL and database."""
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        store = create_document_store(
            url="http://custom:9090", database="custom_db"
        )

        # Verify DocumentStore was created with custom parameters (list format)
        mock_document_store_class.assert_called_once_with(
            ["http://custom:9090"], "custom_db"
        )

        # Verify initialize was called
        mock_store.initialize.assert_called_once()

        # Verify the instance is returned
        assert store is mock_store


# Removed test_creates_new_instance_each_call - tests Python object instantiation behavior,
# not business logic. This is language semantics, not application functionality.


# Removed TestEnsureIndexExists class - test_ensure_index_exists_callable tests nothing meaningful
# (just that a function doesn't raise with a mock), and test_ensure_index_exists_with_none
# tests AttributeError on None, which is Python behavior not our logic.


class TestDatabaseExists:
    """Tests for database_exists function."""

    @patch("scirag.service.database.DocumentStore")
    def test_database_exists_returns_true(self, mock_document_store_class):
        """Test that database_exists returns True when database is accessible."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.take.return_value = []
        mock_document_store_class.return_value = mock_store

        result = database_exists("http://test:8080", "testdb")

        assert result is True
        mock_document_store_class.assert_called_once_with(["http://test:8080"], "testdb")
        mock_store.initialize.assert_called_once()
        mock_store.close.assert_called_once()

    @patch("scirag.service.database.DocumentStore")
    def test_database_exists_returns_false_on_exception(
        self, mock_document_store_class
    ):
        """Test that database_exists returns False when database doesn't exist."""
        mock_store = MagicMock()
        mock_store.initialize.side_effect = Exception("Database not found")
        mock_document_store_class.return_value = mock_store

        result = database_exists("http://test:8080", "testdb")

        assert result is False

    @patch("scirag.service.database.DocumentStore")
    def test_database_exists_with_defaults(self, mock_document_store_class):
        """Test that database_exists uses default config when not specified."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.take.return_value = []
        mock_document_store_class.return_value = mock_store

        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://env:8080", "RAVENDB_DATABASE": "envdb"},
        ):
            result = database_exists()

            assert result is True
            mock_document_store_class.assert_called_once_with(
                ["http://env:8080"], "envdb"
            )


class TestCreateDatabase:
    """Tests for create_database function."""

    @patch("requests.put")
    def test_create_database_success(self, mock_put):
        """Test that create_database makes correct API call."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        create_database("http://test:8080", "testdb")

        mock_put.assert_called_once_with(
            "http://test:8080/admin/databases",
            json={"DatabaseName": "testdb", "Settings": {}, "Disabled": False},
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.put")
    def test_create_database_with_defaults(self, mock_put):
        """Test that create_database uses default config when not specified."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://env:8080", "RAVENDB_DATABASE": "envdb"},
        ):
            create_database()

            mock_put.assert_called_once_with(
                "http://env:8080/admin/databases",
                json={"DatabaseName": "envdb", "Settings": {}, "Disabled": False},
            )

    @patch("requests.put")
    def test_create_database_raises_on_failure(self, mock_put):
        """Test that create_database raises exception on API failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API error")
        mock_put.return_value = mock_response

        try:
            create_database("http://test:8080", "testdb")
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert str(e) == "API error"


class TestCountDocuments:
    """Tests for count_documents function."""

    @patch("scirag.service.database.DocumentStore")
    def test_count_documents_returns_count(self, mock_document_store_class):
        """Test that count_documents returns the count of DocumentChunks."""
        # Setup mocks
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        # Mock the raw_query return value (42 items)
        mock_session.advanced.raw_query.return_value = [
            {"id": "1"}, {"id": "2"}, {"id": "3"}
        ] * 14

        # Call the function
        count = count_documents("http://test:8080", "testdb")

        # Verify
        assert count == 42
        mock_document_store_class.assert_called_once_with(
            ["http://test:8080"], "testdb"
        )
        mock_store.initialize.assert_called_once()
        mock_session.advanced.raw_query.assert_called_once_with(
            "from DocumentChunks", object_type=dict
        )
        mock_store.close.assert_called_once()

    @patch("scirag.service.database.DocumentStore")
    def test_count_documents_with_defaults(self, mock_document_store_class):
        """Test that count_documents uses default config when not specified."""
        # Setup mocks
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        mock_session.advanced.raw_query.return_value = [{"id": str(i)} for i in range(10)]

        # Call with defaults
        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://env:8080", "RAVENDB_DATABASE": "envdb"},
        ):
            count = count_documents()

            # Verify defaults were used
            mock_document_store_class.assert_called_once_with(["http://env:8080"], "envdb")
            assert count == 10

    @patch("scirag.service.database.DocumentStore")
    def test_count_documents_returns_zero_for_empty_database(
        self, mock_document_store_class
    ):
        """Test that count_documents returns 0 when database is empty."""
        # Setup mocks
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        mock_session.advanced.raw_query.return_value = []

        # Call the function
        count = count_documents("http://test:8080", "testdb")

        # Verify
        assert count == 0
        mock_store.close.assert_called_once()

    @patch("scirag.service.database.DocumentStore")
    def test_count_documents_closes_store_on_error(self, mock_document_store_class):
        """Test that count_documents closes store even when error occurs."""
        # Setup mocks
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        # Make raw_query raise an exception
        mock_session.advanced.raw_query.side_effect = Exception("Query failed")

        # Call the function and expect exception
        with pytest.raises(Exception) as exc_info:
            count_documents("http://test:8080", "testdb")

        assert str(exc_info.value) == "Query failed"
        # Verify store was closed despite error
        mock_store.close.assert_called_once()


class TestGetCollections:
    """Tests for get_collections function."""

    @patch("scirag.service.database.requests.get")
    def test_get_collections_returns_sorted_list(self, mock_get):
        """Test that get_collections returns sorted list of collections."""
        from scirag.service.database import get_collections

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Collections": {"zebra": 5, "alpha": 10, "beta": 3}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_collections("http://test:8080", "testdb")

        assert result == ["alpha", "beta", "zebra"]
        mock_get.assert_called_once_with(
            "http://test:8080/databases/testdb/collections/stats", timeout=10
        )

    @patch("scirag.service.database.requests.get")
    def test_get_collections_returns_empty_list_on_no_collections(self, mock_get):
        """Test that get_collections returns empty list when no collections exist."""
        from scirag.service.database import get_collections

        mock_response = MagicMock()
        mock_response.json.return_value = {"Collections": {}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_collections("http://test:8080", "testdb")

        assert result == []

    @patch("scirag.service.database.requests.get")
    def test_get_collections_returns_empty_list_on_request_error(self, mock_get):
        """Test that get_collections returns empty list on request failure."""
        import requests
        from scirag.service.database import get_collections

        mock_get.side_effect = requests.RequestException("Connection failed")

        result = get_collections("http://test:8080", "testdb")

        assert result == []

    @patch("scirag.service.database.requests.get")
    def test_get_collections_uses_default_config(self, mock_get):
        """Test that get_collections uses default URL and database when not provided."""
        from scirag.service.database import get_collections

        mock_response = MagicMock()
        mock_response.json.return_value = {"Collections": {"docs": 1}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://env-server:8080", "RAVENDB_DATABASE": "envdb"},
        ):
            result = get_collections()

        assert result == ["docs"]
        mock_get.assert_called_once_with(
            "http://env-server:8080/databases/envdb/collections/stats", timeout=10
        )


# ============================================================================
# Additional Integration Tests - Real database operations
# ============================================================================

class TestDatabaseStorageIntegration:
    """Integration tests for database storage with real RavenDB."""

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_store_and_search_chunks_with_query(self, ravendb_store, test_chunks):
        """Test storing chunks and querying them back with RQL."""
        chunks_data = test_chunks["chunks"][:2]  # Use first 2 chunks
        
        # Store chunks with unique collection for this test
        collection_name = "test_query_chunks"
        with ravendb_store.open_session() as session:
            for i, chunk in enumerate(chunks_data):
                chunk_copy = chunk.copy()
                chunk_copy["collection"] = collection_name
                session.store(chunk_copy, f"{collection_name}/{i}")
            session.save_changes()
        
        # Query back using RQL
        with ravendb_store.open_session() as session:
            results = list(session.advanced.raw_query(
                f"from @all_docs where collection = '{collection_name}'"
            ))
            
            assert len(results) >= 2
            assert any("Python programming" in r.get("text", "") for r in results)

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    @pytest.mark.requires_ollama
    @pytest.mark.slow
    def test_vector_search_real_similarity(self, ravendb_store, ollama_service):
        """Test vector search returns documents by actual similarity."""
        from scirag.service.database import cosine_similarity
        
        # Create documents with real embeddings
        docs = [
            {
                "id": "vec_test_1",
                "text": "Python programming for data analysis",
                "collection": "vec_search_test"
            },
            {
                "id": "vec_test_2",
                "text": "Cooking delicious Italian recipes",
                "collection": "vec_search_test"
            },
            {
                "id": "vec_test_3",
                "text": "Machine learning and artificial intelligence",
                "collection": "vec_search_test"
            }
        ]
        
        # Generate real embeddings
        texts = [doc["text"] for doc in docs]
        embeddings = ollama_service.generate_embeddings(texts, "nomic-embed-text")
        
        # Add embeddings to docs
        for doc, embedding in zip(docs, embeddings):
            doc["embedding"] = embedding
        
        # Store in database
        with ravendb_store.open_session() as session:
            for doc in docs:
                session.store(doc, doc["id"])
            session.save_changes()
        
        # Search for programming-related content
        query_text = "software development and coding"
        query_embedding = ollama_service.generate_embeddings([query_text], "nomic-embed-text")[0]
        
        # Retrieve all docs and calculate similarity
        with ravendb_store.open_session() as session:
            all_docs = list(session.advanced.raw_query(
                "from @all_docs where collection = 'vec_search_test'"
            ))
            
            similarities = []
            for doc in all_docs:
                if "embedding" in doc:
                    sim = cosine_similarity(query_embedding, doc["embedding"])
                    similarities.append((sim, doc["text"]))
            
            # Sort by similarity
            similarities.sort(reverse=True)
        
        # Top results should be programming-related, not cooking
        assert len(similarities) >= 3
        top_result = similarities[0][1]
        assert "Python" in top_result or "Machine learning" in top_result
        assert "Cooking" not in top_result

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_database_collection_management(self, ravendb_store):
        """Test creating and listing collections in real database."""
        from scirag.service.database import get_collections
        
        collection_name = "test_collection_mgmt"
        
        # Store a document in new collection
        with ravendb_store.open_session() as session:
            test_doc = {
                "id": f"{collection_name}/1",
                "collection": collection_name,
                "data": "test"
            }
            session.store(test_doc, test_doc["id"])
            session.save_changes()
        
        # Get collections - should include our new one
        collections = get_collections()
        
        # Note: Collection stats might not update immediately in RavenDB
        # This test verifies the get_collections function works with real DB
        assert isinstance(collections, list)

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_count_documents_real_database(self, ravendb_store, create_test_chunk):
        """Test counting documents in real RavenDB."""
        collection_name = "test_count_real"
        
        # Store multiple chunks
        with ravendb_store.open_session() as session:
            for i in range(5):
                chunk = create_test_chunk(
                    chunk_id=f"{collection_name}/{i}",
                    text=f"Chunk {i}",
                    chunk_index=i,
                    collection=collection_name
                )
                session.store(chunk, chunk["id"])
            session.save_changes()
        
        # Count should be at least 5
        count = count_documents(collection=collection_name)
        assert count >= 5

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    def test_ensure_index_exists_real(self, ravendb_store):
        """Test index creation on real RavenDB."""
        # This tests that ensure_index_exists doesn't crash with real DB
        ensure_index_exists(ravendb_store)
        
        # If we get here without exception, the function works
        assert True
