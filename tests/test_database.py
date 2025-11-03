"""Tests for the database module."""

import os
from unittest.mock import MagicMock, patch

from scirag.service.database import (
    RavenDBConfig,
    count_documents,
    create_database,
    create_document_store,
    database_exists,
    ensure_index_exists,
    search_documents,
)


class TestRavenDBConfig:
    """Tests for RavenDBConfig class."""

    def test_get_url_default(self):
        """Test get_url returns default value when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            url = RavenDBConfig.get_url()
            assert url == "http://localhost:8080"

    def test_get_url_from_env(self):
        """Test get_url returns value from environment variable."""
        test_url = "http://test-server:8080"
        with patch.dict(os.environ, {"RAVENDB_URL": test_url}):
            url = RavenDBConfig.get_url()
            assert url == test_url

    def test_get_database_name_default(self):
        """Test get_database_name returns default value when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            database = RavenDBConfig.get_database_name()
            assert database == "scirag"

    def test_get_database_name_from_env(self):
        """Test get_database_name returns value from environment variable."""
        test_db = "test_database"
        with patch.dict(os.environ, {"RAVENDB_DATABASE": test_db}):
            database = RavenDBConfig.get_database_name()
            assert database == test_db


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

            # Verify DocumentStore was created with correct parameters
            mock_document_store_class.assert_called_once_with(
                "http://test:8080", "testdb"
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

        # Verify DocumentStore was created with custom parameters
        mock_document_store_class.assert_called_once_with(
            "http://custom:9090", "custom_db"
        )

        # Verify initialize was called
        mock_store.initialize.assert_called_once()

        # Verify the instance is returned
        assert store is mock_store

    @patch("scirag.service.database.DocumentStore")
    def test_creates_new_instance_each_call(self, mock_document_store_class):
        """Test that create_document_store creates a new instance on each call."""
        mock_store1 = MagicMock()
        mock_store2 = MagicMock()
        mock_document_store_class.side_effect = [mock_store1, mock_store2]

        store1 = create_document_store()
        store2 = create_document_store()

        # Verify DocumentStore was created twice
        assert mock_document_store_class.call_count == 2

        # Verify different instances are returned
        assert store1 is mock_store1
        assert store2 is mock_store2
        assert store1 is not store2


class TestEnsureIndexExists:
    """Tests for ensure_index_exists function."""

    def test_ensure_index_exists_callable(self):
        """Test that ensure_index_exists can be called without errors."""
        mock_store = MagicMock()
        # This should not raise any exceptions
        ensure_index_exists(mock_store)

    def test_ensure_index_exists_with_none(self):
        """Test that ensure_index_exists handles None gracefully."""
        # This should not raise any exceptions
        ensure_index_exists(None)  # type: ignore


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
        mock_document_store_class.assert_called_once_with("http://test:8080", "testdb")
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
                "http://env:8080", "envdb"
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

        # Create a mock query chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = [{"id": "1"}, {"id": "2"}, {"id": "3"}] * 14  # 42 items

        # Call the function
        count = count_documents("http://test:8080", "testdb")

        # Verify
        assert count == 42
        mock_document_store_class.assert_called_once_with("http://test:8080", "testdb")
        mock_store.initialize.assert_called_once()
        mock_session.query.assert_called_once()
        mock_query.raw_query.assert_called_once_with("from DocumentChunks")
        mock_store.close.assert_called_once()

    @patch("scirag.service.database.DocumentStore")
    def test_count_documents_with_defaults(self, mock_document_store_class):
        """Test that count_documents uses default config when not specified."""
        # Setup mocks
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = [{"id": str(i)} for i in range(10)]

        # Call with defaults
        with patch.dict(
            os.environ,
            {"RAVENDB_URL": "http://env:8080", "RAVENDB_DATABASE": "envdb"},
        ):
            count = count_documents()

            # Verify defaults were used
            mock_document_store_class.assert_called_once_with("http://env:8080", "envdb")
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

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = []

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

        # Make query raise an exception
        mock_session.query.side_effect = Exception("Query failed")

        # Call the function and expect exception
        try:
            count_documents("http://test:8080", "testdb")
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert str(e) == "Query failed"
            # Verify store was closed despite error
            mock_store.close.assert_called_once()


class TestSearchDocuments:
    """Tests for search_documents function."""

    @patch("scirag.service.database.ollama")
    @patch("scirag.service.database.DocumentStore")
    def test_search_documents_returns_results(
        self, mock_document_store_class, mock_ollama
    ):
        """Test that search_documents returns formatted search results."""
        # Setup Ollama mock
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        # Setup DocumentStore mock
        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        # Mock query results
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = [
            {
                "source_filename": "doc1.pdf",
                "text": "This is test content",
                "chunk_index": 0,
                "@metadata": {"@index-score": 0.95},
            },
            {
                "source_filename": "doc2.pdf",
                "text": "Another test content",
                "chunk_index": 1,
                "@metadata": {"@index-score": 0.85},
            },
        ]

        # Call the function
        results = search_documents("test query", top_k=2)

        # Verify
        assert len(results) == 2
        assert results[0]["source"] == "doc1.pdf"
        assert results[0]["content"] == "This is test content"
        assert results[0]["chunk_index"] == 0
        assert results[0]["score"] == 0.95
        assert results[1]["source"] == "doc2.pdf"
        assert results[1]["score"] == 0.85

        mock_ollama.embed.assert_called_once()
        mock_store.initialize.assert_called_once()
        mock_store.close.assert_called_once()

    @patch("scirag.service.database.ollama")
    @patch("scirag.service.database.DocumentStore")
    def test_search_documents_with_custom_params(
        self, mock_document_store_class, mock_ollama
    ):
        """Test search_documents with custom parameters."""
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}

        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = []

        # Call with custom parameters
        results = search_documents(
            "test query",
            top_k=10,
            embedding_model="custom-model",
            url="http://custom:8080",
            database="customdb",
        )

        # Verify custom params were used
        mock_ollama.embed.assert_called_once_with(model="custom-model", input="test query")
        mock_document_store_class.assert_called_once_with("http://custom:8080", "customdb")
        assert results == []

    @patch("scirag.service.database.ollama")
    def test_search_documents_ollama_connection_error(self, mock_ollama):
        """Test that search_documents raises ConnectionError when Ollama is unreachable."""
        mock_ollama.embed.side_effect = ConnectionError("Connection refused")

        try:
            search_documents("test query")
            assert False, "Expected ConnectionError to be raised"
        except ConnectionError as e:
            assert "Cannot connect to Ollama" in str(e)

    @patch("scirag.service.database.ollama")
    def test_search_documents_ollama_invalid_response(self, mock_ollama):
        """Test that search_documents raises ValueError for invalid Ollama response."""
        mock_ollama.embed.return_value = {"invalid_key": "no embeddings"}

        try:
            search_documents("test query")
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Invalid response from Ollama" in str(e)

    @patch("scirag.service.database.ollama")
    @patch("scirag.service.database.DocumentStore")
    def test_search_documents_closes_store_on_error(
        self, mock_document_store_class, mock_ollama
    ):
        """Test that search_documents closes store even when error occurs."""
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}

        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session
        mock_session.query.side_effect = Exception("Query failed")

        try:
            search_documents("test query")
            assert False, "Expected exception to be raised"
        except Exception:
            # Verify store was closed despite error
            mock_store.close.assert_called_once()

    @patch("scirag.service.database.ollama")
    @patch("scirag.service.database.DocumentStore")
    def test_search_documents_uses_env_defaults(
        self, mock_document_store_class, mock_ollama
    ):
        """Test that search_documents uses environment variable defaults."""
        mock_ollama.embed.return_value = {"embeddings": [[0.1]]}

        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.raw_query.return_value = []

        with patch.dict(
            os.environ,
            {
                "OLLAMA_EMBEDDING_MODEL": "env-model",
                "RAVENDB_URL": "http://env:8080",
                "RAVENDB_DATABASE": "envdb",
            },
        ):
            search_documents("test query")

            mock_ollama.embed.assert_called_once_with(model="env-model", input="test query")
            mock_document_store_class.assert_called_once_with("http://env:8080", "envdb")

    @patch("scirag.service.database.ollama")
    @patch("scirag.service.database.DocumentStore")
    def test_search_documents_returns_metadata(
        self, mock_document_store_class, mock_ollama
    ):
        """Test that search_documents returns complete metadata from documents."""
        mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}

        mock_store = MagicMock()
        mock_document_store_class.return_value = mock_store

        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session

        # Mock query chain with query_collection
        mock_query = MagicMock()
        mock_session.query_collection.return_value = mock_query
        mock_query.vector_search.return_value = mock_query
        mock_query.order_by_score.return_value = mock_query
        mock_query.take.return_value = [
            {
                "source_filename": "test.pdf",
                "text": "test content",
                "chunk_index": 0,
                "metadata": {
                    "file_size": 12345,
                    "creation_date": 1699000000.0,
                    "modification_date": 1699000001.0,
                    "page_count": 5,
                    "title": "Test Document",
                },
                "@metadata": {},
            }
        ]

        results = search_documents("test query", top_k=1)

        assert len(results) == 1
        assert "metadata" in results[0]
        assert results[0]["metadata"]["creation_date"] == 1699000000.0
        assert results[0]["metadata"]["file_size"] == 12345
        assert results[0]["metadata"]["page_count"] == 5
        assert results[0]["metadata"]["title"] == "Test Document"


