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
        """Test that ensure_index_exists handles None by raising AttributeError."""
        # This should raise AttributeError when None is passed
        with pytest.raises(AttributeError):
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
