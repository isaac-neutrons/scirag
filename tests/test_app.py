"""Tests for the Flask application module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from scirag.client.app import app, format_context


class TestFormatContext:
    """Tests for the format_context function."""

    def test_format_context_with_chunks(self):
        """Test formatting multiple document chunks into context."""
        chunks = [
            {
                "source": "paper1.pdf",
                "content": "Content from paper 1",
                "chunk_index": 0,
            },
            {
                "source": "paper2.pdf",
                "content": "Content from paper 2",
                "chunk_index": 1,
            },
        ]

        result = format_context(chunks)

        assert "[Source 1: paper1.pdf, Chunk 0]" in result
        assert "Content from paper 1" in result
        assert "[Source 2: paper2.pdf, Chunk 1]" in result
        assert "Content from paper 2" in result

    def test_format_context_empty_chunks(self):
        """Test formatting with no chunks returns appropriate message."""
        result = format_context([])
        assert "No relevant information found" in result

    def test_format_context_missing_fields(self):
        """Test formatting handles missing fields gracefully."""
        chunks = [{"content": "Some content"}]
        result = format_context(chunks)
        assert "Unknown" in result  # Default source
        assert "Some content" in result


class TestIndexEndpoint:
    """Tests for the / (index) endpoint."""

    def test_index_returns_chat_page(self):
        """Test index endpoint serves the chat interface."""
        with app.test_client() as client:
            response = client.get("/")

            assert response.status_code == 200
            assert b"SciRAG Document Chat" in response.data
            assert b"chat-container" in response.data


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_status(self):
        """Test health endpoint returns service status."""
        with app.test_client() as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert "status" in data
            assert "llm_service" in data


class TestChatEndpoint:
    """Tests for the /api/chat endpoint."""

    def test_chat_missing_query(self):
        """Test chat endpoint with missing query returns 400."""
        with app.test_client() as client:
            response = client.post(
                "/api/chat", data=json.dumps({}), content_type="application/json"
            )

            assert response.status_code == 400
            data = json.loads(response.data)
            assert "error" in data
            assert "query" in data["error"]

    @patch("scirag.client.app.retrieve_document_chunks_impl")
    @patch("scirag.client.app.llm_service")
    def test_chat_success(self, mock_llm_service, mock_retrieve):
        """Test successful chat request flow."""
        # Mock document retrieval - make it an async function that returns the list
        async def mock_retrieve_impl(query, top_k):
            return [
                {
                    "source": "test.pdf",
                    "content": "Test content",
                    "chunk_index": 0,
                }
            ]

        mock_retrieve.side_effect = mock_retrieve_impl

        # Mock LLM service response
        mock_llm_service.generate_response = AsyncMock(
            return_value="This is the answer"
        )

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test question"}),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert "response" in data
            assert "sources" in data
            assert data["response"] == "This is the answer"

    @patch("scirag.client.app.retrieve_document_chunks_impl")
    @patch("scirag.client.app.llm_service")
    def test_chat_with_custom_top_k(self, mock_llm_service, mock_retrieve):
        """Test chat request with custom top_k parameter."""
        # Mock document retrieval
        async def mock_retrieve_impl(query, top_k):
            return []

        mock_retrieve.side_effect = mock_retrieve_impl

        # Mock LLM service response
        mock_llm_service.generate_response = AsyncMock(return_value="Answer")

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test", "top_k": 10}),
                content_type="application/json",
            )

            assert response.status_code == 200
            # Verify retrieve was called with correct top_k
            mock_retrieve.assert_called_once_with("test", 10)

    @patch("scirag.client.app.retrieve_document_chunks_impl")
    def test_chat_retrieval_error_handling(self, mock_retrieve):
        """Test error handling when document retrieval fails."""
        # Mock retrieve to raise an error
        mock_retrieve.side_effect = Exception("Retrieval failed")

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test"}),
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data

    @patch("scirag.client.app.retrieve_document_chunks_impl")
    @patch("scirag.client.app.llm_service")
    def test_chat_llm_error_handling(self, mock_llm_service, mock_retrieve):
        """Test error handling when LLM call fails."""
        # Mock document retrieval
        mock_retrieve.return_value = AsyncMock(return_value=[])()

        # Mock LLM service to raise an error
        mock_llm_service.generate_response = AsyncMock(
            side_effect=Exception("LLM generation failed")
        )

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test"}),
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data

    @patch("scirag.client.app.retrieve_document_chunks_impl")
    @patch("scirag.client.app.llm_service")
    def test_chat_includes_sources(self, mock_llm_service, mock_retrieve):
        """Test that response includes source information."""
        # Mock document retrieval with multiple sources
        async def mock_retrieve_impl(query, top_k):
            return [
                {"source": "doc1.pdf", "content": "Content 1", "chunk_index": 0},
                {"source": "doc2.pdf", "content": "Content 2", "chunk_index": 1},
            ]

        mock_retrieve.side_effect = mock_retrieve_impl

        # Mock LLM service response
        mock_llm_service.generate_response = AsyncMock(return_value="Answer")

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test"}),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data["sources"]) == 2
            assert data["sources"][0]["source"] == "doc1.pdf"
            assert data["sources"][0]["chunk_index"] == 0
            assert data["sources"][1]["source"] == "doc2.pdf"
            assert data["sources"][1]["chunk_index"] == 1


class TestInitializeServices:
    """Tests for service initialization."""

    @patch("scirag.client.app.get_llm_service")
    def test_initialize_services_creates_llm_service(self, mock_get_llm_service):
        """Test that initialize_services creates LLM service."""
        from scirag.client.app import initialize_services

        # Mock get_llm_service
        mock_llm = MagicMock()
        mock_get_llm_service.return_value = mock_llm

        initialize_services()

        # Verify LLM service was created
        mock_get_llm_service.assert_called_once()
        config = mock_get_llm_service.call_args[0][0]
        assert config["service"] == "ollama"
        assert "host" in config
        assert "model" in config


class TestMain:
    """Tests for the main entry point."""

    @patch("scirag.client.app.app.run")
    @patch("scirag.client.app.initialize_services")
    def test_main_starts_flask_app(self, mock_initialize, mock_run):
        """Test that main() initializes services and starts Flask app."""
        from scirag.client.app import main

        # Execute
        main()

        # Verify services were initialized
        mock_initialize.assert_called_once()

        # Verify Flask app was started
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert "host" in call_kwargs
        assert "port" in call_kwargs
        assert "debug" in call_kwargs
