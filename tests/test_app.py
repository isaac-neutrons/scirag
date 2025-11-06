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

        assert "Source: paper1.pdf" in result
        assert "Content from paper 1" in result
        assert "Source: paper2.pdf" in result
        assert "Content from paper 2" in result

    def test_format_context_empty_chunks(self):
        """Test formatting with no chunks returns appropriate message."""
        result = format_context([])
        assert "No relevant context found" in result

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

    @patch("scirag.client.app.mcp_client")
    def test_chat_retrieval_error_handling(self, mock_mcp_client):
        """Test error handling when document retrieval fails."""

        # Mock MCP client to raise an error
        async def mock_call_tool(tool_name, params):
            raise Exception("Retrieval failed")

        mock_mcp_client.__aenter__.return_value.call_tool = mock_call_tool

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test"}),
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data

    @patch("scirag.client.app.mcp_client")
    @patch("scirag.client.app.llm_service")
    def test_chat_llm_error_handling(self, mock_llm_service, mock_mcp_client):
        """Test error handling when LLM call fails."""
        # Mock MCP client call_tool
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text=json.dumps([]))]

        async def mock_call_tool(tool_name, params):
            return mock_result

        mock_mcp_client.__aenter__.return_value.call_tool = mock_call_tool

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
