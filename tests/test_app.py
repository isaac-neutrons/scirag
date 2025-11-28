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

    @patch("scirag.client.app.Client")
    def test_chat_retrieval_error_handling(self, mock_client_class):
        """Test error handling when document retrieval fails."""
        # Mock MCP client to raise an error
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        async def mock_call_tool(tool_name, params):
            raise Exception("Retrieval failed")

        mock_client.__aenter__.return_value.call_tool = mock_call_tool

        with app.test_client() as client:
            response = client.post(
                "/api/chat",
                data=json.dumps({"query": "test"}),
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data

    @patch("scirag.client.app.Client")
    @patch("scirag.client.app.llm_service")
    def test_chat_llm_error_handling(self, mock_llm_service, mock_client_class):
        """Test error handling when LLM call fails."""
        # Mock MCP client call_tool
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text=json.dumps([]))]

        async def mock_call_tool(tool_name, params):
            return mock_result

        mock_client.__aenter__.return_value.call_tool = mock_call_tool

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


class TestUploadPage:
    """Tests for the /upload endpoint."""

    def test_upload_page_returns_html(self):
        """Test that upload page renders correctly."""
        with app.test_client() as client:
            response = client.get("/upload")
            assert response.status_code == 200
            assert b"Document Upload" in response.data
            assert b"Collection" in response.data
            assert b"Drag & Drop" in response.data


class TestCollectionsEndpoint:
    """Tests for the /api/collections endpoint."""

    @patch("scirag.client.app.asyncio")
    @patch("scirag.client.app.Client")
    def test_collections_returns_list(self, mock_client_class, mock_asyncio):
        """Test that collections endpoint returns list of collections."""
        # Mock asyncio event loop
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = ["papers", "reports", "research"]

        with app.test_client() as client:
            response = client.get("/api/collections")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert data["collections"] == ["papers", "reports", "research"]

    @patch("scirag.client.app.asyncio")
    @patch("scirag.client.app.Client")
    def test_collections_returns_empty_list(self, mock_client_class, mock_asyncio):
        """Test that collections endpoint returns empty list when no collections exist."""
        # Mock asyncio event loop
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = []

        with app.test_client() as client:
            response = client.get("/api/collections")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert data["collections"] == []


class TestUploadEndpoint:
    """Tests for the /api/upload endpoint."""

    def test_upload_no_files_returns_error(self):
        """Test that upload without files returns error."""
        with app.test_client() as client:
            response = client.post("/api/upload", data={})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data["success"] is False
            assert "No files" in data["error"]

    def test_upload_empty_files_returns_error(self):
        """Test that upload with empty file list returns error."""
        with app.test_client() as client:
            response = client.post(
                "/api/upload",
                data={"files": [], "collection": "test"},
                content_type="multipart/form-data",
            )
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data["success"] is False

    @patch("scirag.client.app.asyncio")
    @patch("scirag.client.app.extract_chunks_from_pdf")
    def test_upload_pdf_success(self, mock_extract, mock_asyncio):
        """Test successful PDF upload and ingestion."""
        from io import BytesIO

        # Mock the extract function to return chunks (dicts, not objects)
        mock_chunks = [{"id": "test_chunk_0", "text": "test content"}]
        mock_extract.return_value = mock_chunks

        # Mock asyncio.new_event_loop and run_until_complete
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = {
            "success": True,
            "chunks_stored": 1
        }

        with app.test_client() as client:
            # Create a fake PDF file
            data = {
                "files": (BytesIO(b"%PDF-1.4 test content"), "test.pdf"),
                "collection": "test-collection",
            }
            response = client.post(
                "/api/upload",
                data=data,
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["success"] is True
            assert result["collection"] == "test-collection"
            assert len(result["details"]) == 1
            assert result["details"][0]["status"] == "success"

    def test_upload_non_pdf_rejected(self):
        """Test that non-PDF files are rejected."""
        from io import BytesIO

        with app.test_client() as client:
            data = {
                "files": (BytesIO(b"not a pdf"), "test.txt"),
                "collection": "test",
            }
            response = client.post(
                "/api/upload",
                data=data,
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            result = json.loads(response.data)
            # Since all files were rejected, success should be False
            assert result["success"] is False
            assert "not allowed" in result["details"][0]["error"]

    @patch("scirag.client.app.extract_chunks_from_pdf")
    def test_upload_handles_ingest_error(self, mock_extract):
        """Test that ingest errors are handled gracefully."""
        from io import BytesIO

        mock_extract.side_effect = Exception("Ingest failed")

        with app.test_client() as client:
            data = {
                "files": (BytesIO(b"%PDF-1.4 test"), "test.pdf"),
                "collection": "test",
            }
            response = client.post(
                "/api/upload",
                data=data,
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["success"] is False
            assert result["details"][0]["status"] == "error"
            assert "Ingest failed" in result["details"][0]["error"]
