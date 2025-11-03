"""Tests for the MCP server module."""

from unittest.mock import patch

import pytest

from scirag.service.mcp_server import retrieve_document_chunks_impl


class TestRetrieveDocumentChunks:
    """Tests for the retrieve_document_chunks tool."""

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_success(self, mock_search_documents):
        """Test successful retrieval of document chunks."""
        # Mock search_documents to return results with all required fields
        mock_search_documents.return_value = [
            {
                "source": "test1.pdf",
                "content": "Content from test1",
                "chunk_index": 0,
                "score": 0.0,
                "metadata": {"file_size": 12345, "creation_date": 1699000000.0},
            },
            {
                "source": "test2.pdf",
                "content": "Content from test2",
                "chunk_index": 1,
                "score": 0.0,
                "metadata": {"file_size": 67890, "creation_date": 1699000001.0},
            },
        ]

        # Execute
        results = await retrieve_document_chunks_impl("test query", top_k=2)

        # Verify results are returned as-is from search_documents
        assert len(results) == 2
        assert results[0]["source"] == "test1.pdf"
        assert results[0]["content"] == "Content from test1"
        assert results[0]["chunk_index"] == 0
        assert results[0]["metadata"]["file_size"] == 12345
        assert results[1]["source"] == "test2.pdf"
        assert results[1]["content"] == "Content from test2"
        assert results[1]["chunk_index"] == 1

        # Verify search_documents was called with correct parameters
        mock_search_documents.assert_called_once_with(query="test query", top_k=2)

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_with_custom_top_k(self, mock_search_documents):
        """Test retrieval with custom top_k parameter."""
        mock_search_documents.return_value = []

        # Execute with custom top_k
        await retrieve_document_chunks_impl("test query", top_k=10)

        # Verify search_documents was called with top_k=10
        mock_search_documents.assert_called_once_with(query="test query", top_k=10)

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_no_results(self, mock_search_documents):
        """Test retrieval when no chunks match the query."""
        # Mock empty results
        mock_search_documents.return_value = []

        # Execute
        results = await retrieve_document_chunks_impl("nonexistent query")

        # Verify empty list is returned
        assert len(results) == 0
        assert results == []

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_connection_error(self, mock_search_documents):
        """Test handling of connection errors from search_documents."""
        # Mock connection error
        mock_search_documents.side_effect = ConnectionError("Cannot connect to Ollama")

        # Execute and expect exception to be re-raised
        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            await retrieve_document_chunks_impl("test query")

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_value_error(self, mock_search_documents):
        """Test handling of value errors from search_documents."""
        # Mock value error
        mock_search_documents.side_effect = ValueError("Invalid model")

        # Execute and expect exception to be re-raised
        with pytest.raises(ValueError, match="Invalid model"):
            await retrieve_document_chunks_impl("test query")

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_unexpected_error(self, mock_search_documents):
        """Test handling of unexpected errors from search_documents."""
        # Mock unexpected error
        mock_search_documents.side_effect = RuntimeError("Unexpected issue")

        # Execute and expect exception to be re-raised
        with pytest.raises(RuntimeError, match="Unexpected issue"):
            await retrieve_document_chunks_impl("test query")

    @pytest.mark.asyncio
    @patch("scirag.service.mcp_server.search_documents")
    async def test_retrieve_chunks_returns_metadata(self, mock_search_documents):
        """Test that metadata is included in results."""
        # Mock result with complete metadata
        mock_search_documents.return_value = [
            {
                "source": "paper.pdf",
                "content": "Scientific content about quantum physics",
                "chunk_index": 42,
                "score": 0.0,
                "metadata": {
                    "file_size": 247822,
                    "creation_date": 1762037494.9822903,
                    "page_count": 3,
                    "title": "Research Paper",
                },
            }
        ]

        # Execute
        results = await retrieve_document_chunks_impl("quantum")

        # Verify result structure includes metadata
        assert len(results) == 1
        result = results[0]
        assert "source" in result
        assert "content" in result
        assert "chunk_index" in result
        assert "metadata" in result
        assert result["metadata"]["file_size"] == 247822
        assert result["metadata"]["page_count"] == 3
        assert result["metadata"]["title"] == "Research Paper"


class TestMain:
    """Tests for the main entry point."""

    @patch("scirag.service.mcp_server.mcp.run")
    def test_main_runs_mcp_server(self, mock_run):
        """Test that main() starts the MCP server with stdio transport."""
        from scirag.service.mcp_server import main

        # Execute
        main()

        # Verify MCP server was started with stdio transport
        mock_run.assert_called_once_with(transport="stdio")
