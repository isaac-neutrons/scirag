"""Tests for the CLI module."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scirag.client.cli import count, ingest, search


class TestIngestCLI:
    """Tests for the ingest CLI command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_single_file(self, mock_db_exists, mock_extract, mock_async_run, tmp_path):
        """Test ingest command with a directory containing one PDF."""
        # Create a temporary directory with a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test", "source": "doc.pdf"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Found 1 PDF file" in result.output
        assert "Storing 1 chunks" in result.output
        mock_extract.assert_called_once()
        mock_async_run.assert_called_once()

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_multiple_files(self, mock_db_exists, mock_extract, mock_async_run, tmp_path):
        """Test ingest command with a directory of PDFs."""
        # Create temporary PDF files
        mock_db_exists.return_value = True
        pdf1 = tmp_path / "file1.pdf"
        pdf1.write_text("dummy1")
        pdf2 = tmp_path / "file2.pdf"
        pdf2.write_text("dummy2")

        mock_chunks1 = [{"id": "file1.pdf_chunk_0", "text": "test1"}]
        mock_chunks2 = [{"id": "file2.pdf_chunk_0", "text": "test2"}]
        mock_extract.side_effect = [mock_chunks1, mock_chunks2]
        mock_async_run.return_value = {"success": True, "chunks_stored": 2}

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Found 2 PDF file(s)" in result.output
        assert mock_extract.call_count == 2
        mock_async_run.assert_called_once()

    def test_ingest_nonexistent_directory(self):
        """Test ingest command with non-existent directory."""
        result = self.runner.invoke(ingest, ["/nonexistent/path"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_with_error(self, mock_db_exists, mock_extract, tmp_path):
        """Test ingest command when an error occurs during processing."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_extract.side_effect = Exception("Test error")

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0  # CLI doesn't exit on processing errors
        assert "Error processing" in result.output
        assert "Test error" in result.output

    def test_ingest_missing_argument(self):
        """Test ingest command with missing directory argument."""
        result = self.runner.invoke(ingest, [])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_empty_directory(self, mock_db_exists, tmp_path):
        """Test ingest command with directory containing no PDFs."""
        mock_db_exists.return_value = True
        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "No PDF files found" in result.output

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_custom_embedding_model(
        self, mock_db_exists, mock_extract, mock_async_run, tmp_path
    ):
        """Test ingest command with custom embedding model."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        result = self.runner.invoke(ingest, [str(tmp_path), "--embedding-model", "custom-model"])

        assert result.exit_code == 0
        assert "Using embedding model: custom-model" in result.output

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_custom_collection(self, mock_db_exists, mock_extract, mock_async_run, tmp_path):
        """Test ingest command with custom collection name."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        result = self.runner.invoke(ingest, [str(tmp_path), "--collection", "research-papers"])

        assert result.exit_code == 0
        assert "Collection: research-papers" in result.output
        assert "collection: 'research-papers'" in result.output

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    @patch.dict("os.environ", {"EMBEDDING_MODEL": "env-model"})
    def test_ingest_with_env_model(self, mock_db_exists, mock_extract, mock_async_run, tmp_path):
        """Test ingest command uses model from environment variable."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Using embedding model: env-model" in result.output

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_cli_flag_overrides_env(
        self, mock_db_exists, mock_extract, mock_async_run, tmp_path
    ):
        """Test that --embedding-model CLI flag overrides environment variable."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        with patch.dict("os.environ", {"EMBEDDING_MODEL": "env-model"}):
            result = self.runner.invoke(ingest, [str(tmp_path), "--embedding-model", "cli-model"])

        assert result.exit_code == 0
        assert "Using embedding model: cli-model" in result.output
        # Verify CLI flag takes precedence

    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_database_not_exists_no_flag(self, mock_db_exists, tmp_path):
        """Test that CLI aborts with helpful message when database doesn't exist."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "--create-database" in result.output

    @patch("scirag.client.cli_helpers.create_database")
    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_database_created_with_flag(
        self, mock_db_exists, mock_extract, mock_async_run, mock_create_db, tmp_path
    ):
        """Test that CLI creates database when --create-database flag is used."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        mock_async_run.return_value = {"success": True, "chunks_stored": 1}

        result = self.runner.invoke(ingest, [str(tmp_path), "--create-database"])

        assert result.exit_code == 0
        assert "Database created successfully" in result.output
        mock_create_db.assert_called_once()

    @patch("scirag.client.cli_helpers.create_database")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_database_creation_fails(self, mock_db_exists, mock_create_db, tmp_path):
        """Test that CLI handles database creation failure gracefully."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False
        mock_create_db.side_effect = Exception("Connection failed")

        result = self.runner.invoke(ingest, [str(tmp_path), "--create-database"])

        assert result.exit_code != 0
        assert "Failed to create database" in result.output
        assert "Connection failed" in result.output

    @patch("scirag.client.cli.asyncio.run")
    @patch("scirag.client.cli.extract_chunks_from_pdf")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_ingest_store_chunks_database_error(
        self, mock_db_exists, mock_extract, mock_async_run, tmp_path
    ):
        """Test that CLI handles DatabaseDoesNotExistException during storage."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [{"id": "doc.pdf_chunk_0", "text": "test"}]
        mock_extract.return_value = mock_chunks
        # Simulate database error with message matching what the CLI checks for
        mock_async_run.side_effect = Exception("Database does not exist")

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "--create-database" in result.output


class TestCountCLI:
    """Tests for the count CLI command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scirag.client.cli_helpers.count_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_count_success(self, mock_db_exists, mock_count):
        """Test count command successfully returns document count."""
        mock_db_exists.return_value = True
        mock_count.return_value = 42

        result = self.runner.invoke(count)

        assert result.exit_code == 0
        assert "42 document chunk(s)" in result.output
        mock_count.assert_called_once()

    @patch("scirag.client.cli_helpers.count_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_count_zero_documents(self, mock_db_exists, mock_count):
        """Test count command when database is empty."""
        mock_db_exists.return_value = True
        mock_count.return_value = 0

        result = self.runner.invoke(count)

        assert result.exit_code == 0
        assert "0 document chunk(s)" in result.output

    @patch("scirag.client.cli_helpers.database_exists")
    def test_count_database_not_exists(self, mock_db_exists):
        """Test count command when database doesn't exist."""
        mock_db_exists.return_value = False

        result = self.runner.invoke(count)

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "scirag-ingest" in result.output
        assert "--create-database" in result.output

    @patch("scirag.client.cli_helpers.count_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_count_database_error(self, mock_db_exists, mock_count):
        """Test count command when database query fails."""
        mock_db_exists.return_value = True
        mock_count.side_effect = Exception("Connection failed")

        result = self.runner.invoke(count)

        assert result.exit_code != 0
        assert "Error counting documents" in result.output


class TestSearchCLI:
    """Tests for the search CLI command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_success(self, mock_db_exists, mock_search):
        """Test search command successfully returns results."""
        mock_db_exists.return_value = True
        mock_search.return_value = [
            {
                "source": "doc1.pdf",
                "content": "This is relevant content about quantum mechanics",
                "chunk_index": 0,
                "score": 0.95,
            },
            {
                "source": "doc2.pdf",
                "content": "Another relevant section discussing quantum theory",
                "chunk_index": 2,
                "score": 0.87,
            },
        ]

        result = self.runner.invoke(search, ["quantum mechanics"])

        assert result.exit_code == 0
        assert "Searching for: 'quantum mechanics'" in result.output
        assert "Found 2 result(s)" in result.output
        assert "doc1.pdf" in result.output
        assert "doc2.pdf" in result.output
        assert "0.9500" in result.output
        assert "0.8700" in result.output
        mock_search.assert_called_once()

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_with_custom_top_k(self, mock_db_exists, mock_search):
        """Test search command with custom top-k parameter."""
        mock_db_exists.return_value = True
        mock_search.return_value = []

        result = self.runner.invoke(search, ["test query", "--top-k", "3"])

        assert result.exit_code == 0
        assert "top 3 results" in result.output
        # Verify top_k parameter was passed
        call_args = mock_search.call_args
        assert call_args[1]["top_k"] == 3

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_with_custom_model(self, mock_db_exists, mock_search):
        """Test search command with custom embedding model."""
        mock_db_exists.return_value = True
        mock_search.return_value = []

        result = self.runner.invoke(search, ["test query", "--embedding-model", "custom-model"])

        assert result.exit_code == 0
        # Verify embedding_model parameter was passed
        call_args = mock_search.call_args
        assert call_args[1]["embedding_model"] == "custom-model"

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_no_results(self, mock_db_exists, mock_search):
        """Test search command when no results found."""
        mock_db_exists.return_value = True
        mock_search.return_value = []

        result = self.runner.invoke(search, ["nonexistent query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_database_not_exists(self, mock_db_exists):
        """Test search command when database doesn't exist."""
        mock_db_exists.return_value = False

        result = self.runner.invoke(search, ["test query"])

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "scirag-ingest" in result.output

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_connection_error(self, mock_db_exists, mock_search):
        """Test search command when connection fails."""
        mock_db_exists.return_value = True
        mock_search.side_effect = ConnectionError("Cannot connect to Ollama")

        result = self.runner.invoke(search, ["test query"])

        assert result.exit_code != 0
        assert "Connection error" in result.output
        assert "Cannot connect to Ollama" in result.output

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_value_error(self, mock_db_exists, mock_search):
        """Test search command when value error occurs."""
        mock_db_exists.return_value = True
        mock_search.side_effect = ValueError("Invalid model")

        result = self.runner.invoke(search, ["test query"])

        assert result.exit_code != 0
        assert "Error:" in result.output
        assert "Invalid model" in result.output

    @patch("scirag.client.cli.search_documents")
    @patch("scirag.client.cli_helpers.database_exists")
    def test_search_truncates_long_content(self, mock_db_exists, mock_search):
        """Test that search command truncates long content for display."""
        mock_db_exists.return_value = True
        long_content = "a" * 300  # Content longer than 200 chars
        mock_search.return_value = [
            {
                "source": "doc1.pdf",
                "content": long_content,
                "chunk_index": 0,
                "score": 0.95,
            }
        ]

        result = self.runner.invoke(search, ["test query"])

        assert result.exit_code == 0
        assert "..." in result.output  # Should have ellipsis for truncated content
        # Should not contain the full 300 characters
        assert long_content not in result.output


# ============================================================================
# Integration Tests - Test with real components
# ============================================================================


class TestIngestCLIIntegration:
    """Integration tests for ingest CLI command with real components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @pytest.mark.integration
    def test_ingest_real_pdf_extraction(self, sample_pdf):
        """Test ingest command with real PDF file (no DB/embeddings)."""
        # Test just the PDF extraction part without database
        from scirag.client.ingest import extract_chunks_from_pdf

        # Verify PDF can be processed
        chunks = extract_chunks_from_pdf(sample_pdf, collection="test-cli")
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all(len(chunk["text"]) > 0 for chunk in chunks)

    @pytest.mark.integration
    @pytest.mark.requires_ravendb
    @pytest.mark.slow
    def test_ingest_full_workflow_with_database(self, sample_pdf, tmp_path, ravendb_store):
        """Test complete ingest workflow with real database."""
        # Copy sample PDF to temp directory for CLI
        import shutil

        test_dir = tmp_path / "pdfs"
        test_dir.mkdir()
        test_pdf = test_dir / "sample.pdf"
        shutil.copy(sample_pdf, test_pdf)

        result = self.runner.invoke(
            ingest, [str(test_dir), "--collection", "test-cli-integration", "--create-database"]
        )

        # Should succeed
        assert result.exit_code == 0
        assert "Found 1 PDF file" in result.output
        assert "successfully" in result.output.lower() or "stored" in result.output.lower()

    @pytest.mark.integration
    def test_ingest_validates_real_pdf_format(self, tmp_path):
        """Test that ingest properly validates actual PDF files."""
        # Create a file that's not a real PDF
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is not a PDF file")

        # Should handle gracefully (either skip or error)
        result = self.runner.invoke(ingest, [str(tmp_path)])

        # Should not crash completely
        assert "Error" in result.output or "No PDF files" in result.output or result.exit_code != 0
