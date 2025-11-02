"""Tests for the CLI module."""

from unittest.mock import patch

from click.testing import CliRunner
from pyravendb.custom_exceptions.exceptions import DatabaseDoesNotExistException

from scirag.client.cli import ingest
from scirag.client.ingest import DocumentChunk


class TestIngestCLI:
    """Tests for the ingest CLI command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_single_file(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test ingest command with a directory containing one PDF."""
        # Create a temporary directory with a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Found 1 PDF file" in result.output
        assert "Storing 1 chunks in RavenDB" in result.output
        mock_ingest.assert_called_once()
        mock_store.assert_called_once_with(mock_chunks)

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_multiple_files(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test ingest command with a directory of PDFs."""
        # Create temporary PDF files
        mock_db_exists.return_value = True
        pdf1 = tmp_path / "file1.pdf"
        pdf1.write_text("dummy1")
        pdf2 = tmp_path / "file2.pdf"
        pdf2.write_text("dummy2")

        mock_chunks1 = [
            DocumentChunk(
                id="file1.pdf_chunk_0",
                source_filename="file1.pdf",
                chunk_index=0,
                text="test1",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_chunks2 = [
            DocumentChunk(
                id="file2.pdf_chunk_0",
                source_filename="file2.pdf",
                chunk_index=0,
                text="test2",
                embedding=[0.2],
                metadata={"file_size": 200, "page_count": 2},
            )
        ]
        mock_ingest.side_effect = [mock_chunks1, mock_chunks2]

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Found 2 PDF file(s)" in result.output
        assert mock_ingest.call_count == 2
        mock_store.assert_called_once()
        # Verify all chunks were passed together
        call_args = mock_store.call_args[0][0]
        assert len(call_args) == 2

    def test_ingest_nonexistent_directory(self):
        """Test ingest command with non-existent directory."""
        result = self.runner.invoke(ingest, ["/nonexistent/path"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_with_error(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test ingest command when an error occurs during processing."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_ingest.side_effect = Exception("Test error")

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0  # CLI doesn't exit on processing errors
        assert "Error processing" in result.output
        assert "Test error" in result.output
        # store_chunks should not be called if all PDFs fail
        mock_store.assert_not_called()

    def test_ingest_missing_argument(self):
        """Test ingest command with missing directory argument."""
        result = self.runner.invoke(ingest, [])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("scirag.client.cli.database_exists")
    def test_ingest_empty_directory(self, mock_db_exists, tmp_path):
        """Test ingest command with directory containing no PDFs."""
        mock_db_exists.return_value = True
        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "No PDF files found" in result.output

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_custom_embedding_model(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test ingest command with custom embedding model."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks

        result = self.runner.invoke(
            ingest, [str(tmp_path), "--embedding-model", "custom-model"]
        )

        assert result.exit_code == 0
        assert "Using embedding model: custom-model" in result.output
        # Verify ingest_pdf was called with custom model
        assert mock_ingest.call_args[0][1] == "custom-model"

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    @patch.dict("os.environ", {"OLLAMA_EMBEDDING_MODEL": "env-model"})
    def test_ingest_with_env_model(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test ingest command uses model from environment variable."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Using embedding model: env-model" in result.output
        # Verify ingest_pdf was called with env model
        assert mock_ingest.call_args[0][1] == "env-model"

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_cli_flag_overrides_env(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test that --embedding-model CLI flag overrides environment variable."""
        # Create a PDF file
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks

        with patch.dict("os.environ", {"OLLAMA_EMBEDDING_MODEL": "env-model"}):
            result = self.runner.invoke(
                ingest, [str(tmp_path), "--embedding-model", "cli-model"]
            )

        assert result.exit_code == 0
        assert "Using embedding model: cli-model" in result.output
        # Verify CLI flag takes precedence


    @patch("scirag.client.cli.database_exists")
    def test_ingest_database_not_exists_no_flag(self, mock_db_exists, tmp_path):
        """Test that CLI aborts with helpful message when database doesn't exist."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "--create-database" in result.output

    @patch("scirag.client.cli.create_database")
    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_database_created_with_flag(
        self, mock_ingest, mock_store, mock_db_exists, mock_create_db, tmp_path
    ):
        """Test that CLI creates database when --create-database flag is used."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks

        result = self.runner.invoke(ingest, [str(tmp_path), "--create-database"])

        assert result.exit_code == 0
        assert "Database created successfully" in result.output
        mock_create_db.assert_called_once()

    @patch("scirag.client.cli.create_database")
    @patch("scirag.client.cli.database_exists")
    def test_ingest_database_creation_fails(
        self, mock_db_exists, mock_create_db, tmp_path
    ):
        """Test that CLI handles database creation failure gracefully."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = False
        mock_create_db.side_effect = Exception("Connection failed")

        result = self.runner.invoke(ingest, [str(tmp_path), "--create-database"])

        assert result.exit_code != 0
        assert "Failed to create database" in result.output
        assert "Connection failed" in result.output

    @patch("scirag.client.cli.database_exists")
    @patch("scirag.client.cli.store_chunks")
    @patch("scirag.client.cli.ingest_pdf")
    def test_ingest_store_chunks_database_error(
        self, mock_ingest, mock_store, mock_db_exists, tmp_path
    ):
        """Test that CLI handles DatabaseDoesNotExistException during storage."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("dummy")

        mock_db_exists.return_value = True
        mock_chunks = [
            DocumentChunk(
                id="doc.pdf_chunk_0",
                source_filename="doc.pdf",
                chunk_index=0,
                text="test",
                embedding=[0.1],
                metadata={"file_size": 100, "page_count": 1},
            )
        ]
        mock_ingest.return_value = mock_chunks
        mock_store.side_effect = DatabaseDoesNotExistException("Database scirag does not exists")

        result = self.runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code != 0
        assert "Database does not exist" in result.output
        assert "--create-database" in result.output
