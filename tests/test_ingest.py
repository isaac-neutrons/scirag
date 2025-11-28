"""Tests for the ingest module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from scirag.client.ingest import (
    chunk_text,
    extract_text_from_pdf,
    extract_chunks_from_pdf,
)


class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function."""

    @patch("scirag.client.ingest.fitz.open")
    def test_extract_text_single_page(self, mock_fitz_open):
        """Test extracting text from a single-page PDF."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 content"
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        text = extract_text_from_pdf(Path("test.pdf"))

        assert text == "Page 1 content"
        mock_fitz_open.assert_called_once_with(Path("test.pdf"))
        mock_doc.close.assert_called_once()

    @patch("scirag.client.ingest.fitz.open")
    def test_extract_text_multiple_pages(self, mock_fitz_open):
        """Test extracting text from a multi-page PDF."""
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2"
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_doc

        text = extract_text_from_pdf(Path("test.pdf"))

        assert text == "Page 1Page 2"
        mock_doc.close.assert_called_once()


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_text_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_chunk_size(self):
        """Test chunking text that's exactly the chunk size."""
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_overlap(self):
        """Test chunking text with overlap between chunks."""
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        assert len(chunks) == 2
        # Check that chunks overlap
        chunk1_words = chunks[0].split()
        chunk2_words = chunks[1].split()
        assert len(chunk1_words) == 500
        # Second chunk should start 450 words in (500 - 50 overlap)
        assert chunk2_words[0] == chunk1_words[450]

    def test_chunk_text_no_overlap(self):
        """Test chunking text with no overlap."""
        text = " ".join([f"word{i}" for i in range(1000)])
        chunks = chunk_text(text, chunk_size=500, overlap=0)

        assert len(chunks) == 2
        assert len(chunks[0].split()) == 500
        assert len(chunks[1].split()) == 500

    def test_chunk_text_custom_sizes(self):
        """Test chunking with custom chunk and overlap sizes."""
        text = " ".join(["word"] * 300)
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 100 for chunk in chunks)


class TestExtractChunksFromPDF:
    """Tests for extract_chunks_from_pdf function."""

    @patch("scirag.client.ingest.fitz.open")
    @patch("scirag.client.ingest.chunk_text")
    @patch("scirag.client.ingest.extract_text_from_pdf")
    def test_extract_chunks_from_pdf(
        self, mock_extract_text, mock_chunk_text, mock_fitz_open
    ):
        """Test extracting chunks from a PDF file."""
        # Mock the PDF document
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.metadata = {"title": "Test Title", "author": "Test Author"}
        mock_fitz_open.return_value = mock_doc

        # Mock text extraction and chunking
        mock_extract_text.return_value = "Sample text from PDF"
        mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]

        # Create a mock path with stat
        mock_path = MagicMock(spec=Path)
        mock_path.name = "test.pdf"
        mock_stat = MagicMock()
        mock_stat.st_size = 12345
        mock_stat.st_mtime = 1234567890.0
        mock_stat.st_ctime = 1234567890.0
        mock_path.stat.return_value = mock_stat

        chunks = extract_chunks_from_pdf(mock_path, collection="TestCollection")

        assert len(chunks) == 2
        assert chunks[0]["text"] == "Chunk 1"
        assert chunks[0]["source_filename"] == "test.pdf"
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["metadata"]["file_size"] == 12345
        assert chunks[0]["metadata"]["page_count"] == 5
        assert chunks[0]["metadata"]["title"] == "Test Title"
        assert chunks[0]["metadata"]["author"] == "Test Author"

        assert chunks[1]["text"] == "Chunk 2"
        assert chunks[1]["chunk_index"] == 1

