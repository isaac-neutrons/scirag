"""Tests for the ingest module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from scirag.client.ingest import (
    DocumentChunk,
    chunk_text,
    extract_text_from_pdf,
    ingest_pdf,
    store_chunks,
)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        metadata = {
            "file_size": 12345,
            "modification_date": 1234567890.0,
            "creation_date": 1234567890.0,
            "page_count": 5,
            "ingestion_date": 1234567890.0,
        }

        chunk = DocumentChunk(
            id="test.pdf_chunk_0",
            source_filename="test.pdf",
            chunk_index=0,
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata,
        )

        assert chunk.id == "test.pdf_chunk_0"
        assert chunk.source_filename == "test.pdf"
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata["file_size"] == 12345
        assert chunk.metadata["page_count"] == 5


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


class TestStoreChunks:
    """Tests for store_chunks function."""

    @patch("scirag.client.ingest.ensure_index_exists")
    @patch("scirag.client.ingest.create_document_store")
    def test_store_chunks(self, mock_create_store, mock_ensure_index):
        """Test storing chunks in RavenDB."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store.open_session.return_value.__enter__.return_value = mock_session
        mock_create_store.return_value = mock_store

        metadata = {
            "file_size": 12345,
            "modification_date": 1234567890.0,
            "creation_date": 1234567890.0,
            "page_count": 5,
            "ingestion_date": 1234567890.0,
        }

        chunks = [
            DocumentChunk(
                id="test.pdf_chunk_0",
                source_filename="test.pdf",
                chunk_index=0,
                text="chunk text",
                embedding=[0.1, 0.2, 0.3],
                metadata=metadata,
            )
        ]

        store_chunks(chunks)

        # Verify store was created and closed
        mock_create_store.assert_called_once()
        mock_ensure_index.assert_called_once_with(mock_store)
        mock_store.close.assert_called_once()

        # Verify session operations
        mock_session.store.assert_called_once()
        mock_session.save_changes.assert_called_once()

        # Verify the DocumentChunk object was stored directly
        call_args = mock_session.store.call_args
        stored_chunk = call_args[0][0]
        stored_id = call_args[0][1]

        # Verify it's the DocumentChunk object
        assert isinstance(stored_chunk, DocumentChunk)
        assert stored_chunk.id == "test.pdf_chunk_0"
        assert stored_chunk.source_filename == "test.pdf"
        assert stored_chunk.chunk_index == 0
        assert stored_chunk.text == "chunk text"
        assert stored_chunk.embedding == [0.1, 0.2, 0.3]
        assert stored_chunk.metadata["file_size"] == 12345
        assert stored_chunk.metadata["page_count"] == 5
        assert stored_id == "test.pdf_chunk_0"

