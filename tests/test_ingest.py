"""Tests for the ingest module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scirag.client.ingest import (
    chunk_text,
    extract_text_from_pdf,
    extract_chunks_from_pdf,
)


class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function."""

    @pytest.mark.integration
    def test_extract_text_from_real_pdf(self, sample_pdf):
        """Test extracting text from a real PDF file."""
        text = extract_text_from_pdf(sample_pdf)
        
        # Verify text was extracted
        assert len(text) > 0
        
        # Verify expected content is present
        assert "Test Document for SciRAG" in text
        assert "Python programming" in text
        assert "machine learning" in text
        
        # Verify multi-page extraction works
        assert "vector embeddings" in text  # From page 2
        assert "semantic search" in text  # From page 2

    @pytest.mark.integration
    def test_extract_text_handles_nonexistent_file(self):
        """Test that extract_text_from_pdf handles missing files appropriately."""
        nonexistent_path = Path("/nonexistent/file.pdf")
        
        with pytest.raises(Exception):  # PyMuPDF raises FileNotFoundError or similar
            extract_text_from_pdf(nonexistent_path)


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

    @pytest.mark.integration
    def test_extract_chunks_from_real_pdf(self, sample_pdf):
        """Test extracting chunks from a real PDF file."""
        chunks = extract_chunks_from_pdf(sample_pdf, collection="test-collection")
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify chunk structure
        first_chunk = chunks[0]
        assert "text" in first_chunk
        assert "source_filename" in first_chunk
        assert "chunk_index" in first_chunk
        assert "metadata" in first_chunk
        
        # Verify content
        assert first_chunk["source_filename"] == "sample.pdf"
        assert len(first_chunk["text"]) > 0
        
        # Verify metadata includes expected fields
        metadata = first_chunk["metadata"]
        assert "file_size" in metadata
        assert "page_count" in metadata
        assert metadata["page_count"] == 2  # Our sample has 2 pages
        assert metadata["file_size"] > 0
        
        # Verify chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    @pytest.mark.integration
    def test_extract_chunks_preserves_content(self, sample_pdf):
        """Test that chunks preserve the original PDF content."""
        chunks = extract_chunks_from_pdf(sample_pdf)
        
        # Combine all chunk text
        combined_text = " ".join(chunk["text"] for chunk in chunks)
        
        # Verify key content is preserved
        assert "Python programming" in combined_text
        assert "machine learning" in combined_text
        assert "vector embeddings" in combined_text


