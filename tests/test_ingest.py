"""Tests for the ingest module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from scirag.client.ingest import (
    DocumentChunk,
    chunk_text,
    extract_text_from_pdf,
    generate_embeddings,
    ingest_pdf,
    store_chunks,
)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        chunk = DocumentChunk(
            id="test.pdf_chunk_0",
            source_filename="test.pdf",
            chunk_index=0,
            text="Sample text",
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.id == "test.pdf_chunk_0"
        assert chunk.source_filename == "test.pdf"
        assert chunk.chunk_index == 0
        assert chunk.text == "Sample text"
        assert chunk.embedding == [0.1, 0.2, 0.3]


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


class TestGenerateEmbeddings:
    """Tests for generate_embeddings function."""

    @patch("scirag.client.ingest.ollama.embed")
    def test_generate_embeddings_single_text(self, mock_embed):
        """Test generating embeddings for a single text."""
        mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        embeddings = generate_embeddings(["test text"], "nomic-embed-text")

        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(model="nomic-embed-text", input="test text")

    @patch("scirag.client.ingest.ollama.embed")
    def test_generate_embeddings_multiple_texts(self, mock_embed):
        """Test generating embeddings for multiple texts."""
        mock_embed.side_effect = [
            {"embeddings": [[0.1, 0.2]]},
            {"embeddings": [[0.3, 0.4]]},
            {"embeddings": [[0.5, 0.6]]},
        ]

        texts = ["text1", "text2", "text3"]
        embeddings = generate_embeddings(texts, "nomic-embed-text")

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        assert embeddings[2] == [0.5, 0.6]
        assert mock_embed.call_count == 3


class TestIngestPDF:
    """Tests for ingest_pdf function."""

    @patch("scirag.client.ingest.generate_embeddings")
    @patch("scirag.client.ingest.chunk_text")
    @patch("scirag.client.ingest.extract_text_from_pdf")
    def test_ingest_pdf_complete_flow(
        self, mock_extract, mock_chunk, mock_generate, capsys
    ):
        """Test complete PDF ingestion flow."""
        mock_extract.return_value = "Sample text from PDF"
        mock_chunk.return_value = ["chunk1", "chunk2"]
        mock_generate.return_value = [[0.1, 0.2], [0.3, 0.4]]

        pdf_path = Path("test.pdf")
        chunks = ingest_pdf(pdf_path, "nomic-embed-text")

        assert len(chunks) == 2
        assert chunks[0].id == "test.pdf_chunk_0"
        assert chunks[0].source_filename == "test.pdf"
        assert chunks[0].chunk_index == 0
        assert chunks[0].text == "chunk1"
        assert chunks[0].embedding == [0.1, 0.2]

        assert chunks[1].id == "test.pdf_chunk_1"
        assert chunks[1].chunk_index == 1
        assert chunks[1].text == "chunk2"
        assert chunks[1].embedding == [0.3, 0.4]

        # Check console output
        captured = capsys.readouterr()
        assert "Processing test.pdf" in captured.out
        assert "✓ Processed test.pdf" in captured.out


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

        chunks = [
            DocumentChunk(
                id="test.pdf_chunk_0",
                source_filename="test.pdf",
                chunk_index=0,
                text="chunk text",
                embedding=[0.1, 0.2, 0.3],
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

        # Verify stored data structure
        call_args = mock_session.store.call_args
        stored_dict = call_args[0][0]
        assert stored_dict["id"] == "test.pdf_chunk_0"
        assert stored_dict["source_filename"] == "test.pdf"
        assert stored_dict["chunk_index"] == 0
        assert stored_dict["text"] == "chunk text"
        assert stored_dict["embedding"] == [0.1, 0.2, 0.3]
        assert stored_dict["@metadata"]["@collection"] == "DocumentChunks"
