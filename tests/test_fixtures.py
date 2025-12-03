"""Test that fixtures are properly configured and accessible."""

import json
from pathlib import Path

import pytest


def test_sample_pdf_exists(sample_pdf):
    """Test that sample PDF fixture exists and is readable."""
    assert sample_pdf.exists()
    assert sample_pdf.suffix == ".pdf"
    assert sample_pdf.stat().st_size > 0


def test_sample_embeddings_structure(sample_embeddings):
    """Test that sample embeddings have correct structure."""
    assert "samples" in sample_embeddings
    assert "model" in sample_embeddings
    assert "embedding_dim" in sample_embeddings
    
    samples = sample_embeddings["samples"]
    assert len(samples) > 0
    
    for sample in samples:
        assert "text" in sample
        assert "embedding" in sample
        assert isinstance(sample["embedding"], list)
        assert len(sample["embedding"]) > 0


def test_test_chunks_structure(test_chunks):
    """Test that test chunks have correct structure."""
    assert "chunks" in test_chunks
    
    chunks = test_chunks["chunks"]
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert "source_filename" in chunk
        assert "chunk_index" in chunk
        assert "metadata" in chunk


def test_create_test_chunk_factory(create_test_chunk):
    """Test the create_test_chunk factory fixture."""
    chunk = create_test_chunk()
    
    assert chunk["id"] == "test_chunk_0"
    assert chunk["text"] == "Test chunk text"
    assert chunk["source_filename"] == "test.pdf"
    assert chunk["chunk_index"] == 0
    
    # Test custom parameters
    custom_chunk = create_test_chunk(
        chunk_id="custom_id",
        text="Custom text",
        source="custom.pdf",
        chunk_index=5
    )
    
    assert custom_chunk["id"] == "custom_id"
    assert custom_chunk["text"] == "Custom text"
    assert custom_chunk["chunk_index"] == 5


def test_mock_embedding_fixture(mock_embedding):
    """Test the mock embedding fixture."""
    assert isinstance(mock_embedding, list)
    assert len(mock_embedding) == 8
    assert all(isinstance(x, (int, float)) for x in mock_embedding)
