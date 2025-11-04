"""PDF ingestion pipeline for extracting, chunking, and storing documents."""

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import ollama
from dotenv import load_dotenv

from scirag.service.database import create_document_store, ensure_index_exists

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with its embedding.

    Attributes:
        id: Unique identifier for the chunk (format: filename_chunk_index)
        source_filename: Original PDF filename
        chunk_index: int of this chunk in the document
        text: The text content of the chunk
        embedding: Vector embedding of the text
        metadata: Dictionary containing file metadata (creation_date, modification_date,
                 file_size, page_count, ingestion_date)
    """

    id: str
    source_filename: str
    chunk_index: int
    text: str
    embedding: list[float]
    metadata: dict[str, str | int | float]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Concatenated text from all pages
    """
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks based on word count.

    Args:
        text: The text to chunk
        chunk_size: Target number of words per chunk (default: 500)
        overlap: Number of words to overlap between chunks (default: 50)

    Returns:
        list[str]: List of text chunks
    """
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move start position, accounting for overlap
        start = end - overlap

        # Ensure we don't have an infinite loop
        if start >= len(words):
            break

    return chunks


def generate_embeddings(texts: list[str], model: str) -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama.

    Args:
        texts: List of text strings to embed
        model: Name of the embedding model to use

    Returns:
        list[list[float]]: List of embedding vectors
    """
    embeddings = []

    for text in texts:
        response = ollama.embed(model=model, input=text)
        embeddings.append(response["embeddings"][0])

    return embeddings


def ingest_pdf(pdf_path: Path, embedding_model: str) -> list[DocumentChunk]:
    """Process a PDF file into document chunks with embeddings.

    Args:
        pdf_path: Path to the PDF file
        embedding_model: Name of the embedding model to use

    Returns:
        list[DocumentChunk]: List of document chunks with embeddings
    """
    logging.info(f"Processing {pdf_path.name}...")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    logging.info(f"  Extracted {len(text)} characters")

    # Extract file metadata
    file_stat = pdf_path.stat()
    doc = fitz.open(pdf_path)

    metadata = {
        "file_size": file_stat.st_size,
        "modification_date": file_stat.st_mtime,
        "creation_date": file_stat.st_ctime,
        "page_count": len(doc),
        "ingestion_date": file_stat.st_mtime,  # Using mtime as ingestion timestamp
    }

    # Add PDF metadata if available
    pdf_metadata = doc.metadata
    if pdf_metadata:
        if pdf_metadata.get("title"):
            metadata["title"] = pdf_metadata["title"]
        if pdf_metadata.get("author"):
            metadata["author"] = pdf_metadata["author"]
        if pdf_metadata.get("creationDate"):
            metadata["pdf_creation_date"] = pdf_metadata["creationDate"]

    doc.close()

    # Chunk text
    text_chunks = chunk_text(text)
    logging.info(f"  Created {len(text_chunks)} chunks")

    # Generate embeddings
    logging.info("  Generating embeddings...")
    embeddings = generate_embeddings(text_chunks, embedding_model)

    # Create DocumentChunk objects
    document_chunks = []
    filename = pdf_path.name

    for idx, (text_content, embedding) in enumerate(zip(text_chunks, embeddings)):
        doc_chunk = DocumentChunk(
            id=f"{filename}_chunk_{idx}",
            source_filename=filename,
            chunk_index=idx,
            text=text_content,
            embedding=embedding,
            metadata=metadata.copy(),
        )
        document_chunks.append(doc_chunk)

    logging.info(f"  ✓ Processed {filename}")
    return document_chunks


def store_chunks(chunks: list[DocumentChunk]) -> None:
    """Store document chunks in RavenDB.

    The DocumentChunk dataclass objects are stored directly in RavenDB,
    which will automatically serialize them to JSON.

    Args:
        chunks: List of DocumentChunk objects to store
    """
    store = create_document_store()

    # Ensure the index exists
    ensure_index_exists(store)

    # Store chunks in a single session
    with store.open_session() as session:
        for chunk in chunks:
            # Store the DocumentChunk object directly
            session.store(chunk, chunk.id)

        session.save_changes()

    store.close()
