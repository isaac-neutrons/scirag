"""PDF ingestion pipeline for extracting, chunking, and storing documents."""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import ollama
from dotenv import load_dotenv

from scirag.service.database import create_document_store, ensure_index_exists

# Load environment variables
load_dotenv()


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with its embedding.

    Attributes:
        id: Unique identifier for the chunk (format: filename_chunk_index)
        source_filename: Original PDF filename
        chunk_index: Index of this chunk in the document
        text: The text content of the chunk
        embedding: Vector embedding of the text
    """

    id: str
    source_filename: str
    chunk_index: int
    text: str
    embedding: list[float]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Concatenated text from all pages

    Example:
        >>> text = extract_text_from_pdf(Path("document.pdf"))
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

    Example:
        >>> chunks = chunk_text("This is a long document...", chunk_size=100, overlap=10)
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

    Example:
        >>> embeddings = generate_embeddings(["text1", "text2"], "nomic-embed-text")
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

    Example:
        >>> chunks = ingest_pdf(Path("doc.pdf"), "nomic-embed-text")
    """
    print(f"Processing {pdf_path.name}...")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(text)} characters")

    # Chunk text
    text_chunks = chunk_text(text)
    print(f"  Created {len(text_chunks)} chunks")

    # Generate embeddings
    print("  Generating embeddings...")
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
        )
        document_chunks.append(doc_chunk)

    print(f"  ✓ Processed {filename}")
    return document_chunks


def store_chunks(chunks: list[DocumentChunk]) -> None:
    """Store document chunks in RavenDB.

    Args:
        chunks: List of DocumentChunk objects to store

    Example:
        >>> store_chunks(document_chunks)
    """
    store = create_document_store()

    # Ensure the index exists
    ensure_index_exists(store)

    # Store chunks in a single session
    with store.open_session() as session:
        for chunk in chunks:
            # Convert dataclass to dict for storage
            chunk_dict = {
                "id": chunk.id,
                "source_filename": chunk.source_filename,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "@metadata": {"@collection": "DocumentChunks"},
            }
            session.store(chunk_dict, chunk.id)

        session.save_changes()

    store.close()
