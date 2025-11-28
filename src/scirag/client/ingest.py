"""PDF ingestion pipeline for extracting and chunking documents."""

import logging
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)


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


def extract_chunks_from_pdf(
    pdf_path: Path,
    collection: str = "DocumentChunks",
) -> list[dict]:
    """Extract text chunks from a PDF file without generating embeddings.

    This function extracts text, chunks it, and returns chunk dictionaries
    ready to be sent to the MCP server for embedding generation and storage.

    Args:
        pdf_path: Path to the PDF file
        collection: Name of the collection to store chunks in (default: DocumentChunks)

    Returns:
        list[dict]: List of chunk dictionaries with text, source_filename,
                   chunk_index, and metadata
    """
    logging.info(f"Extracting chunks from {pdf_path.name}...")

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
        "ingestion_date": file_stat.st_mtime,
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

    # Create chunk dictionaries (without embeddings)
    chunks = []
    filename = pdf_path.name

    for idx, text_content in enumerate(text_chunks):
        chunk = {
            "text": text_content,
            "source_filename": filename,
            "chunk_index": idx,
            "metadata": metadata.copy(),
        }
        chunks.append(chunk)

    logging.info(f"  ✓ Extracted {len(chunks)} chunks from {filename}")
    return chunks
