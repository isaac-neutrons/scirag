"""Command-line interface for SciRAG using Click."""

import os
from pathlib import Path

import click

from scirag.client.ingest import ingest_pdf, store_chunks


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help=(
        "Ollama embedding model to use "
        "(default: from OLLAMA_EMBEDDING_MODEL env or 'nomic-embed-text')"
    ),
)
def ingest(directory: Path, embedding_model: str | None) -> None:
    """Ingest PDF files from DIRECTORY into the SciRAG knowledge base.

    Processes all PDF files in the specified directory, extracts text,
    chunks it, generates embeddings, and stores in RavenDB.

    Example:
        scirag-ingest documents/
        scirag-ingest documents/ --embedding-model custom-model
    """
    # Get embedding model
    model = embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # Get PDF files
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        click.echo(f"No PDF files found in '{directory}'")
        return

    click.echo(f"Found {len(pdf_files)} PDF file(s)")
    click.echo(f"Using embedding model: {model}")
    click.echo()

    # Process each PDF
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            chunks = ingest_pdf(pdf_path, model)
            all_chunks.extend(chunks)
        except Exception as e:
            click.echo(f"  ✗ Error processing {pdf_path.name}: {e}", err=True)
            continue

    # Store all chunks
    if all_chunks:
        click.echo()
        click.echo(f"Storing {len(all_chunks)} chunks in RavenDB...")
        store_chunks(all_chunks)
        click.echo("✓ Ingestion complete!")
    else:
        click.echo("No chunks to store.")


if __name__ == "__main__":
    ingest()
