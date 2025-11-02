"""Command-line interface for SciRAG using Click."""

import os
from pathlib import Path

import click
from pyravendb.custom_exceptions.exceptions import DatabaseDoesNotExistException

from scirag.client.ingest import ingest_pdf, store_chunks
from scirag.service.database import create_database, database_exists


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
@click.option(
    "--create-database",
    "create_database_flag",
    is_flag=True,
    default=False,
    help="Create the RavenDB database if it doesn't exist",
)
def ingest(directory: Path, embedding_model: str | None, create_database_flag: bool) -> None:
    """Ingest PDF files from DIRECTORY into the SciRAG knowledge base.

    Processes all PDF files in the specified directory, extracts text,
    chunks it, generates embeddings, and stores in RavenDB.

    Example:
        scirag-ingest documents/
        scirag-ingest documents/ --embedding-model custom-model
        scirag-ingest documents/ --create-database
    """
    # Check if database exists
    if not database_exists():
        if create_database_flag:
            click.echo("Database does not exist. Creating database...")
            try:
                create_database()
                click.echo("✓ Database created successfully!")
            except Exception as e:
                click.echo(f"✗ Failed to create database: {e}", err=True)
                click.echo(
                    "\nPlease ensure RavenDB is running and accessible.", err=True
                )
                raise click.Abort()
        else:
            click.echo(
                "✗ Error: Database does not exist!", err=True
            )
            click.echo(
                "\nPlease run the command with --create-database flag to create it:",
                err=True,
            )
            click.echo(
                f"  scirag-ingest {directory} --create-database", err=True
            )
            click.echo(
                "\nOr ensure RavenDB is running and the database exists.", err=True
            )
            raise click.Abort()

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
        try:
            store_chunks(all_chunks)
            click.echo("✓ Ingestion complete!")
        except DatabaseDoesNotExistException:
            click.echo(
                "\n✗ Error: Database does not exist!", err=True
            )
            click.echo(
                "Please run the command with --create-database flag:", err=True
            )
            click.echo(
                f"  scirag-ingest {directory} --create-database", err=True
            )
            raise click.Abort()
        except Exception as e:
            click.echo(f"\n✗ Error storing chunks: {e}", err=True)
            raise click.Abort()
    else:
        click.echo("No chunks to store.")


if __name__ == "__main__":
    ingest()
