"""Command-line interface for SciRAG using Click."""

import os
from pathlib import Path

import click

from scirag.client.ingest import ingest_pdf, store_chunks
from scirag.service.database import (
    count_documents,
    create_database,
    database_exists,
    delete_database,
    search_documents,
)


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
        except Exception as e:
            # Check if it's a database not found error
            error_msg = str(e).lower()
            if "database" in error_msg and ("not" in error_msg or "exist" in error_msg):
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
            click.echo(f"\n✗ Error storing chunks: {e}", err=True)
            raise click.Abort()
    else:
        click.echo("No chunks to store.")


@click.command()
def count() -> None:
    """Show the number of document chunks in the database.

    Queries RavenDB and displays the total count of DocumentChunk documents.

    Example:
        scirag-count
    """
    # Check if database exists
    if not database_exists():
        click.echo("✗ Error: Database does not exist!", err=True)
        click.echo("\nPlease create the database first using:", err=True)
        click.echo("  scirag-ingest <directory> --create-database", err=True)
        raise click.Abort()

    try:
        doc_count = count_documents()
        click.echo(f"📊 Database contains {doc_count} document chunk(s)")
    except Exception as e:
        click.echo(f"✗ Error counting documents: {e}", err=True)
        click.echo("\nPlease ensure RavenDB is running and accessible.", err=True)
        raise click.Abort()


@click.command()
@click.argument("query", type=str)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top similar documents to return (default: 5)",
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
def search(query: str, top_k: int, embedding_model: str | None) -> None:
    """Search for similar documents using vector search.

    Performs semantic search on the document chunks stored in RavenDB
    and displays the top-k most similar results.

    QUERY is the text to search for.

    Example:
        scirag-search "quantum mechanics"
        scirag-search "machine learning algorithms" --top-k 3
        scirag-search "neural networks" --embedding-model custom-model
    """
    # Check if database exists
    if not database_exists():
        click.echo("✗ Error: Database does not exist!", err=True)
        click.echo("\nPlease create the database first using:", err=True)
        click.echo("  scirag-ingest <directory> --create-database", err=True)
        raise click.Abort()

    click.echo(f"🔍 Searching for: '{query}'")
    click.echo(f"   Returning top {top_k} results...")
    click.echo()

    try:
        results = search_documents(query, top_k=top_k, embedding_model=embedding_model)

        if not results:
            click.echo("No results found.")
            return

        click.echo(f"✅ Found {len(results)} result(s):\n")

        for i, result in enumerate(results, 1):
            score = result.get("score", 0.0)
            source = result["source"]
            chunk_idx = result["chunk_index"]
            content = result["content"]

            # Truncate content for display
            max_content_length = 200
            display_content = (
                content[:max_content_length] + "..."
                if len(content) > max_content_length
                else content
            )

            click.echo(f"{i}. [{source} - chunk #{chunk_idx}] (score: {score:.4f})")
            click.echo(f"   {display_content}")
            click.echo()

    except ConnectionError as e:
        click.echo(f"✗ Connection error: {e}", err=True)
        click.echo("\nPlease ensure Ollama and RavenDB are running.", err=True)
        raise click.Abort()
    except ValueError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"✗ Unexpected error: {e}", err=True)
        click.echo("\nPlease check your configuration and try again.", err=True)
        raise click.Abort()


@click.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
def delete_db(yes: bool) -> None:
    """Delete the RavenDB database and all its contents.

    WARNING: This operation is irreversible and will delete all ingested documents,
    embeddings, and indexes from the database.

    Example:
        scirag-delete-db          # Will prompt for confirmation
        scirag-delete-db --yes    # Skip confirmation
    """
    # Get database info
    from scirag.service.database import RavenDBConfig

    url = RavenDBConfig.get_url()
    db_name = RavenDBConfig.get_database_name()

    # Check if database exists
    if not database_exists():
        click.echo(f"✓ Database '{db_name}' does not exist at {url}")
        click.echo("Nothing to delete.")
        return

    # Confirm deletion unless --yes flag is provided
    if not yes:
        click.echo(f"⚠️  WARNING: You are about to delete the database '{db_name}'")
        click.echo(f"   Location: {url}")
        click.echo()
        click.echo("This will permanently delete:")
        click.echo("  • All ingested documents")
        click.echo("  • All embeddings")
        click.echo("  • All indexes")
        click.echo("  • All metadata")
        click.echo()

        # Get document count
        try:
            doc_count = count_documents()
            click.echo(f"📊 Current database contains: {doc_count} document chunk(s)")
            click.echo()
        except Exception:
            pass

        if not click.confirm("Are you sure you want to proceed?", default=False):
            click.echo("Deletion cancelled.")
            return

    # Delete the database
    click.echo(f"🗑️  Deleting database '{db_name}'...")
    try:
        delete_database()
        click.echo(f"✓ Database '{db_name}' successfully deleted!")
        click.echo()
        click.echo("To create a new database, run:")
        click.echo(f"  scirag-ingest <directory> --create-database")
    except Exception as e:
        click.echo(f"✗ Error deleting database: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    ingest()
