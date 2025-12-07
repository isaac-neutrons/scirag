"""Command-line interface for SciRAG using Click."""

import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from scirag.client.cli_helpers import (
    ensure_database_exists,
    format_search_result,
    get_database_info,
)
from scirag.client.ingest import extract_chunks_from_pdf
from scirag.constants import DEFAULT_LOCAL_MCP_URL, get_embedding_model
from scirag.service.database import (
    database_exists,
    delete_database,
    search_documents,
)
from scirag.service.mcp_helpers import call_mcp_tool

# Load environment variables
load_dotenv()


@click.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Ollama embedding model to use (default: from EMBEDDING_MODEL env or 'nomic-embed-text')",
)
@click.option(
    "--create-database",
    "create_database_flag",
    is_flag=True,
    default=False,
    help="Create the RavenDB database if it doesn't exist",
)
@click.option(
    "--collection",
    type=str,
    default="DocumentChunks",
    help="Collection name for organizing documents in the database (default: 'DocumentChunks')",
)
def ingest(
    directory: Path,
    embedding_model: str | None,
    create_database_flag: bool,
    collection: str,
) -> None:
    """Ingest PDF files from DIRECTORY into the SciRAG knowledge base.

    Example:
        scirag-ingest documents/
        scirag-ingest documents/ --create-database
        scirag-ingest documents/ --collection research-papers
    """
    ensure_database_exists(create_if_missing=create_database_flag, directory=str(directory))

    model = embedding_model or get_embedding_model()
    pdf_files = list(directory.glob("*.pdf"))

    if not pdf_files:
        click.echo(f"No PDF files found in '{directory}'")
        return

    click.echo(f"Found {len(pdf_files)} PDF file(s)")
    click.echo(f"Using embedding model: {model}")
    click.echo(f"Collection: {collection}\n")

    local_mcp_server_url = os.getenv("LOCAL_MCP_SERVER_URL", DEFAULT_LOCAL_MCP_URL)

    # Process each PDF and collect all chunks
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            chunks = extract_chunks_from_pdf(pdf_path, collection)
            all_chunks.extend(chunks)
            click.echo(f"  ✓ Extracted {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            click.echo(f"  ✗ Error processing {pdf_path.name}: {e}", err=True)

    if not all_chunks:
        click.echo("No chunks to store.")
        return

    click.echo(f"\nStoring {len(all_chunks)} chunks via MCP server (collection: '{collection}')...")
    try:
        store_result = asyncio.run(
            call_mcp_tool(
                local_mcp_server_url,
                "store_document_chunks",
                {"chunks": all_chunks, "collection": collection},
            )
        )
        if store_result.get("success"):
            chunks_stored = store_result.get('chunks_stored', 0)
            click.echo(f"✓ Ingestion complete! Stored {chunks_stored} chunks.")
        else:
            error_msg = store_result.get('message', 'Unknown error')
            click.echo(f"\n✗ Error storing chunks: {error_msg}", err=True)
            raise click.Abort()
    except click.Abort:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "database" in error_msg and ("not" in error_msg or "exist" in error_msg):
            click.echo("\n✗ Error: Database does not exist!", err=True)
            click.echo(f"  Run: scirag-ingest {directory} --create-database", err=True)
        else:
            click.echo(f"\n✗ Error storing chunks: {e}", err=True)
        raise click.Abort()


@click.command()
def count() -> None:
    """Show the number of document chunks in the database.

    Example:
        scirag-count
    """
    ensure_database_exists()
    _, _, doc_count = get_database_info()
    if doc_count is not None:
        click.echo(f"📊 Database contains {doc_count} document chunk(s)")
    else:
        click.echo("✗ Error counting documents", err=True)
        raise click.Abort()


@click.command()
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Ollama embedding model to use (default: from EMBEDDING_MODEL env or 'nomic-embed-text')",
)
def search(query: str, top_k: int, embedding_model: str | None) -> None:
    """Search for similar documents using vector search.

    QUERY is the text to search for.

    Example:
        scirag-search "quantum mechanics"
        scirag-search "machine learning" --top-k 3
    """
    ensure_database_exists()

    click.echo(f"🔍 Searching for: '{query}'")
    click.echo(f"   Returning top {top_k} results...\n")

    try:
        results = search_documents(query, top_k=top_k, embedding_model=embedding_model)

        if not results:
            click.echo("No results found.")
            return

        click.echo(f"✅ Found {len(results)} result(s):\n")
        for i, result in enumerate(results, 1):
            click.echo(format_search_result(i, result))

    except ConnectionError as e:
        click.echo(f"✗ Connection error: {e}", err=True)
        click.echo("\nPlease ensure Ollama and RavenDB are running.", err=True)
        raise click.Abort()
    except ValueError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"✗ Unexpected error: {e}", err=True)
        raise click.Abort()


@click.command()
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt")
def delete_db(yes: bool) -> None:
    """Delete the RavenDB database and all its contents.

    WARNING: This is irreversible and deletes all documents, embeddings, and indexes.

    Example:
        scirag-delete-db          # Will prompt for confirmation
        scirag-delete-db --yes    # Skip confirmation
    """
    url, db_name, doc_count = get_database_info()

    if not database_exists():
        click.echo(f"✓ Database '{db_name}' does not exist at {url}")
        return

    if not yes:
        click.echo(f"⚠️  WARNING: You are about to delete the database '{db_name}'")
        click.echo(f"   Location: {url}\n")
        click.echo("This will permanently delete:")
        click.echo("  • All ingested documents")
        click.echo("  • All embeddings")
        click.echo("  • All indexes\n")

        if doc_count is not None:
            click.echo(f"📊 Current database contains: {doc_count} document chunk(s)\n")

        if not click.confirm("Are you sure you want to proceed?", default=False):
            click.echo("Deletion cancelled.")
            return

    click.echo(f"🗑️  Deleting database '{db_name}'...")
    try:
        delete_database()
        click.echo(f"✓ Database '{db_name}' successfully deleted!")
        click.echo("\nTo create a new database, run:")
        click.echo("  scirag-ingest <directory> --create-database")
    except Exception as e:
        click.echo(f"✗ Error deleting database: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    ingest()
