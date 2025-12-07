"""Helper functions for CLI commands."""

import click

from scirag.service.database import (
    count_documents,
    create_database,
    database_exists,
)


def ensure_database_exists(
    create_if_missing: bool = False,
    directory: str | None = None,
) -> bool:
    """Check if database exists, optionally create it.

    Args:
        create_if_missing: If True, attempt to create the database
        directory: Directory path for error message context

    Returns:
        True if database exists (or was created), False otherwise

    Raises:
        click.Abort: If database doesn't exist and can't be created
    """
    if database_exists():
        return True

    if create_if_missing:
        click.echo("Database does not exist. Creating database...")
        try:
            create_database()
            click.echo("✓ Database created successfully!")
            return True
        except Exception as e:
            click.echo(f"✗ Failed to create database: {e}", err=True)
            click.echo("\nPlease ensure RavenDB is running and accessible.", err=True)
            raise click.Abort()

    # Database doesn't exist and we're not creating it
    click.echo("✗ Error: Database does not exist!", err=True)
    click.echo("\nPlease create the database first using:", err=True)
    if directory:
        click.echo(f"  scirag-ingest {directory} --create-database", err=True)
    else:
        click.echo("  scirag-ingest <directory> --create-database", err=True)
    raise click.Abort()


def format_search_result(index: int, result: dict, max_length: int = 200) -> str:
    """Format a search result for display.

    Args:
        index: Result number (1-based)
        result: Search result dict with score, source, chunk_index, content
        max_length: Maximum content length before truncation

    Returns:
        Formatted string for display
    """
    score = result.get("score", 0.0)
    source = result["source"]
    chunk_idx = result["chunk_index"]
    content = result["content"]

    display_content = (
        content[:max_length] + "..." if len(content) > max_length else content
    )

    lines = [
        f"{index}. [{source} - chunk #{chunk_idx}] (score: {score:.4f})",
        f"   {display_content}",
        "",
    ]
    return "\n".join(lines)


def get_database_info() -> tuple[str, str, int | None]:
    """Get database connection info and document count.

    Returns:
        Tuple of (url, database_name, document_count or None if error)
    """
    from scirag.service.database import RavenDBConfig

    url = RavenDBConfig.get_url()
    db_name = RavenDBConfig.get_database_name()

    doc_count = None
    try:
        doc_count = count_documents()
    except Exception:
        pass

    return url, db_name, doc_count
