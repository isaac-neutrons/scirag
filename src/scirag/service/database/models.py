"""Data models for RavenDB document storage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class DocumentChunk:
    """A document chunk with embedding for vector search.

    This class represents a chunk of text from a document, along with its
    embedding vector and metadata. It is designed to be stored in RavenDB
    and used for vector similarity search.

    Note: eq=False ensures each instance is unique and hashable by identity,
    which is required for RavenDB's session entity tracking.

    Attributes:
        Id: RavenDB document ID (set by RavenDB or manually)
        source_filename: Original document filename
        chunk_index: Index of this chunk in the document
        text: The text content of the chunk
        embedding: Vector embedding of the text
        metadata: Additional metadata (file_size, etc.)
        collection: Collection name for grouping documents
    """

    Id: str | None = None
    source_filename: str = ""
    chunk_index: int = 0
    text: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    collection: str = "DocumentChunks"

    def __hash__(self) -> int:
        """Hash by object identity for RavenDB session tracking."""
        return id(self)
