# Test Fixtures

This directory contains test data for the SciRAG test suite.

## Files

### `sample.pdf`
A simple 2-page PDF document used for testing PDF text extraction and chunking.

**Content:**
- Page 1: Information about Python programming and machine learning in scientific research
- Page 2: Information about vector embeddings and semantic search

**Usage:**
```python
from pathlib import Path

test_pdf = Path("tests/fixtures/sample.pdf")
text = extract_text_from_pdf(test_pdf)
chunks = extract_chunks_from_pdf(test_pdf)
```

### `sample_embeddings.json`
Sample embeddings for testing vector similarity and search operations.

**Structure:**
- Simplified 8-dimensional embeddings (real embeddings are 768-dimensional)
- Contains embeddings for programming-related and unrelated topics
- Used to test cosine similarity and vector search

**Usage:**
```python
import json

with open("tests/fixtures/sample_embeddings.json") as f:
    data = json.load(f)
    embeddings = data["samples"]
```

### `test_chunks.json`
Sample document chunks structured for RavenDB storage testing.

**Structure:**
- Pre-chunked text from the sample PDF
- Includes metadata (file size, page count, etc.)
- Ready to be stored in RavenDB for testing

**Usage:**
```python
import json

with open("tests/fixtures/test_chunks.json") as f:
    data = json.load(f)
    chunks = data["chunks"]
```

## Adding New Fixtures

When adding test data:

1. Keep fixtures minimal but realistic
2. Include a description in this README
3. Use consistent naming (`test_*.json`, `sample_*.pdf`)
4. Add fixture loaders to `conftest.py` if reused across tests

## Fixture Principles

- **Real Data**: Use actual file formats (real PDFs, not mocks)
- **Minimal Size**: Keep files small to maintain fast test execution
- **Representative**: Data should reflect real-world usage patterns
- **Self-Documenting**: Include metadata explaining the fixture purpose
