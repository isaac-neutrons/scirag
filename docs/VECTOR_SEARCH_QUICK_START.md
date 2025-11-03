# Quick Reference: Enable Vector Search

## Current Status
- ❌ Vector search disabled (using fallback)
- ✅ Documents stored with embeddings
- ✅ Search returns results (no semantic ranking)
- ⚠️ Scores show 0.0000

## Fastest Method (5 minutes)

### Via RavenDB Studio UI

1. **Open RavenDB Studio**
   ```
   http://localhost:8080
   ```

2. **Navigate to Indexes**
   - Click on database: `scirag`
   - Left sidebar → `Indexes`
   - Click `New Index` → `Static Index`

3. **Create Index**
   - **Name:** `DocumentChunks/ByEmbedding`
   - **Map:**
   ```csharp
   from chunk in docs.DocumentChunks
   select new {
       source_filename = chunk.source_filename,
       chunk_index = chunk.chunk_index,
       text = chunk.text,
       embedding = CreateSpatialField(chunk.embedding),
       metadata = chunk.metadata
   }
   ```

4. **Save and Wait**
   - Click `Save`
   - Wait for status: "Up-to-date"

5. **Test**
   ```bash
   scirag-search "electrode" --top-k 3
   ```
   
   Should show non-zero scores like: `(score: 0.8542)`

## Verification Commands

```bash
# Check if index exists
python3 -c "import requests; print(requests.get('http://localhost:8080/databases/scirag/indexes').json())"

# Test search
scirag-search "test query" --top-k 5

# Check document count
scirag-count
```

## If It Still Doesn't Work

See full troubleshooting guide: `docs/ravendb_vector_search_setup.md`

Common issues:
- Index is "Stale" (wait for indexing to complete)
- RavenDB version doesn't support vector search (upgrade to 6.x)
- Index name mismatch (must be exactly `DocumentChunks/ByEmbedding`)

## Alternative: Programmatic Setup

If you want to automate index creation, implement the HTTP API approach:

```python
# In src/scirag/service/database.py, update ensure_index_exists()
import requests

def ensure_index_exists(store: DocumentStore) -> None:
    index_definition = {
        "Name": "DocumentChunks/ByEmbedding",
        "Maps": [
            """from chunk in docs.DocumentChunks
            select new {
                source_filename = chunk.source_filename,
                chunk_index = chunk.chunk_index,
                text = chunk.text,
                embedding = CreateSpatialField(chunk.embedding),
                metadata = chunk.metadata
            }"""
        ],
        "Type": "Map"
    }
    
    url = f"{store.url}/databases/{store.database}/admin/indexes"
    response = requests.put(url, json=index_definition)
    response.raise_for_status()
```

Then run:
```bash
scirag-ingest documents/
```

## What You'll Get

**Before (current fallback):**
```
1. [paper.pdf - chunk #0] (score: 0.0000)
   Some text content...
```

**After (real vector search):**
```
1. [paper.pdf - chunk #3] (score: 0.9245)
   Highly relevant content matching your query...

2. [paper.pdf - chunk #7] (score: 0.8891)
   Another relevant section...
```

## Documentation

- Full setup guide: `docs/ravendb_vector_search_setup.md`
- Developer notes: `docs/developer_notes.md` (Step 10.1)
- RavenDB docs: https://ravendb.net/docs
