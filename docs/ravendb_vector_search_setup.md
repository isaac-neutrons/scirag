# RavenDB Vector Search Setup Guide

## Overview

This guide provides step-by-step instructions to enable real vector search functionality in the scirag application. Currently, the search uses a fallback mechanism that returns documents without semantic similarity ranking.

## Current Status

- ✅ Documents are stored with embeddings (768-dimensional vectors from nomic-embed-text)
- ✅ Search command works but uses fallback query
- ❌ Vector search index doesn't exist
- ❌ Semantic similarity scores not calculated

## Document Structure

Documents in the `DocumentChunks` collection have this structure:

```json
{
  "id": "statement_of_resch34347.pdf_chunk_0",
  "source_filename": "statement_of_resch34347.pdf",
  "chunk_index": 0,
  "text": "The actual text content...",
  "embedding": [0.0592537, -0.0119797, ...], // 768-dimensional vector
  "metadata": {
    "file_size": 123456,
    "page_count": 5,
    ...
  }
}
```

## Option 1: Manual Index Creation via RavenDB Studio (Recommended for Testing)

### Step 1: Access RavenDB Studio

1. Open your browser and navigate to: **http://localhost:8080**
2. Click on your database: **scirag**

### Step 2: Create a Static Index

1. In the left sidebar, click **"Indexes"**
2. Click **"New Index"** button
3. Select **"Static Index"**

### Step 3: Define the Index

Enter the following index definition:

**Index Name:** `DocumentChunks/ByEmbedding`

**Map Function:**
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

**Note:** RavenDB 5.x and 6.x have different approaches to vector search. The above is a simplified version. You may need to adjust based on your RavenDB version.

### Step 4: Save and Wait for Indexing

1. Click **"Save"**
2. Wait for the index to complete processing (status will show "Up-to-date")
3. This may take a few seconds to minutes depending on the number of documents

### Step 5: Test the Index

Run a search to verify it works:

```bash
scirag-search "electrode" --top-k 3
```

If successful, you should see similarity scores instead of 0.0000.

## Option 2: Programmatic Index Creation (Production Approach)

### Step 1: Update the `ensure_index_exists()` Function

The current implementation in `src/scirag/service/database.py` is a placeholder. Here's how to implement it:

```python
def ensure_index_exists(store: DocumentStore) -> None:
    """Ensure the vector search index exists in RavenDB.
    
    Creates a static index named 'DocumentChunks/ByEmbedding' for vector search
    on the embedding field of DocumentChunks collection.
    """
    import requests
    
    index_name = "DocumentChunks/ByEmbedding"
    
    # Check if index exists
    url = f"{store.url}/databases/{store.database}/indexes"
    response = requests.get(url)
    
    if response.status_code == 200:
        existing_indexes = response.json()
        if index_name in existing_indexes:
            return  # Index already exists
    
    # Create the index
    index_definition = {
        "Name": index_name,
        "Maps": [
            """
            from chunk in docs.DocumentChunks
            select new {
                source_filename = chunk.source_filename,
                chunk_index = chunk.chunk_index,
                text = chunk.text,
                embedding = CreateSpatialField(chunk.embedding),
                metadata = chunk.metadata
            }
            """
        ],
        "Type": "Map"
    }
    
    create_url = f"{store.url}/databases/{store.database}/admin/indexes"
    response = requests.put(create_url, json=index_definition)
    response.raise_for_status()
```

### Step 2: Test the Implementation

Run the ingestion process to trigger index creation:

```bash
scirag-ingest documents/
```

### Step 3: Verify the Index

Check that the index was created:

```bash
python3 -c "
import requests
response = requests.get('http://localhost:8080/databases/scirag/indexes')
print('Indexes:', response.json())
"
```

## Option 3: RavenDB Vector Search Extension (RavenDB 6.0+)

If using RavenDB 6.0 or later, you can use the native vector search capabilities:

### Step 1: Enable Vector Search

In RavenDB Studio:
1. Go to **Settings** → **Features**
2. Enable **"Vector Search"** (if available in your version)

### Step 2: Create Vector Index

The index definition needs to specify the vector field explicitly:

```csharp
from chunk in docs.DocumentChunks
select new {
    source_filename = chunk.source_filename,
    chunk_index = chunk.chunk_index,
    text = chunk.text,
    // Vector field with dimensions
    embedding = chunk.embedding.ToArray(),
    metadata = chunk.metadata
}
```

### Step 3: Configure Vector Field

In the index configuration:
1. Set **embedding** field type to **"Vector"**
2. Specify dimensions: **768**
3. Choose distance metric: **"Cosine"** (recommended for text embeddings)

## Verification Steps

After creating the index, verify it works:

### 1. Check Index Status

```bash
python3 -c "
import requests
response = requests.get('http://localhost:8080/databases/scirag/indexes/DocumentChunks/ByEmbedding')
if response.status_code == 200:
    print('Index exists and is ready')
    print(response.json())
else:
    print(f'Index not found: {response.status_code}')
"
```

### 2. Test Vector Search Query

```bash
python3 -c "
from pyravendb.store.document_store import DocumentStore
import ollama

store = DocumentStore('http://localhost:8080', 'scirag')
store.initialize()

# Generate query embedding
response = ollama.embed(model='nomic-embed-text', input='electrode')
query_vector = response['embeddings'][0]

with store.open_session() as session:
    rql_query = '''
        from index 'DocumentChunks/ByEmbedding' 
        where vector.search(embedding, \$query_vector) 
        limit 3
    '''
    results = list(session.query(object_type=dict).raw_query(
        rql_query, 
        query_parameters={'query_vector': query_vector}
    ))
    print(f'Vector search returned {len(results)} results')
    
store.close()
"
```

### 3. Test via CLI

```bash
scirag-search "electrode materials" --top-k 5
```

Expected output should show non-zero similarity scores:
```
✅ Found 5 result(s):

1. [statement_of_resch34347.pdf - chunk #0] (score: 0.8542)
   Text about electrodes...

2. [statement_of_resch34347.pdf - chunk #2] (score: 0.7891)
   More relevant content...
```

## Troubleshooting

### Issue: Index Creation Fails

**Symptoms:** Error when creating index through API or Studio

**Solutions:**
1. Check RavenDB version: `docker logs <ravendb-container>` or check Studio
2. Verify embedding field exists in documents
3. Check RavenDB logs for specific error messages

### Issue: Vector Search Returns No Results

**Symptoms:** Index exists but search returns empty list

**Solutions:**
1. Verify index is "Up-to-date" (not stale)
2. Check if documents have embeddings: `scirag-count`
3. Verify embedding dimensions match (should be 768)
4. Re-ingest documents if embedding format changed

### Issue: Scores Are Still 0.0000

**Symptoms:** Results returned but scores show 0.0000

**Solutions:**
1. This indicates fallback query is still being used
2. Verify index name matches exactly: `DocumentChunks/ByEmbedding`
3. Check RQL query syntax in `database.py`
4. Confirm vector search is supported in your RavenDB version

### Issue: Connection Refused

**Symptoms:** Cannot access http://localhost:8080

**Solutions:**
1. Check if RavenDB is running: `docker ps | grep raven`
2. Start RavenDB: `docker start <container-name>`
3. Check port mapping in docker configuration

## RavenDB Version Considerations

### RavenDB 5.x
- Limited native vector search support
- May require custom similarity calculations
- Consider upgrading to 6.x for better vector support

### RavenDB 6.x
- Native vector search support
- Dedicated vector field types
- Better performance for large embedding collections
- Recommended for production use

## Next Steps After Setup

Once vector search is working:

1. **Update Documentation:**
   - Remove "limitation" notes from developer_notes.md
   - Document the index creation process
   - Update example outputs with real scores

2. **Performance Optimization:**
   - Monitor index performance with large document sets
   - Consider caching frequently searched queries
   - Adjust `top_k` default based on use case

3. **Enhanced Features:**
   - Add metadata filtering to searches
   - Implement query result caching
   - Add relevance feedback mechanisms

4. **Testing:**
   - Add integration tests for vector search
   - Test with various query types
   - Benchmark search performance

## References

- RavenDB Documentation: https://ravendb.net/docs
- RavenDB Indexes: https://ravendb.net/docs/article-page/6.0/csharp/indexes/what-are-indexes
- Vector Search in RavenDB: Check your version's documentation
- pyravendb Documentation: https://github.com/ravendb/ravendb-python-client

## Summary

To enable real vector search:

1. **Manual (Quick):** Create index via RavenDB Studio at http://localhost:8080
2. **Programmatic (Production):** Implement `ensure_index_exists()` with HTTP API calls
3. **Verify:** Test with `scirag-search` and check for non-zero similarity scores

Current fallback works for testing but doesn't provide semantic relevance ranking. Real vector search will significantly improve search quality by returning documents based on semantic similarity to the query.
