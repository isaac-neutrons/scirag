"""End-to-end integration tests for the complete RAG pipeline."""

import pytest


class TestRAGPipelineIntegration:
    """End-to-end tests for the complete RAG workflow."""

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    @pytest.mark.slow
    def test_pdf_to_embeddings_pipeline(self, sample_pdf, ollama_service):
        """Test complete pipeline: PDF → chunks → embeddings."""
        from scirag.client.ingest import extract_chunks_from_pdf
        
        # Extract chunks from PDF
        chunks = extract_chunks_from_pdf(sample_pdf, collection="e2e-test")
        assert len(chunks) > 0
        
        # Generate embeddings for chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = ollama_service.generate_embeddings(texts, "nomic-embed-text")
        
        # Verify embeddings generated
        assert len(embeddings) == len(chunks)
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        # Verify complete chunk structure
        for chunk in chunks:
            assert "text" in chunk
            assert "embedding" in chunk
            assert "metadata" in chunk
            assert len(chunk["embedding"]) == 768

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    @pytest.mark.requires_ravendb
    @pytest.mark.slow
    def test_full_rag_pipeline(self, sample_pdf, ollama_service, ravendb_store):
        """Test complete RAG pipeline: PDF → chunks → embeddings → storage → search → retrieval."""
        from scirag.client.ingest import extract_chunks_from_pdf
        from scirag.service.database import cosine_similarity
        
        # Step 1: Extract chunks from PDF
        chunks = extract_chunks_from_pdf(sample_pdf, collection="full-rag-test")
        assert len(chunks) > 0
        
        # Step 2: Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = ollama_service.generate_embeddings(texts, "nomic-embed-text")
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        # Step 3: Store in RavenDB
        with ravendb_store.open_session() as session:
            for chunk in chunks:
                chunk_id = f"full_rag_test/{chunk['chunk_index']}"
                session.store(chunk, chunk_id)
            session.save_changes()
        
        # Step 4: Generate query embedding
        query = "What is Python used for?"
        query_embedding = ollama_service.generate_embeddings([query], "nomic-embed-text")[0]
        
        # Step 5: Retrieve and calculate similarities
        with ravendb_store.open_session() as session:
            retrieved_chunks = []
            for i in range(len(chunks)):
                chunk_id = f"full_rag_test/{i}"
                chunk = session.load(chunk_id)
                if chunk:
                    similarity = cosine_similarity(query_embedding, chunk["embedding"])
                    retrieved_chunks.append((similarity, chunk))
            
            # Sort by similarity
            retrieved_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Step 6: Verify results
        assert len(retrieved_chunks) > 0
        
        # Most similar chunk should contain relevant content
        best_match = retrieved_chunks[0][1]
        combined_text = " ".join(c[1]["text"] for c in retrieved_chunks)
        
        # Verify relevant content is present
        assert "Python" in combined_text or "programming" in combined_text

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    def test_semantic_search_relevance(self, ollama_service):
        """Test that semantic search returns relevant results."""
        from scirag.service.database import cosine_similarity
        
        # Create a mini corpus
        documents = [
            "Python is a programming language used for data science and machine learning",
            "JavaScript is used for web development and frontend applications",
            "SQL is used for database queries and data manipulation",
            "Cooking pasta requires boiling water and adding salt",
        ]
        
        # Generate embeddings for documents
        doc_embeddings = ollama_service.generate_embeddings(documents, "nomic-embed-text")
        
        # Query about programming
        query = "programming languages for software development"
        query_embedding = ollama_service.generate_embeddings([query], "nomic-embed-text")[0]
        
        # Calculate similarities
        similarities = [
            (i, cosine_similarity(query_embedding, doc_emb))
            for i, doc_emb in enumerate(doc_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Top results should be programming-related (indices 0, 1, 2), not cooking (index 3)
        top_3_indices = [s[0] for s in similarities[:3]]
        assert 3 not in top_3_indices  # Cooking should not be in top 3
        assert 0 in top_3_indices or 1 in top_3_indices  # Programming languages should be
        
        # Python/programming doc should have higher similarity than cooking
        python_sim = similarities[0][1] if similarities[0][0] == 0 else similarities[[s[0] for s in similarities].index(0)][1]
        cooking_sim = similarities[3][1] if similarities[3][0] == 3 else similarities[[s[0] for s in similarities].index(3)][1]
        assert python_sim > cooking_sim

    @pytest.mark.integration
    @pytest.mark.requires_ollama
    def test_chunk_overlap_preserves_context(self, sample_pdf, ollama_service):
        """Test that chunk overlap helps preserve semantic context."""
        from scirag.client.ingest import extract_chunks_from_pdf, chunk_text
        
        # Extract text and create chunks with overlap
        from scirag.client.ingest import extract_text_from_pdf
        text = extract_text_from_pdf(sample_pdf)
        
        # Create chunks with different overlap settings
        chunks_no_overlap = chunk_text(text, chunk_size=100, overlap=0)
        chunks_with_overlap = chunk_text(text, chunk_size=100, overlap=20)
        
        # Chunks with overlap should have more chunks (due to redundancy)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
        
        # Verify overlap: last words of chunk N should appear in first words of chunk N+1
        if len(chunks_with_overlap) > 1:
            chunk_0_words = chunks_with_overlap[0].split()
            chunk_1_words = chunks_with_overlap[1].split()
            
            # Some words from end of chunk 0 should be in beginning of chunk 1
            overlap_found = any(
                word in chunk_1_words[:25]  # Check first 25 words of chunk 1
                for word in chunk_0_words[-25:]  # Against last 25 words of chunk 0
            )
            assert overlap_found
