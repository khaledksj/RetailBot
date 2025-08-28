"""
Tests for retrieval pipeline functionality and vector search.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.rag import RAGPipeline
from app.core.db import InMemoryVectorStore
from app.core.models import Chunk, ChatResponse, Source
from app.core.llm import EmbeddingService, ChatService


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    async def rag_pipeline(self):
        with patch('app.core.rag.get_vector_store') as mock_get_store:
            mock_store = InMemoryVectorStore()
            await mock_store.initialize()
            mock_get_store.return_value = mock_store
            
            pipeline = RAGPipeline()
            return pipeline, mock_store
    
    @pytest.fixture
    def sample_chunks_with_embeddings(self):
        """Create sample chunks with embeddings for testing."""
        return [
            (Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                filename="installation_manual.pdf",
                page=5,
                chunk_idx=0,
                content="To install the product, first ensure all power is disconnected. Remove the old unit by unscrewing the mounting brackets. Install the new unit by reversing the process.",
                content_tokens=30,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ), 0.95),
            (Chunk(
                chunk_id="chunk_2",
                doc_id="doc_1",
                filename="installation_manual.pdf",
                page=12,
                chunk_idx=1,
                content="Safety precautions: Always wear protective equipment when handling electrical components. Ensure the main circuit breaker is off before beginning work.",
                content_tokens=25,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            ), 0.87),
            (Chunk(
                chunk_id="chunk_3",
                doc_id="doc_2",
                filename="troubleshooting_guide.pdf",
                page=3,
                chunk_idx=0,
                content="If the unit fails to start, check the power connection first. Verify that the circuit breaker has not tripped. Check all wire connections for loose contacts.",
                content_tokens=28,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
            ), 0.82),
            (Chunk(
                chunk_id="chunk_4",
                doc_id="doc_2",
                filename="troubleshooting_guide.pdf",
                page=8,
                chunk_idx=1,
                content="Common error codes: E01 indicates power supply failure. E02 indicates sensor malfunction. E03 indicates communication error.",
                content_tokens=22,
                embedding=[0.4, 0.5, 0.6, 0.7, 0.8]
            ), 0.75)
        ]
    
    @pytest.mark.asyncio
    async def test_maximal_marginal_relevance(self, rag_pipeline):
        """Test MMR reranking functionality."""
        pipeline, mock_store = rag_pipeline
        
        # Create chunks with similar embeddings (to test diversity)
        chunks_with_scores = [
            (Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                filename="test.pdf",
                page=1,
                chunk_idx=0,
                content="Content about installation process",
                content_tokens=20,
                embedding=[1.0, 0.0, 0.0]  # Very similar to query
            ), 0.95),
            (Chunk(
                chunk_id="chunk_2", 
                doc_id="doc_1",
                filename="test.pdf",
                page=1,
                chunk_idx=1,
                content="More content about installation process",
                content_tokens=22,
                embedding=[0.9, 0.1, 0.0]  # Similar to chunk_1 and query
            ), 0.90),
            (Chunk(
                chunk_id="chunk_3",
                doc_id="doc_1", 
                filename="test.pdf",
                page=2,
                chunk_idx=0,
                content="Content about safety procedures",
                content_tokens=21,
                embedding=[0.0, 0.0, 1.0]  # Different topic
            ), 0.75)
        ]
        
        query_embedding = [1.0, 0.0, 0.0]
        
        result = pipeline._maximal_marginal_relevance(
            chunks_with_scores,
            query_embedding,
            top_n=2
        )
        
        assert len(result) == 2
        # First should be highest scoring
        assert result[0][0].chunk_id == "chunk_1"
        # Second should be diverse (chunk_3) rather than similar (chunk_2)
        assert result[1][0].chunk_id == "chunk_3"
    
    def test_format_context(self, rag_pipeline):
        """Test context formatting for LLM prompt."""
        pipeline, _ = rag_pipeline
        
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                filename="manual.pdf",
                page=5,
                chunk_idx=0,
                content="Installation instructions here.",
                content_tokens=20,
                embedding=[]
            ),
            Chunk(
                chunk_id="chunk_2",
                doc_id="doc_1",
                filename="guide.pdf", 
                page=3,
                chunk_idx=0,
                content="Safety procedures here.",
                content_tokens=18,
                embedding=[]
            )
        ]
        
        context = pipeline._format_context(chunks)
        
        expected = "[manual.pdf – p.5]\nInstallation instructions here.\n\n[guide.pdf – p.3]\nSafety procedures here."
        assert context == expected
    
    def test_extract_sources(self, rag_pipeline):
        """Test source extraction from chunks."""
        pipeline, _ = rag_pipeline
        
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                filename="manual.pdf",
                page=5,
                chunk_idx=0,
                content="Long content that should be truncated for snippet display because it exceeds the typical length limit for previews",
                content_tokens=25,
                embedding=[]
            ),
            Chunk(
                chunk_id="chunk_2",
                doc_id="doc_1",
                filename="manual.pdf",
                page=5,  # Same page, should deduplicate
                chunk_idx=1,
                content="More content from same page.",
                content_tokens=20,
                embedding=[]
            ),
            Chunk(
                chunk_id="chunk_3",
                doc_id="doc_1",
                filename="guide.pdf",
                page=3,
                chunk_idx=0,
                content="Short content.",
                content_tokens=15,
                embedding=[]
            )
        ]
        
        sources = pipeline._extract_sources(chunks)
        
        # Should deduplicate same page sources
        assert len(sources) == 2
        
        # Check source details
        source1 = next(s for s in sources if s.filename == "manual.pdf")
        assert source1.page == 5
        assert source1.snippet.endswith("...")  # Should be truncated
        
        source2 = next(s for s in sources if s.filename == "guide.pdf")
        assert source2.page == 3
        assert not source2.snippet.endswith("...")  # Short, no truncation
    
    @pytest.mark.asyncio
    async def test_process_query_with_results(self, rag_pipeline, sample_chunks_with_embeddings):
        """Test complete query processing with results."""
        pipeline, mock_store = rag_pipeline
        
        # Mock the vector store search
        with patch.object(mock_store, 'similarity_search') as mock_search:
            mock_search.return_value = sample_chunks_with_embeddings
            
            # Mock embedding service
            with patch.object(pipeline.embedding_service, 'create_embeddings') as mock_embed:
                mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
                
                # Mock chat service
                with patch.object(pipeline.chat_service, 'generate_response') as mock_chat:
                    mock_chat.return_value = "To install the product, first disconnect power as described in [installation_manual.pdf – p.5]. Follow safety precautions in [installation_manual.pdf – p.12]."
                    
                    response = await pipeline.process_query(
                        query="How do I install the product?",
                        session_id="test_session",
                        temperature=0.2
                    )
                    
                    assert isinstance(response, ChatResponse)
                    assert response.answer is not None
                    assert len(response.sources) > 0
                    assert response.session_id == "test_session"
                    
                    # Verify methods were called
                    mock_embed.assert_called_once()
                    mock_search.assert_called_once()
                    mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_query_no_results(self, rag_pipeline):
        """Test query processing when no relevant chunks are found."""
        pipeline, mock_store = rag_pipeline
        
        # Mock empty search results
        with patch.object(mock_store, 'similarity_search') as mock_search:
            mock_search.return_value = []
            
            with patch.object(pipeline.embedding_service, 'create_embeddings') as mock_embed:
                mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
                
                response = await pipeline.process_query(
                    query="What is the meaning of life?",
                    session_id="test_session"
                )
                
                assert isinstance(response, ChatResponse)
                assert response.answer == "I couldn't find this in the manuals."
                assert len(response.sources) == 0
                assert response.session_id == "test_session"
    
    @pytest.mark.asyncio
    async def test_process_query_long_input_truncation(self, rag_pipeline):
        """Test that long queries are truncated."""
        pipeline, mock_store = rag_pipeline
        
        # Create very long query
        long_query = "How do I install? " * 1000  # Repeat to exceed token limit
        
        with patch.object(pipeline.chat_service, 'count_tokens') as mock_count:
            mock_count.return_value = 5000  # Exceed max_input_tokens
            
            with patch.object(pipeline.chat_service, 'truncate_text') as mock_truncate:
                mock_truncate.return_value = "How do I install? " * 100  # Truncated version
                
                with patch.object(mock_store, 'similarity_search') as mock_search:
                    mock_search.return_value = []
                    
                    with patch.object(pipeline.embedding_service, 'create_embeddings') as mock_embed:
                        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
                        
                        response = await pipeline.process_query(
                            query=long_query,
                            session_id="test_session"
                        )
                        
                        # Verify truncation was called
                        mock_truncate.assert_called_once()
                        assert isinstance(response, ChatResponse)


class TestVectorSimilaritySearch:
    """Test vector similarity search functionality."""
    
    @pytest.fixture
    async def populated_vector_store(self):
        """Create vector store with test data."""
        store = InMemoryVectorStore()
        await store.initialize()
        
        # Add test chunks with known embeddings
        test_chunks = [
            Chunk(
                chunk_id="installation_chunk",
                doc_id="doc_1", 
                filename="installation.pdf",
                page=1,
                chunk_idx=0,
                content="Follow these steps to install the device safely.",
                content_tokens=20,
                embedding=[1.0, 0.0, 0.0]  # Installation vector
            ),
            Chunk(
                chunk_id="safety_chunk",
                doc_id="doc_1",
                filename="safety.pdf", 
                page=1,
                chunk_idx=0,
                content="Always wear protective gear when working.",
                content_tokens=18,
                embedding=[0.0, 1.0, 0.0]  # Safety vector
            ),
            Chunk(
                chunk_id="troubleshooting_chunk",
                doc_id="doc_2",
                filename="troubleshooting.pdf",
                page=1,
                chunk_idx=0,
                content="If device fails to start, check power connection.",
                content_tokens=22,
                embedding=[0.0, 0.0, 1.0]  # Troubleshooting vector
            )
        ]
        
        embeddings = [chunk.embedding for chunk in test_chunks]
        
        await store.store_document(
            filename="test_manuals.pdf",
            content_hash="test_hash",
            chunks=test_chunks,
            embeddings=embeddings
        )
        
        return store
    
    @pytest.mark.asyncio
    async def test_similarity_search_installation_query(self, populated_vector_store):
        """Test search for installation-related query."""
        # Query vector similar to installation embedding
        query_embedding = [0.9, 0.1, 0.0]
        
        results = await populated_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=3
        )
        
        assert len(results) == 3
        
        # First result should be installation chunk (highest similarity)
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_id == "installation_chunk"
        assert top_score > 0.8  # High similarity
        
        # Results should be sorted by score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio 
    async def test_similarity_search_safety_query(self, populated_vector_store):
        """Test search for safety-related query."""
        # Query vector similar to safety embedding  
        query_embedding = [0.1, 0.9, 0.0]
        
        results = await populated_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(results) == 2
        
        # First result should be safety chunk
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_id == "safety_chunk"
        assert top_score > 0.8
    
    @pytest.mark.asyncio
    async def test_similarity_search_empty_store(self):
        """Test search on empty vector store."""
        store = InMemoryVectorStore()
        await store.initialize()
        
        results = await store.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_similarity_search_zero_vector(self, populated_vector_store):
        """Test search with zero query vector."""
        query_embedding = [0.0, 0.0, 0.0]
        
        results = await populated_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=3
        )
        
        # Should return empty results due to zero norm
        assert results == []


class TestRetrievalChunkSelection:
    """Test chunk selection and retrieval quality."""
    
    @pytest.mark.asyncio
    async def test_retrieval_returns_expected_chunks(self):
        """Test that retrieval returns chunks with expected IDs for known queries."""
        # This test validates that the retrieval system returns
        # semantically relevant chunks for specific queries
        
        store = InMemoryVectorStore()
        await store.initialize()
        
        # Create chunks with embeddings that simulate semantic similarity
        installation_chunks = [
            Chunk(
                chunk_id="install_step_1",
                doc_id="manual_1",
                filename="installation_guide.pdf",
                page=2,
                chunk_idx=0,
                content="Step 1: Prepare the installation area by clearing all debris.",
                content_tokens=25,
                embedding=[0.8, 0.2, 0.1, 0.0, 0.0]  # Installation-focused
            ),
            Chunk(
                chunk_id="install_step_2", 
                doc_id="manual_1",
                filename="installation_guide.pdf",
                page=3,
                chunk_idx=1,
                content="Step 2: Mount the device using the provided brackets.",
                content_tokens=23,
                embedding=[0.9, 0.1, 0.0, 0.0, 0.0]  # Installation-focused
            ),
            Chunk(
                chunk_id="safety_warning",
                doc_id="manual_1",
                filename="safety_manual.pdf",
                page=1,
                chunk_idx=0,
                content="Warning: Always disconnect power before installation.",
                content_tokens=20,
                embedding=[0.3, 0.7, 0.0, 0.0, 0.0]  # Safety-focused
            ),
            Chunk(
                chunk_id="troubleshoot_power",
                doc_id="manual_2",
                filename="troubleshooting.pdf",
                page=5,
                chunk_idx=0,
                content="If device won't turn on, check power connections first.",
                content_tokens=28,
                embedding=[0.1, 0.1, 0.8, 0.0, 0.0]  # Troubleshooting-focused
            )
        ]
        
        embeddings = [chunk.embedding for chunk in installation_chunks]
        
        await store.store_document(
            filename="complete_manual.pdf",
            content_hash="complete_hash",
            chunks=installation_chunks,
            embeddings=embeddings
        )
        
        # Test installation query
        installation_query_embedding = [0.85, 0.15, 0.0, 0.0, 0.0]
        installation_results = await store.similarity_search(
            query_embedding=installation_query_embedding,
            top_k=3
        )
        
        # Should prioritize installation chunks
        installation_chunk_ids = [chunk.chunk_id for chunk, _ in installation_results]
        assert "install_step_1" in installation_chunk_ids[:2]  # Top 2 results
        assert "install_step_2" in installation_chunk_ids[:2]  # Top 2 results
        
        # Test troubleshooting query
        troubleshoot_query_embedding = [0.0, 0.1, 0.9, 0.0, 0.0]
        troubleshoot_results = await store.similarity_search(
            query_embedding=troubleshoot_query_embedding,
            top_k=2
        )
        
        # Should prioritize troubleshooting chunk
        top_troubleshoot_chunk = troubleshoot_results[0][0]
        assert top_troubleshoot_chunk.chunk_id == "troubleshoot_power"

