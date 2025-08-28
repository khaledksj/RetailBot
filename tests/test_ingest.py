"""
Tests for PDF ingestion functionality and deduplication logic.
"""

import pytest
import asyncio
import hashlib
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.pdf import PDFProcessor
from app.core.chunker import DocumentChunker
from app.core.db import InMemoryVectorStore
from app.core.models import Chunk


class TestPDFProcessor:
    """Test PDF text extraction functionality."""
    
    @pytest.fixture
    def pdf_processor(self):
        return PDFProcessor()
    
    def create_mock_pdf_content(self, text_content: str) -> bytes:
        """Create mock PDF content for testing."""
        # This is a simplified mock - in real tests you'd use a proper PDF library
        # For now, we'll simulate PDF content as bytes
        return text_content.encode('utf-8')
    
    @pytest.mark.asyncio
    async def test_extract_text_success(self, pdf_processor):
        """Test successful text extraction from PDF."""
        mock_content = self.create_mock_pdf_content("Test manual content\nPage 1 information")
        
        with patch.object(pdf_processor, '_extract_with_pypdf') as mock_pypdf:
            mock_pypdf.return_value = ["Test manual content", "Page 1 information"]
            
            result = await pdf_processor.extract_text(mock_content)
            
            assert len(result) == 2
            assert result[0] == "Test manual content"
            assert result[1] == "Page 1 information"
    
    @pytest.mark.asyncio
    async def test_extract_text_fallback_to_pdfminer(self, pdf_processor):
        """Test fallback to pdfminer when pypdf fails."""
        mock_content = self.create_mock_pdf_content("Test content")
        
        with patch.object(pdf_processor, '_extract_with_pypdf') as mock_pypdf, \
             patch.object(pdf_processor, '_extract_with_pdfminer') as mock_pdfminer:
            
            mock_pypdf.side_effect = Exception("pypdf failed")
            mock_pdfminer.return_value = ["Test content from pdfminer"]
            
            result = await pdf_processor.extract_text(mock_content)
            
            assert len(result) == 1
            assert result[0] == "Test content from pdfminer"
            mock_pypdf.assert_called_once()
            mock_pdfminer.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text_all_methods_fail(self, pdf_processor):
        """Test when all extraction methods fail."""
        mock_content = self.create_mock_pdf_content("Invalid content")
        
        with patch.object(pdf_processor, '_extract_with_pypdf') as mock_pypdf, \
             patch.object(pdf_processor, '_extract_with_pdfminer') as mock_pdfminer:
            
            mock_pypdf.side_effect = Exception("pypdf failed")
            mock_pdfminer.side_effect = Exception("pdfminer failed")
            
            result = await pdf_processor.extract_text(mock_content)
            
            assert result == []
    
    def test_validate_pdf_valid(self, pdf_processor):
        """Test PDF validation with valid content."""
        with patch('app.core.pdf.PdfReader') as mock_reader:
            mock_reader.return_value.pages = [MagicMock(), MagicMock()]  # 2 pages
            
            result = pdf_processor.validate_pdf(b"mock pdf content")
            
            assert result is True
    
    def test_validate_pdf_invalid(self, pdf_processor):
        """Test PDF validation with invalid content."""
        with patch('app.core.pdf.PdfReader') as mock_reader:
            mock_reader.side_effect = Exception("Invalid PDF")
            
            result = pdf_processor.validate_pdf(b"invalid content")
            
            assert result is False


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    @pytest.fixture
    def chunker(self):
        return DocumentChunker()
    
    def test_normalize_text(self, chunker):
        """Test text normalization."""
        input_text = "  Multiple   spaces\n\n\n\nExcessive newlines  "
        expected = "Multiple spaces\n\nExcessive newlines"
        
        result = chunker._normalize_text(input_text)
        
        assert result == expected
    
    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "This is a test sentence."
        token_count = chunker._count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_chunk_text_simple(self, chunker):
        """Test chunking of simple text."""
        text = "This is a simple manual page with some content."
        filename = "test_manual.pdf"
        page_number = 1
        
        chunks = chunker.chunk_text(text, page_number, filename)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.filename == filename for chunk in chunks)
        assert all(chunk.page == page_number for chunk in chunks)
        assert all(chunk.content_tokens > 0 for chunk in chunks)
    
    def test_chunk_text_long_content(self, chunker):
        """Test chunking of long content that needs splitting."""
        # Create long text that exceeds max_chunk_tokens
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(200)])
        filename = "long_manual.pdf"
        page_number = 1
        
        chunks = chunker.chunk_text(long_text, page_number, filename)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all(chunk.content_tokens <= chunker.max_tokens for chunk in chunks)
    
    def test_chunk_text_empty(self, chunker):
        """Test chunking of empty text."""
        chunks = chunker.chunk_text("", 1, "empty.pdf")
        
        assert chunks == []


class TestInMemoryVectorStore:
    """Test in-memory vector store functionality."""
    
    @pytest.fixture
    async def vector_store(self):
        store = InMemoryVectorStore()
        await store.initialize()
        return store
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Chunk(
                chunk_id="chunk_1",
                doc_id="doc_1",
                filename="manual1.pdf",
                page=1,
                chunk_idx=0,
                content="This is the first chunk about product installation.",
                content_tokens=50,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            Chunk(
                chunk_id="chunk_2",
                doc_id="doc_1",
                filename="manual1.pdf",
                page=2,
                chunk_idx=1,
                content="This is the second chunk about troubleshooting.",
                content_tokens=45,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_store_document(self, vector_store, sample_chunks):
        """Test storing a document with chunks."""
        filename = "test_manual.pdf"
        content_hash = "abc123"
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
        
        doc_id = await vector_store.store_document(
            filename=filename,
            content_hash=content_hash,
            chunks=sample_chunks,
            embeddings=embeddings
        )
        
        assert doc_id is not None
        assert len(vector_store.chunks) == 2
        assert len(vector_store.embeddings) == 2
        assert content_hash in vector_store.content_hashes
    
    @pytest.mark.asyncio
    async def test_document_exists(self, vector_store, sample_chunks):
        """Test document existence check."""
        content_hash = "test_hash_123"
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
        
        # Document should not exist initially
        exists_before = await vector_store.document_exists(content_hash)
        assert not exists_before
        
        # Store document
        await vector_store.store_document(
            filename="test.pdf",
            content_hash=content_hash,
            chunks=sample_chunks,
            embeddings=embeddings
        )
        
        # Document should exist now
        exists_after = await vector_store.document_exists(content_hash)
        assert exists_after
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store, sample_chunks):
        """Test similarity search functionality."""
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
        
        # Store documents first
        await vector_store.store_document(
            filename="test.pdf",
            content_hash="hash123",
            chunks=sample_chunks,
            embeddings=embeddings
        )
        
        # Search with similar query embedding
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = await vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=2
        )
        
        assert len(results) == 2
        assert all(len(result) == 2 for result in results)  # (chunk, score) tuples
        assert all(isinstance(result[0], Chunk) for result in results)
        assert all(isinstance(result[1], float) for result in results)
        
        # Results should be sorted by similarity score (descending)
        scores = [result[1] for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_get_counts(self, vector_store, sample_chunks):
        """Test document and chunk count methods."""
        # Initially empty
        assert await vector_store.get_document_count() == 0
        assert await vector_store.get_chunk_count() == 0
        
        # Store document
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]
        await vector_store.store_document(
            filename="test.pdf",
            content_hash="hash123",
            chunks=sample_chunks,
            embeddings=embeddings
        )
        
        # Check counts
        assert await vector_store.get_document_count() == 1
        assert await vector_store.get_chunk_count() == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, vector_store):
        """Test health check functionality."""
        result = await vector_store.health_check()
        assert result is True


class TestIngestionDeduplication:
    """Test end-to-end ingestion with deduplication."""
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test that duplicate documents are detected by content hash."""
        vector_store = InMemoryVectorStore()
        await vector_store.initialize()
        
        # Same content should produce same hash
        content1 = b"Test PDF content for deduplication"
        content2 = b"Test PDF content for deduplication"  # Identical
        content3 = b"Different PDF content"
        
        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        hash3 = hashlib.sha256(content3).hexdigest()
        
        assert hash1 == hash2
        assert hash1 != hash3
        
        # Store first document
        sample_chunk = Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            filename="manual1.pdf",
            page=1,
            chunk_idx=0,
            content="Test content",
            content_tokens=20,
            embedding=[0.1, 0.2, 0.3]
        )
        
        await vector_store.store_document(
            filename="manual1.pdf",
            content_hash=hash1,
            chunks=[sample_chunk],
            embeddings=[[0.1, 0.2, 0.3]]
        )
        
        # Check duplicate detection
        assert await vector_store.document_exists(hash1)  # Should exist
        assert await vector_store.document_exists(hash2)  # Same hash, should exist
        assert not await vector_store.document_exists(hash3)  # Different, should not exist
    
    @pytest.mark.asyncio
    async def test_force_reingest(self):
        """Test force re-ingestion functionality."""
        vector_store = InMemoryVectorStore()
        await vector_store.initialize()
        
        content_hash = "test_hash_for_force"
        filename = "test_manual.pdf"
        
        # Create initial chunk
        initial_chunk = Chunk(
            chunk_id="chunk_initial",
            doc_id="doc_initial",
            filename=filename,
            page=1,
            chunk_idx=0,
            content="Initial content",
            content_tokens=20,
            embedding=[0.1, 0.2, 0.3]
        )
        
        # Store initial document
        doc_id_1 = await vector_store.store_document(
            filename=filename,
            content_hash=content_hash,
            chunks=[initial_chunk],
            embeddings=[[0.1, 0.2, 0.3]]
        )
        
        # Verify initial state
        assert await vector_store.get_document_count() == 1
        assert await vector_store.get_chunk_count() == 1
        
        # Force re-ingest (in real implementation, this would involve
        # checking force flag and allowing re-ingestion)
        updated_chunk = Chunk(
            chunk_id="chunk_updated",
            doc_id="doc_updated",
            filename=filename,
            page=1,
            chunk_idx=0,
            content="Updated content",
            content_tokens=25,
            embedding=[0.2, 0.3, 0.4]
        )
        
        # In real implementation, force=True would allow this
        # For now, we'll simulate by storing with different content
        doc_id_2 = await vector_store.store_document(
            filename=filename,
            content_hash=content_hash + "_updated",  # Different hash
            chunks=[updated_chunk],
            embeddings=[[0.2, 0.3, 0.4]]
        )
        
        # Should have more documents now
        assert await vector_store.get_document_count() == 2
        assert await vector_store.get_chunk_count() == 2

