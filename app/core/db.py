"""
Vector database interface and implementations for the Shop Manual Chatbot RAG system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime
from uuid import uuid4
import json

from app.core.models import Chunk
from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass
    
    @abstractmethod
    async def store_document(
        self,
        filename: str,
        content_hash: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> str:
        """Store document chunks with embeddings."""
        pass
    
    @abstractmethod
    async def document_exists(self, content_hash: str) -> bool:
        """Check if document already exists."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        pass
    
    @abstractmethod
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vector store is healthy."""
        pass

class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation for development and testing."""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> doc_info
        self.chunks: Dict[str, Chunk] = {}  # chunk_id -> chunk
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
        self.content_hashes: Dict[str, str] = {}  # content_hash -> doc_id
        
    async def initialize(self) -> None:
        """Initialize the in-memory store."""
        logger.info("Initializing in-memory vector store")
        # Nothing to initialize for in-memory store
        
    async def store_document(
        self,
        filename: str,
        content_hash: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> str:
        """Store document chunks with embeddings in memory."""
        doc_id = str(uuid4())
        
        logger.info(f"Storing document in memory", extra={
            "doc_id": doc_id,
            "document_name": filename,
            "chunks_count": len(chunks),
            "content_hash": content_hash[:16] + "..."
        })
        
        # Store document metadata
        self.documents[doc_id] = {
            "filename": filename,
            "content_hash": content_hash,
            "created_at": datetime.utcnow(),
            "chunk_count": len(chunks)
        }
        
        # Store content hash mapping
        self.content_hashes[content_hash] = doc_id
        
        # Store chunks and embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.doc_id = doc_id
            chunk.embedding = embedding
            self.chunks[chunk.chunk_id] = chunk
            self.embeddings[chunk.chunk_id] = np.array(embedding, dtype=np.float32)
        
        logger.info(f"Document stored successfully", extra={
            "doc_id": doc_id,
            "total_chunks": len(self.chunks),
            "total_documents": len(self.documents)
        })
        
        return doc_id
    
    async def document_exists(self, content_hash: str) -> bool:
        """Check if document already exists by content hash."""
        return content_hash in self.content_hashes
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using cosine similarity."""
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        similarities = []
        
        for chunk_id, embedding in self.embeddings.items():
            # Calculate cosine similarity
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm == 0:
                continue
                
            similarity = np.dot(query_vec, embedding) / (query_norm * embedding_norm)
            similarities.append((chunk_id, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for chunk_id, score in similarities[:top_k]:
            chunk = self.chunks[chunk_id]
            results.append((chunk, score))
        
        logger.info(f"Similarity search completed", extra={
            "query_embedding_dim": len(query_embedding),
            "total_chunks_searched": len(self.embeddings),
            "results_returned": len(results),
            "top_score": results[0][1] if results else 0
        })
        
        return results
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)
    
    async def health_check(self) -> bool:
        """Check if in-memory store is healthy."""
        return True

# Vector store singleton
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get vector store singleton."""
    global _vector_store
    
    if _vector_store is None:
        backend = settings.vector_backend.lower()
        
        if backend == "memory":
            _vector_store = InMemoryVectorStore()
        elif backend == "pgvector":
            # TODO: Implement PostgreSQL + pgvector backend
            logger.warning("pgvector backend not implemented, using memory")
            _vector_store = InMemoryVectorStore()
        elif backend == "chroma":
            # TODO: Implement Chroma backend
            logger.warning("chroma backend not implemented, using memory")
            _vector_store = InMemoryVectorStore()
        else:
            logger.warning(f"Unknown vector backend '{backend}', using memory")
            _vector_store = InMemoryVectorStore()
    
    return _vector_store
