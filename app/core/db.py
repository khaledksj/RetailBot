"""
Vector database interface and implementations for the Shop Manual Chatbot RAG system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime
from uuid import uuid4, UUID
import json
import asyncpg
import os

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
        embeddings: List[List[float]],
        tenant_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None
    ) -> str:
        """Store document chunks with embeddings."""
        pass
    
    @abstractmethod
    async def document_exists(self, content_hash: str, tenant_id: Optional[UUID] = None) -> bool:
        """Check if document already exists."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        tenant_id: Optional[UUID] = None
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


class SupabaseVectorStore(VectorStore):
    """Supabase/PostgreSQL vector store with pgvector extension."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self) -> None:
        """Initialize the Supabase connection pool."""
        logger.info("Initializing Supabase vector store")
        
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60,
                statement_cache_size=0  # Disable statement caching for Supabase pgbouncer
            )
            logger.info("Supabase vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase vector store: {str(e)}")
            raise
            
    async def store_document(
        self,
        filename: str,
        content_hash: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> str:
        """Store document chunks with embeddings in Supabase."""
        if not self.pool:
            await self.initialize()
            
        logger.info(f"Storing document in Supabase", extra={
            "document_name": filename,
            "chunks_count": len(chunks),
            "content_hash": content_hash[:16] + "..."
        })
        
        async with self.pool.acquire() as conn:  # type: ignore
            async with conn.transaction():
                # Set tenant context if provided
                if tenant_id:
                    await conn.execute("SET LOCAL app.tenant_id = $1", str(tenant_id))
                
                # Insert document
                doc_id = await conn.fetchval(
                    "INSERT INTO documents (filename, content_hash, chunk_count, tenant_id, created_by) VALUES ($1, $2, $3, $4, $5) RETURNING doc_id",
                    filename, content_hash, len(chunks), tenant_id, created_by
                )
                
                # Insert chunks with proper vector formatting
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.doc_id = str(doc_id)
                    chunk.embedding = embedding
                    
                    # Convert embedding list to PostgreSQL vector format
                    vector_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    await conn.execute(
                        """INSERT INTO chunks 
                           (chunk_id, doc_id, filename, page, chunk_idx, content, content_tokens, embedding, created_at) 
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)""",
                        chunk.chunk_id,
                        doc_id,
                        chunk.filename,
                        chunk.page,
                        chunk.chunk_idx,
                        chunk.content,
                        chunk.content_tokens,
                        vector_str,
                        chunk.created_at
                    )
                
                logger.info(f"Document stored successfully in Supabase", extra={
                    "doc_id": str(doc_id),
                    "chunks_stored": len(chunks)
                })
                
                return str(doc_id)
    
    async def document_exists(self, content_hash: str) -> bool:
        """Check if document already exists by content hash."""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:  # type: ignore
            if tenant_id:
                await conn.execute("SET LOCAL app.tenant_id = $1", str(tenant_id))
                result = await conn.fetchval(
                    "SELECT document_exists_by_hash_tenant($1, $2)",
                    content_hash, tenant_id
                )
            else:
                result = await conn.fetchval(
                    "SELECT document_exists_by_hash($1)",
                    content_hash
                )
            return result
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using pgvector cosine similarity."""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:  # type: ignore
            # Convert query embedding to PostgreSQL vector format
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use tenant-aware search if tenant_id provided
            if tenant_id:
                await conn.execute("SET LOCAL app.tenant_id = $1", str(tenant_id))
                rows = await conn.fetch(
                    "SELECT * FROM search_similar_chunks_tenant($1::vector, $2, $3, $4)",
                    query_vector_str, tenant_id, 0.0, top_k
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM search_similar_chunks($1::vector, $2, $3)",
                    query_vector_str, 0.0, top_k
                )
            
            results = []
            for row in rows:
                chunk = Chunk(
                    chunk_id=row['chunk_id'],
                    doc_id=str(row['doc_id']),
                    filename=row['filename'],
                    page=row['page'],
                    chunk_idx=row['chunk_idx'],
                    content=row['content'],
                    content_tokens=row['content_tokens'],
                    embedding=query_embedding,  # We don't need to fetch the full embedding
                    created_at=row['created_at']
                )
                
                similarity_score = float(row['similarity_score'])
                results.append((chunk, similarity_score))
            
            logger.info(f"Similarity search completed", extra={
                "query_embedding_dim": len(query_embedding),
                "results_returned": len(results),
                "top_score": results[0][1] if results else 0
            })
            
            return results
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:  # type: ignore
            result = await conn.fetchval("SELECT COUNT(*) FROM documents")
            return int(result)
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        if not self.pool:
            await self.initialize()
            
        async with self.pool.acquire() as conn:  # type: ignore
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            return int(result)
    
    async def health_check(self) -> bool:
        """Check if Supabase connection is healthy."""
        try:
            if not self.pool:
                await self.initialize()
                
            async with self.pool.acquire() as conn:  # type: ignore
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            return False


# Vector store singleton
_vector_store: Optional[VectorStore] = None

def get_vector_store() -> VectorStore:
    """Get vector store singleton."""
    global _vector_store
    
    if _vector_store is None:
        backend = settings.vector_backend.lower()
        
        if backend == "memory":
            _vector_store = InMemoryVectorStore()
        elif backend == "supabase" or backend == "pgvector":
            # Supabase/PostgreSQL + pgvector backend
            supabase_url = settings.supabase_url or os.getenv("DATABASE_URL")
            if supabase_url:
                _vector_store = SupabaseVectorStore(supabase_url)
                logger.info("Using Supabase vector store")
            else:
                logger.warning("Supabase URL not configured, using memory")
                _vector_store = InMemoryVectorStore()
        elif backend == "chroma":
            # TODO: Implement Chroma backend
            logger.warning("chroma backend not implemented, using memory")
            _vector_store = InMemoryVectorStore()
        else:
            logger.warning(f"Unknown vector backend '{backend}', using memory")
            _vector_store = InMemoryVectorStore()
    
    return _vector_store
