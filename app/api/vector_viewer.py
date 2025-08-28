"""
Vector Database Viewer API endpoint.
"""

from fastapi import APIRouter
from typing import Dict, Any, List
from app.core.db import get_vector_store
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/database/overview")
async def get_database_overview():
    """Get overview of the vector database contents."""
    try:
        vector_store = get_vector_store()
        
        # Get basic stats
        doc_count = await vector_store.get_document_count()
        chunk_count = await vector_store.get_chunk_count()
        
        # Get document list if it's InMemoryVectorStore
        documents_info = []
        if hasattr(vector_store, 'documents'):
            for doc_id, doc_info in vector_store.documents.items():
                documents_info.append({
                    "doc_id": doc_id,
                    "filename": doc_info.get("filename", "Unknown"),
                    "chunk_count": doc_info.get("chunk_count", 0),
                    "created_at": str(doc_info.get("created_at", "Unknown"))
                })
        
        # Get sample chunks
        sample_chunks = []
        if hasattr(vector_store, 'chunks'):
            for i, (chunk_id, chunk) in enumerate(vector_store.chunks.items()):
                if i >= 5:  # Limit to first 5 chunks
                    break
                sample_chunks.append({
                    "chunk_id": chunk_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                    "token_count": chunk.content_tokens
                })
        
        return {
            "status": "success",
            "database_type": "In-Memory Vector Store",
            "statistics": {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "embedding_dimensions": 1536 if chunk_count > 0 else 0
            },
            "documents": documents_info,
            "sample_chunks": sample_chunks
        }
        
    except Exception as e:
        logger.error(f"Error getting database overview: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/database/chunks")
async def get_all_chunks():
    """Get all chunks in the database."""
    try:
        vector_store = get_vector_store()
        
        all_chunks = []
        if hasattr(vector_store, 'chunks'):
            for chunk_id, chunk in vector_store.chunks.items():
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_idx,
                    "content": chunk.content,
                    "token_count": chunk.content_tokens,
                    "has_embedding": len(chunk.embedding) > 0,
                    "created_at": str(chunk.created_at)
                })
        
        return {
            "status": "success",
            "chunk_count": len(all_chunks),
            "chunks": all_chunks
        }
        
    except Exception as e:
        logger.error(f"Error getting all chunks: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/database/embeddings")
async def get_embeddings_info():
    """Get information about stored embeddings."""
    try:
        vector_store = get_vector_store()
        
        embeddings_info = []
        if hasattr(vector_store, 'embeddings'):
            for chunk_id, embedding in vector_store.embeddings.items():
                embeddings_info.append({
                    "chunk_id": chunk_id,
                    "embedding_size": len(embedding) if hasattr(embedding, '__len__') else 0,
                    "embedding_preview": embedding[:5].tolist() if hasattr(embedding, 'tolist') else []
                })
        
        return {
            "status": "success",
            "total_embeddings": len(embeddings_info),
            "embedding_dimension": 1536 if embeddings_info else 0,
            "embeddings": embeddings_info[:10]  # Show first 10
        }
        
    except Exception as e:
        logger.error(f"Error getting embeddings info: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }