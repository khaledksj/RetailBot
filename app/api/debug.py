"""
Debug API endpoints for the Shop Manual Chatbot RAG system.
"""

from fastapi import APIRouter, Query
from typing import List
from app.core.logging import get_logger
from app.core.models import DebugSearchResponse, ChunkPreview
from app.core.llm import EmbeddingService
from app.core.db import get_vector_store
from app.core.arabic_extractor import extraction_health
from fastapi import HTTPException

router = APIRouter()
logger = get_logger(__name__)

@router.get("/search", response_model=DebugSearchResponse)
async def debug_search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(8, description="Number of results to return")
):
    """
    Debug endpoint to show retrieved chunks for a query.
    
    - **query**: Search query
    - **top_k**: Number of chunks to retrieve
    """
    logger.info(f"Debug search request", extra={
        "query": query,
        "top_k": top_k
    })
    
    try:
        embedding_service = EmbeddingService()
        vector_store = get_vector_store()
        
        # Generate query embedding
        query_embedding = await embedding_service.create_embeddings([query])
        
        # Search for similar chunks
        results = await vector_store.similarity_search(
            query_embedding=query_embedding[0],
            top_k=top_k
        )
        
        # Format results for debug output
        chunk_previews = []
        for chunk, score in results:
            preview = ChunkPreview(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page=chunk.page,
                content_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                score=score,
                content_tokens=chunk.content_tokens
            )
            chunk_previews.append(preview)
        
        return DebugSearchResponse(
            query=query,
            results_count=len(chunk_previews),
            chunks=chunk_previews
        )
        
    except Exception as e:
        logger.error(f"Debug search error: {str(e)}")
        return DebugSearchResponse(
            query=query,
            results_count=0,
            chunks=[],
            error=str(e)
        )

@router.get("/inspect")
async def inspect_document(doc_id: str = Query(..., description="Document ID to inspect")):
    """
    Inspect stored document content and extraction health.
    
    Args:
        doc_id: Document ID to inspect
        
    Returns:
        Document content preview and health metrics
    """
    try:
        vector_store = get_vector_store()
        
        # Get all chunks for this document - using similarity search with dummy query
        dummy_embedding = [0.0] * 1536  # Assuming 1536-dimensional embeddings
        all_results = await vector_store.similarity_search(
            query_embedding=dummy_embedding,
            top_k=1000  # Get many results
        )
        
        # Filter chunks by document ID
        doc_chunks = [(chunk, score) for chunk, score in all_results if chunk.doc_id == doc_id]
        
        if not doc_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Combine all chunks to get full content
        full_content = "\n".join(chunk.content for chunk, _ in doc_chunks)
        
        # Calculate health metrics
        health = extraction_health(full_content)
        
        # Get document metadata from first chunk
        first_chunk = doc_chunks[0][0]
        filename = first_chunk.filename
        
        return {
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(doc_chunks),
            "content_preview": full_content[:400] + "..." if len(full_content) > 400 else full_content,
            "health_metrics": health,
            "full_length": len(full_content)
        }
        
    except Exception as e:
        logger.error(f"Document inspection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
