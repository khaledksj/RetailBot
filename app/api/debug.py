"""
Debug API endpoints for the Shop Manual Chatbot RAG system.
"""

from fastapi import APIRouter, Query
from typing import List
from app.core.logging import get_logger
from app.core.models import DebugSearchResponse, ChunkPreview
from app.core.llm import EmbeddingService
from app.core.db import get_vector_store

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
