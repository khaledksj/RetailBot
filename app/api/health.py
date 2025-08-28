"""
Health check API endpoints for the Shop Manual Chatbot RAG system.
"""

from fastapi import APIRouter
from app.core.models import HealthResponse
from app.core.db import get_vector_store
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vector_store = get_vector_store()
        
        # Check vector store connectivity
        store_healthy = await vector_store.health_check()
        
        # Get basic stats
        doc_count = await vector_store.get_document_count()
        chunk_count = await vector_store.get_chunk_count()
        
        return HealthResponse(
            status="healthy" if store_healthy else "unhealthy",
            vector_store_connected=store_healthy,
            documents_count=doc_count,
            chunks_count=chunk_count
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            vector_store_connected=False,
            error=str(e)
        )

@router.get("/healthz")
async def simple_health():
    """Simple health check for load balancers."""
    return {"status": "ok"}
