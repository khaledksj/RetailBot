"""
Chat API endpoints for the Shop Manual Chatbot RAG system.
"""

from fastapi import APIRouter, HTTPException, Depends
from app.core.logging import get_logger
from app.core.models import ChatRequest, ChatResponse, User
from app.api.auth_working import get_current_user
from app.core.rag import RAGPipeline

router = APIRouter()
logger = get_logger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Handle chat requests using RAG pipeline.
    
    - **message**: User's question
    - **session_id**: Optional session identifier
    - **temperature**: Response randomness (0.0-1.0)
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    logger.info(f"Chat request received", extra={
        "session_id": request.session_id,
        "message_length": len(request.message),
        "temperature": request.temperature,
        "user_id": str(current_user.id),
        "tenant_id": str(current_user.tenant_id)
    })
    
    try:
        rag_pipeline = RAGPipeline()
        response = await rag_pipeline.process_query(
            query=request.message,
            session_id=request.session_id,
            temperature=request.temperature,
            tenant_id=str(current_user.tenant_id)
        )
        
        logger.info(f"Chat response generated", extra={
            "session_id": request.session_id,
            "sources_count": len(response.sources),
            "response_length": len(response.answer)
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
