"""
Main FastAPI application entry point for the Shop Manual Chatbot RAG system.
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json

from app.core.settings import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.db import get_vector_store
from app.api import ingest, chat, health, debug
from app.api.vector_viewer import router as vector_viewer_router

# Initialize settings and logging
settings = get_settings()
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting Shop Manual Chatbot RAG system")
    
    # Initialize vector store
    vector_store = get_vector_store()
    await vector_store.initialize()
    
    yield
    
    logger.info("Shutting down Shop Manual Chatbot RAG system")

# Create FastAPI app
app = FastAPI(
    title="Shop Manual Chatbot RAG",
    description="Production-grade RAG chatbot for retail shop PDF manuals",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ingest.router, prefix="/api", tags=["ingest"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(debug.router, prefix="/api/debug", tags=["debug"])
app.include_router(vector_viewer_router, prefix="/api", tags=["vector-database"])

# WebSocket endpoint for streaming chat
@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            query = message_data.get("message", "")
            session_id = message_data.get("session_id", "default")
            temperature = message_data.get("temperature", 0.2)
            
            if not query.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Empty query received"
                }))
                continue
            
            logger.info(f"WebSocket chat query received", extra={
                "session_id": session_id,
                "query_length": len(query),
                "temperature": temperature
            })
            
            # Import here to avoid circular imports
            from app.core.rag import RAGPipeline
            
            rag_pipeline = RAGPipeline()
            
            # Stream response
            async for chunk in rag_pipeline.stream_chat_response(
                query=query,
                session_id=session_id,
                temperature=temperature
            ):
                await websocket.send_text(json.dumps(chunk))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"An error occurred: {str(e)}"
        }))

# Mount static files
app.mount("/static", StaticFiles(directory="app/ui/static"), name="static")

# Serve the main UI
@app.get("/")
async def serve_ui():
    """Serve the main UI page."""
    return FileResponse("app/ui/index.html")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
