"""
Pydantic models for the Shop Manual Chatbot RAG system.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class Chunk(BaseModel):
    """Document chunk model."""
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_idx: int
    content: str
    content_tokens: int
    embedding: List[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Source(BaseModel):
    """Source citation model."""
    filename: str
    page: int
    snippet: str

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: List[Source]
    session_id: str

class DocumentInfo(BaseModel):
    """Document processing information."""
    filename: str
    status: str  # success, failed, skipped
    pages_processed: int = 0
    chunks_created: int = 0
    doc_id: Optional[str] = None
    error: Optional[str] = None

class IngestResponse(BaseModel):
    """Ingestion response model."""
    success: bool
    message: str
    documents: List[DocumentInfo]

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str  # healthy, unhealthy
    vector_store_connected: bool = False
    documents_count: int = 0
    chunks_count: int = 0
    error: Optional[str] = None

class ChunkPreview(BaseModel):
    """Chunk preview for debug endpoints."""
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    content_preview: str
    score: float
    content_tokens: int

class DebugSearchResponse(BaseModel):
    """Debug search response model."""
    query: str
    results_count: int
    chunks: List[ChunkPreview]
    error: Optional[str] = None
