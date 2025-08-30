"""
Pydantic models for the Shop Manual Chatbot RAG system.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from enum import Enum

class UserRole(str, Enum):
    """User role enumeration."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"

class TenantPlan(str, Enum):
    """Tenant plan enumeration."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class Tenant(BaseModel):
    """Tenant model."""
    id: UUID
    name: str
    slug: str
    plan: TenantPlan = TenantPlan.FREE
    created_at: datetime
    updated_at: datetime

class User(BaseModel):
    """User model."""
    id: UUID
    tenant_id: UUID
    email: str
    password_hash: Optional[str] = None
    role: UserRole
    created_at: datetime
    updated_at: datetime

class Chunk(BaseModel):
    """Document chunk model."""
    chunk_id: str
    doc_id: str
    tenant_id: UUID
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
    tenant_id: Optional[UUID] = None
    created_by: Optional[UUID] = None
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

# Authentication Models

class UserCreate(BaseModel):
    """User creation model."""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.MEMBER

class UserLogin(BaseModel):
    """User login model."""
    email: str
    password: str

class TenantCreate(BaseModel):
    """Tenant creation model for registration."""
    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., pattern=r'^[a-z0-9-]+$', min_length=3, max_length=30)
    plan: TenantPlan = TenantPlan.FREE

class UserRegistration(BaseModel):
    """User registration with tenant creation."""
    tenant: TenantCreate
    user: UserCreate

class TokenPayload(BaseModel):
    """JWT token payload."""
    user_id: UUID
    tenant_id: UUID
    role: UserRole
    exp: int
    iat: int

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User
    tenant: Tenant

class UserInvite(BaseModel):
    """User invitation model."""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    role: UserRole = UserRole.MEMBER

class InviteAccept(BaseModel):
    """Invitation acceptance model."""
    token: str
    password: str = Field(..., min_length=8)

class TenantUsage(BaseModel):
    """Tenant usage metrics model."""
    tenant_id: UUID
    date: str
    requests_count: int
    tokens_used: int
    storage_bytes: int
