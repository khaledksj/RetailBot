"""
Working authentication service for multi-tenant RAG system.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4
import hashlib
import hmac
import secrets
import json

import asyncpg
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.models import User, Tenant
from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = getattr(settings, 'secret_key', 'dev-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    """Authentication service for multi-tenant system."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        if not self.pool:
            database_url = settings.database_url
            self.pool = await asyncpg.create_pool(
                database_url,
                statement_cache_size=0,  # Disable prepared statements for Supabase compatibility
                command_timeout=30
            )
            logger.info("Auth service initialized with database pool")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def authenticate_user(self, email: str, password: str, tenant_slug: Optional[str] = None) -> Optional[User]:
        """Authenticate user by email and password."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            if tenant_slug:
                # Get tenant by slug first
                tenant_data = await conn.fetchrow(
                    "SELECT id FROM tenants WHERE slug = $1",
                    tenant_slug
                )
                if not tenant_data:
                    return None
                
                # Get user within specific tenant
                user_data = await conn.fetchrow(
                    """
                    SELECT id, tenant_id, email, password_hash, role, created_at, updated_at
                    FROM users 
                    WHERE email = $1 AND tenant_id = $2 AND password_hash IS NOT NULL
                    """,
                    email, tenant_data["id"]
                )
            else:
                # Get user across all tenants (for simple auth)
                user_data = await conn.fetchrow(
                    """
                    SELECT id, tenant_id, email, password_hash, role, created_at, updated_at
                    FROM users 
                    WHERE email = $1 AND password_hash IS NOT NULL
                    """,
                    email
                )
            
            if not user_data:
                return None
            
            # Verify password
            if not self.verify_password(password, user_data["password_hash"]):
                return None
            
            return User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                role=user_data["role"],
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
    
    async def create_tenant_and_owner(self, tenant_name: str, owner_email: str, owner_password: str, full_name: Optional[str] = None) -> tuple[Tenant, User]:
        """Create new tenant with owner user."""
        if not self.pool:
            await self.initialize()
        
        # Generate tenant slug from name
        tenant_slug = tenant_name.lower().replace(" ", "-").replace("_", "-")
        tenant_id = uuid4()
        user_id = uuid4()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Create tenant
                await conn.execute(
                    """
                    INSERT INTO tenants (id, name, slug, plan, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    tenant_id, tenant_name, tenant_slug, "free", datetime.utcnow(), datetime.utcnow()
                )
                
                # Create owner user
                password_hash = self.hash_password(owner_password)
                await conn.execute(
                    """
                    INSERT INTO users (id, tenant_id, email, password_hash, role, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    user_id, tenant_id, owner_email, password_hash, "owner", datetime.utcnow(), datetime.utcnow()
                )
                
                # Get created records
                tenant_data = await conn.fetchrow("SELECT * FROM tenants WHERE id = $1", tenant_id)
                user_data = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
                
                tenant = Tenant(
                    id=tenant_data["id"],
                    name=tenant_data["name"],
                    slug=tenant_data["slug"],
                    plan=tenant_data["plan"],
                    created_at=tenant_data["created_at"],
                    updated_at=tenant_data["updated_at"]
                )
                
                user = User(
                    id=user_data["id"],
                    tenant_id=user_data["tenant_id"],
                    email=user_data["email"],
                    role=user_data["role"],
                    created_at=user_data["created_at"],
                    updated_at=user_data["updated_at"]
                )
                
                return tenant, user

# Global auth service instance
_auth_service = None

async def get_auth_service() -> AuthService:
    """Get global auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
        await _auth_service.initialize()
    return _auth_service