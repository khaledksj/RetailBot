"""
Authentication service for multi-tenant RAG system.
"""

import asyncio
import asyncpg
from typing import Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime

from app.core.models import User, Tenant, UserRole, TenantPlan
from app.core.auth import hash_password, verify_password
from app.core.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

class AuthService:
    """Authentication service for tenant and user management."""
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.db_connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info("AuthService database pool initialized")
    
    async def create_tenant_and_owner(
        self, 
        tenant_name: str, 
        tenant_slug: str, 
        plan: TenantPlan,
        owner_email: str, 
        owner_password: str
    ) -> Tuple[Tenant, User]:
        """Create new tenant with owner user."""
        if not self.pool:
            await self.initialize()
        
        hashed_password = hash_password(owner_password)
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Create tenant
                tenant_data = await conn.fetchrow(
                    \"\"\"
                    INSERT INTO tenants (name, slug, plan)
                    VALUES ($1, $2, $3)
                    RETURNING id, name, slug, plan, created_at, updated_at
                    \"\"\",
                    tenant_name, tenant_slug, plan.value
                )
                
                tenant = Tenant(
                    id=tenant_data["id"],
                    name=tenant_data["name"],
                    slug=tenant_data["slug"],
                    plan=TenantPlan(tenant_data["plan"]),
                    created_at=tenant_data["created_at"],
                    updated_at=tenant_data["updated_at"]
                )
                
                # Create owner user
                user_data = await conn.fetchrow(
                    \"\"\"
                    INSERT INTO users (tenant_id, email, password_hash, role)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, tenant_id, email, password_hash, role, created_at, updated_at
                    \"\"\",
                    tenant.id, owner_email, hashed_password, UserRole.OWNER.value
                )
                
                user = User(
                    id=user_data["id"],
                    tenant_id=user_data["tenant_id"],
                    email=user_data["email"],
                    password_hash=user_data["password_hash"],
                    role=UserRole(user_data["role"]),
                    created_at=user_data["created_at"],
                    updated_at=user_data["updated_at"]
                )
                
                logger.info("Created tenant and owner user", extra={
                    "tenant_id": str(tenant.id),
                    "tenant_slug": tenant.slug,
                    "user_id": str(user.id),
                    "user_email": user.email
                })
                
                return tenant, user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[Tuple[User, Tenant]]:
        """Authenticate user and return user with tenant info."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                \"\"\"
                SELECT u.id, u.tenant_id, u.email, u.password_hash, u.role, 
                       u.created_at, u.updated_at,
                       t.id as tenant_id, t.name, t.slug, t.plan, 
                       t.created_at as tenant_created_at, t.updated_at as tenant_updated_at
                FROM users u
                JOIN tenants t ON u.tenant_id = t.id
                WHERE u.email = $1
                \"\"\",
                email
            )
            
            if not user_data or not user_data["password_hash"]:
                return None
            
            if not verify_password(password, user_data["password_hash"]):
                return None
            
            user = User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                password_hash=user_data["password_hash"],
                role=UserRole(user_data["role"]),
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
            
            tenant = Tenant(
                id=user_data["tenant_id"],
                name=user_data["name"],
                slug=user_data["slug"],
                plan=TenantPlan(user_data["plan"]),
                created_at=user_data["tenant_created_at"],
                updated_at=user_data["tenant_updated_at"]
            )
            
            return user, tenant
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                \"\"\"
                SELECT id, tenant_id, email, password_hash, role, created_at, updated_at
                FROM users WHERE id = $1
                \"\"\",
                user_id
            )
            
            if not user_data:
                return None
            
            return User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                password_hash=user_data["password_hash"],
                role=UserRole(user_data["role"]),
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
    
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            tenant_data = await conn.fetchrow(
                \"\"\"
                SELECT id, name, slug, plan, created_at, updated_at
                FROM tenants WHERE slug = $1
                \"\"\",
                slug
            )
            
            if not tenant_data:
                return None
            
            return Tenant(
                id=tenant_data["id"],
                name=tenant_data["name"],
                slug=tenant_data["slug"],
                plan=TenantPlan(tenant_data["plan"]),
                created_at=tenant_data["created_at"],
                updated_at=tenant_data["updated_at"]
            )
    
    async def create_user_invite(
        self,
        tenant_id: UUID,
        email: str,
        role: UserRole,
        invited_by: UUID
    ) -> User:
        """Create a user invitation."""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                \"\"\"
                INSERT INTO users (tenant_id, email, role)
                VALUES ($1, $2, $3)
                RETURNING id, tenant_id, email, password_hash, role, created_at, updated_at
                \"\"\",
                tenant_id, email, role.value
            )
            
            user = User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                password_hash=user_data["password_hash"],
                role=UserRole(user_data["role"]),
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
            
            logger.info("User invitation created", extra={
                "user_id": str(user.id),
                "tenant_id": str(tenant_id),
                "invited_email": email,
                "invited_by": str(invited_by)
            })
            
            return user
    
    async def accept_invite(self, user_id: UUID, password: str) -> User:
        """Accept invitation by setting user password."""
        if not self.pool:
            await self.initialize()
        
        hashed_password = hash_password(password)
        
        async with self.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                \"\"\"
                UPDATE users 
                SET password_hash = $1, updated_at = NOW()
                WHERE id = $2 AND password_hash IS NULL
                RETURNING id, tenant_id, email, password_hash, role, created_at, updated_at
                \"\"\",
                hashed_password, user_id
            )
            
            if not user_data:
                raise ValueError("Invalid invitation or already accepted")
            
            user = User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                password_hash=user_data["password_hash"],
                role=UserRole(user_data["role"]),
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
            
            logger.info("Invitation accepted", extra={
                "user_id": str(user.id),
                "tenant_id": str(user.tenant_id)
            })
            
            return user

# Global auth service instance
_auth_service: Optional[AuthService] = None

def get_auth_service() -> AuthService:
    """Get global auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(settings.database_url)
    return _auth_service