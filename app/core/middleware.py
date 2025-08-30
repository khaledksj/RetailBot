"""
Multi-tenant middleware for request scoping and authentication.
"""

import asyncpg
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Tuple
from uuid import UUID

from app.core.auth import verify_token
from app.core.auth_service import get_auth_service
from app.core.models import User, Tenant, TokenPayload
from app.core.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
security = HTTPBearer(auto_error=False)

class TenantContext:
    """Context holder for current tenant and user."""
    def __init__(self):
        self.user: Optional[User] = None
        self.tenant: Optional[Tenant] = None
        self.tenant_id: Optional[UUID] = None
        self.user_id: Optional[UUID] = None

# Request-scoped context
tenant_context = TenantContext()

async def get_current_user_and_tenant(request: Request) -> Tuple[Optional[User], Optional[Tenant]]:
    """Extract user and tenant from JWT token."""
    # Try to get token from Authorization header
    credentials: Optional[HTTPAuthorizationCredentials] = await security(request)
    
    if not credentials:
        return None, None
    
    # Verify token
    token_payload = verify_token(credentials.credentials)
    if not token_payload:
        return None, None
    
    # Get auth service
    auth_service = get_auth_service()
    await auth_service.initialize()
    
    # Get user details
    user = await auth_service.get_user_by_id(token_payload.user_id)
    if not user:
        return None, None
    
    # Get tenant details
    async with auth_service.pool.acquire() as conn:
        tenant_data = await conn.fetchrow(
            \"\"\"
            SELECT id, name, slug, plan, created_at, updated_at
            FROM tenants WHERE id = $1
            \"\"\",
            token_payload.tenant_id
        )
        
        if not tenant_data:
            return None, None
        
        tenant = Tenant(
            id=tenant_data["id"],
            name=tenant_data["name"],
            slug=tenant_data["slug"],
            plan=tenant_data["plan"],
            created_at=tenant_data["created_at"],
            updated_at=tenant_data["updated_at"]
        )
    
    return user, tenant

async def set_tenant_context(conn: asyncpg.Connection, tenant_id: UUID) -> None:
    """Set tenant context for RLS in database connection."""
    await conn.execute("SET LOCAL app.tenant_id = $1", str(tenant_id))

def require_auth(require_roles: Optional[list] = None):
    """Decorator to require authentication and optionally specific roles."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs or args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user, tenant = await get_current_user_and_tenant(request)
            
            if not user or not tenant:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing authentication token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Check role requirements
            if require_roles and user.role not in require_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {require_roles}"
                )
            
            # Set context for use in route handlers
            tenant_context.user = user
            tenant_context.tenant = tenant
            tenant_context.tenant_id = tenant.id
            tenant_context.user_id = user.id
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

async def resolve_tenant_from_subdomain(request: Request) -> Optional[Tenant]:
    """Resolve tenant from subdomain (e.g., acme.myapp.com -> acme)."""
    host = request.headers.get("host", "")
    
    # Extract subdomain
    parts = host.split(".")
    if len(parts) < 3:  # No subdomain
        return None
    
    subdomain = parts[0]
    
    # Skip common subdomains
    if subdomain in ["www", "api", "admin"]:
        return None
    
    # Get tenant by slug
    auth_service = get_auth_service()
    await auth_service.initialize()
    
    return await auth_service.get_tenant_by_slug(subdomain)

class TenantScopingMiddleware:
    """Middleware to set tenant context for all database operations."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Extract tenant and user context
        user, tenant = await get_current_user_and_tenant(request)
        
        # If no auth token, try subdomain resolution
        if not tenant:
            tenant = await resolve_tenant_from_subdomain(request)
        
        # Set tenant context for the request
        if tenant:
            tenant_context.tenant = tenant
            tenant_context.tenant_id = tenant.id
            if user:
                tenant_context.user = user
                tenant_context.user_id = user.id
        
        await self.app(scope, receive, send)