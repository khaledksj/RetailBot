"""
Authentication API endpoints for multi-tenant RAG system.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Tuple

from app.core.models import (
    UserRegistration, UserLogin, TokenResponse, UserInvite, 
    InviteAccept, User, Tenant, UserRole
)
from app.core.auth import create_access_token, create_refresh_token, create_invite_token, verify_invite_token
from app.core.auth_service import get_auth_service
from app.core.middleware import get_current_user_and_tenant, require_auth, tenant_context
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/auth/register", response_model=TokenResponse)
async def register(registration: UserRegistration):
    """Register new tenant with owner user."""
    auth_service = get_auth_service()
    await auth_service.initialize()
    
    try:
        # Create tenant and owner user
        tenant, user = await auth_service.create_tenant_and_owner(
            tenant_name=registration.tenant.name,
            tenant_slug=registration.tenant.slug,
            plan=registration.tenant.plan,
            owner_email=registration.user.email,
            owner_password=registration.user.password
        )
        
        # Create tokens
        access_token = create_access_token(user, tenant)
        refresh_token = create_refresh_token(user, tenant)
        
        logger.info("User registered successfully", extra={
            "tenant_id": str(tenant.id),
            "user_id": str(user.id),
            "tenant_slug": tenant.slug
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=1800,  # 30 minutes
            user=user,
            tenant=tenant
        )
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Login user and return tokens."""
    auth_service = get_auth_service()
    await auth_service.initialize()
    
    # Authenticate user
    result = await auth_service.authenticate_user(credentials.email, credentials.password)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    user, tenant = result
    
    # Create tokens
    access_token = create_access_token(user, tenant)
    refresh_token = create_refresh_token(user, tenant)
    
    logger.info("User logged in successfully", extra={
        "tenant_id": str(tenant.id),
        "user_id": str(user.id),
        "user_email": user.email
    })
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,  # 30 minutes
        user=user,
        tenant=tenant
    )

@router.post("/auth/invite")
@require_auth(require_roles=[UserRole.OWNER, UserRole.ADMIN])
async def invite_user(invite: UserInvite):
    """Invite a new user to the tenant (admin/owner only)."""
    auth_service = get_auth_service()
    
    try:
        # Create user invitation
        invited_user = await auth_service.create_user_invite(
            tenant_id=tenant_context.tenant_id,
            email=invite.email,
            role=invite.role,
            invited_by=tenant_context.user_id
        )
        
        # Create invitation token
        invite_token = create_invite_token(
            tenant_id=tenant_context.tenant_id,
            email=invite.email,
            role=invite.role,
            invited_by=tenant_context.user_id
        )
        
        logger.info("User invitation sent", extra={
            "invited_user_id": str(invited_user.id),
            "invited_email": invite.email,
            "invited_by": str(tenant_context.user_id),
            "tenant_id": str(tenant_context.tenant_id)
        })
        
        return {
            "success": True,
            "message": f"Invitation sent to {invite.email}",
            "invite_token": invite_token,
            "user_id": str(invited_user.id)
        }
        
    except Exception as e:
        logger.error(f"Invitation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to send invitation: {str(e)}"
        )

@router.post("/auth/accept-invite", response_model=TokenResponse)
async def accept_invite(acceptance: InviteAccept):
    """Accept invitation and set password."""
    # Verify invitation token
    invite_data = verify_invite_token(acceptance.token)
    if not invite_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invitation token"
        )
    
    auth_service = get_auth_service()
    await auth_service.initialize()
    
    try:
        # Find the invited user
        async with auth_service.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                \"\"\"
                SELECT id, tenant_id, email, password_hash, role, created_at, updated_at
                FROM users 
                WHERE email = $1 AND tenant_id = $2 AND password_hash IS NULL
                \"\"\",
                invite_data["email"], invite_data["tenant_id"]
            )
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invitation not found or already accepted"
                )
        
        # Accept invitation
        user = await auth_service.accept_invite(user_data["id"], acceptance.password)
        
        # Get tenant details
        async with auth_service.pool.acquire() as conn:
            tenant_data = await conn.fetchrow(
                \"\"\"
                SELECT id, name, slug, plan, created_at, updated_at
                FROM tenants WHERE id = $1
                \"\"\",
                user.tenant_id
            )
            
            tenant = Tenant(
                id=tenant_data["id"],
                name=tenant_data["name"],
                slug=tenant_data["slug"],
                plan=tenant_data["plan"],
                created_at=tenant_data["created_at"],
                updated_at=tenant_data["updated_at"]
            )
        
        # Create tokens
        access_token = create_access_token(user, tenant)
        refresh_token = create_refresh_token(user, tenant)
        
        logger.info("Invitation accepted successfully", extra={
            "user_id": str(user.id),
            "tenant_id": str(tenant.id),
            "user_email": user.email
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=1800,
            user=user,
            tenant=tenant
        )
        
    except Exception as e:
        logger.error(f"Accept invitation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to accept invitation: {str(e)}"
        )