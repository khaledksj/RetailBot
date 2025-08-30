"""
Working authentication endpoints for multi-tenant RAG system.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID
from jose import JWTError, jwt

from app.core.auth_working import get_auth_service, SECRET_KEY, ALGORITHM
from app.core.models import User
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
security = HTTPBearer()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_slug: Optional[str] = None

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_name: str
    full_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str
    role: str

class UserInfo(BaseModel):
    id: str
    email: str
    role: str
    tenant_id: str

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Get user from database
        auth_service = await get_auth_service()
        async with auth_service.pool.acquire() as conn:
            user_data = await conn.fetchrow(
                "SELECT id, tenant_id, email, role, created_at, updated_at FROM users WHERE id = $1",
                UUID(user_id)
            )
            
            if user_data is None:
                raise credentials_exception
            
            return User(
                id=user_data["id"],
                tenant_id=user_data["tenant_id"],
                email=user_data["email"],
                role=user_data["role"],
                created_at=user_data["created_at"],
                updated_at=user_data["updated_at"]
            )
            
    except JWTError:
        raise credentials_exception

@router.post("/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register new tenant and owner."""
    auth_service = await get_auth_service()
    
    try:
        # Check if email already exists
        async with auth_service.pool.acquire() as conn:
            existing_user = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1",
                request.email
            )
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        # Create tenant and owner
        tenant, user = await auth_service.create_tenant_and_owner(
            tenant_name=request.tenant_name,
            owner_email=request.email,
            owner_password=request.password,
            full_name=request.full_name
        )
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id), "tenant_id": str(user.tenant_id), "role": user.role}
        )
        
        logger.info(f"New tenant registered: {tenant.name} with owner {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            user_id=str(user.id),
            tenant_id=str(user.tenant_id),
            role=user.role
        )
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login user to their tenant."""
    auth_service = await get_auth_service()
    
    user = await auth_service.authenticate_user(
        email=request.email,
        password=request.password,
        tenant_slug=request.tenant_slug
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id), "tenant_id": str(user.tenant_id), "role": user.role}
    )
    
    logger.info(f"User logged in: {user.email} (tenant: {user.tenant_id})")
    
    return TokenResponse(
        access_token=access_token,
        user_id=str(user.id),
        tenant_id=str(user.tenant_id),
        role=user.role
    )

@router.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserInfo(
        id=str(current_user.id),
        email=current_user.email,
        role=current_user.role,
        tenant_id=str(current_user.tenant_id)
    )