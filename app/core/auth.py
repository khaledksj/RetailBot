"""
Authentication utilities for the multi-tenant Shop Manual Chatbot RAG system.
"""

import os
from jose import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID
from passlib.context import CryptContext
from passlib.hash import bcrypt

from app.core.models import User, Tenant, TokenPayload, UserRole
from app.core.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(user: User, tenant: Tenant, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "user_id": str(user.id),
        "tenant_id": str(tenant.id),
        "role": user.role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user: User, tenant: Tenant) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "user_id": str(user.id),
        "tenant_id": str(tenant.id),
        "role": user.role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[TokenPayload]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        token_payload = TokenPayload(
            user_id=UUID(payload["user_id"]),
            tenant_id=UUID(payload["tenant_id"]),
            role=UserRole(payload["role"]),
            exp=payload["exp"],
            iat=payload["iat"]
        )
        
        return token_payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.warning(f"Malformed token payload: {e}")
        return None

def create_invite_token(tenant_id: UUID, email: str, role: UserRole, invited_by: UUID) -> str:
    """Create invitation token."""
    expire = datetime.utcnow() + timedelta(days=7)  # Invite expires in 7 days
    
    payload = {
        "tenant_id": str(tenant_id),
        "email": email,
        "role": role.value,
        "invited_by": str(invited_by),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "invite"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_invite_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode invitation token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "invite":
            return None
            
        return {
            "tenant_id": UUID(payload["tenant_id"]),
            "email": payload["email"],
            "role": UserRole(payload["role"]),
            "invited_by": UUID(payload["invited_by"])
        }
        
    except jwt.ExpiredSignatureError:
        logger.warning("Invitation token has expired")
        return None
    except jwt.JWTError as e:
        logger.warning(f"Invalid invitation token: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.warning(f"Malformed invitation token payload: {e}")
        return None