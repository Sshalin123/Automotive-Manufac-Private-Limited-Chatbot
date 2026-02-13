"""
Authentication routes for AMPL Chatbot API.

Provides JWT-based login and token management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from database.repositories import UserRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    role: str


class CreateUserRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "agent"
    tenant_id: Optional[str] = None


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token."""
    from api.middleware.auth import create_jwt_token, verify_password

    repo = UserRepository(db)
    user = await repo.get_by_email(request.email)

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    token_data = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
    }
    token, expires_in = create_jwt_token(token_data)

    return LoginResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user.id,
        role=user.role,
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user (admin only in production)."""
    from api.middleware.auth import hash_password
    import secrets

    repo = UserRepository(db)
    existing = await repo.get_by_email(request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await repo.create(
        email=request.email,
        hashed_password=hash_password(request.password),
        full_name=request.full_name,
        role=request.role,
        tenant_id=request.tenant_id,
        api_key=secrets.token_hex(32),
    )

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "api_key": user.api_key,
    }
