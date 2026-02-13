"""
Authentication Middleware for AMPL Chatbot API.

Supports both API key and JWT bearer token authentication.
Includes role-based access control (Gap 14.2).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException, Security, Depends, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# ── Security schemes ───────────────────────────────────────────────
API_KEY_HEADER = "X-API-Key"
API_KEY_QUERY = "api_key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# ── Password hashing ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a password."""
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return ctx.hash(plain)
    except ImportError:
        # Fallback: simple hash (NOT for production)
        import hashlib
        return hashlib.sha256(plain.encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return ctx.verify(plain, hashed)
    except ImportError:
        import hashlib
        return hashlib.sha256(plain.encode()).hexdigest() == hashed


# ── JWT ────────────────────────────────────────────────────────────

def _get_jwt_settings() -> Tuple[str, str, int]:
    """Get JWT config from environment."""
    secret = os.environ.get("JWT_SECRET_KEY", "change-me-in-production")
    algorithm = os.environ.get("JWT_ALGORITHM", "HS256")
    expire_min = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
    return secret, algorithm, expire_min


def create_jwt_token(data: Dict[str, Any]) -> Tuple[str, int]:
    """
    Create a JWT token.

    Returns:
        Tuple of (token_string, expires_in_seconds)
    """
    try:
        from jose import jwt
    except ImportError:
        raise HTTPException(status_code=500, detail="JWT library not installed")

    secret, algorithm, expire_min = _get_jwt_settings()
    expires = datetime.utcnow() + timedelta(minutes=expire_min)
    payload = {**data, "exp": expires}
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token, expire_min * 60


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        from jose import jwt, JWTError
    except ImportError:
        raise HTTPException(status_code=500, detail="JWT library not installed")

    secret, algorithm, _ = _get_jwt_settings()
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


# ── Dependencies ──────────────────────────────────────────────────

def get_api_key():
    """Get the API key from environment."""
    return os.environ.get("AMPL_API_KEY", "")


async def api_key_auth(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> str:
    """Validate API key from header or query parameter."""
    api_key = header_key or query_key
    expected_key = get_api_key()

    # Skip auth if no key configured (development mode)
    if not expected_key:
        logger.warning("API key authentication disabled - no key configured")
        return ""

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if api_key != expected_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> Dict[str, Any]:
    """
    Get current user from JWT token or API key.

    Returns user payload dict with at least: sub, role, tenant_id
    """
    # Try JWT first
    if credentials and credentials.credentials:
        payload = decode_jwt_token(credentials.credentials)
        return payload

    # Fall back to API key
    api_key = header_key or query_key
    expected_key = get_api_key()

    if not expected_key:
        # Dev mode: return anonymous user
        return {"sub": "anonymous", "role": "admin", "tenant_id": None}

    if api_key == expected_key:
        return {"sub": "api_key_user", "role": "admin", "tenant_id": None}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


def require_role(*roles: str) -> Callable:
    """
    Factory that returns a dependency requiring specific roles.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role("admin"))])
    """
    async def _check_role(user: Dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(roles)}",
            )
        return user
    return _check_role


def require_auth(api_key: str = Depends(api_key_auth)):
    """Dependency that requires authentication (legacy)."""
    return api_key
