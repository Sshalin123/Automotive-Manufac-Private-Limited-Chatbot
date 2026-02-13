"""
Authentication Middleware for AMPL Chatbot API.
"""

import os
import logging
from typing import Optional

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader, APIKeyQuery

logger = logging.getLogger(__name__)

# API key header/query parameter names
API_KEY_HEADER = "X-API-Key"
API_KEY_QUERY = "api_key"

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)


def get_api_key():
    """Get the API key from environment."""
    return os.environ.get("AMPL_API_KEY", "")


async def api_key_auth(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
) -> str:
    """
    Validate API key from header or query parameter.

    Args:
        header_key: API key from header
        query_key: API key from query parameter

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
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
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return api_key


def require_auth(api_key: str = Depends(api_key_auth)):
    """Dependency that requires authentication."""
    return api_key
