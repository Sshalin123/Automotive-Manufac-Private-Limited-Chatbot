"""
Rate limiting middleware for AMPL Chatbot API.

Sliding-window counter per client (API key or IP).
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter."""

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self._requests: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health and metrics
        if request.url.path in ("/health", "/metrics", "/docs", "/openapi.json"):
            return await call_next(request)

        client_id = self._get_client_id(request)
        now = time.time()

        # Clean old entries
        window_start = now - self.window_seconds
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        if len(self._requests[client_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._requests[client_id].append(now)
        response = await call_next(request)

        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._requests[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))

        return response

    def _get_client_id(self, request: Request) -> str:
        """Identify client by API key, auth token, or IP."""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}"

        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            return f"token:{auth[7:15]}"

        return f"ip:{request.client.host}" if request.client else "ip:unknown"
