"""
Prometheus metrics middleware for AMPL Chatbot API.

Exposes /metrics endpoint with request counters, latency histograms,
and custom business metrics.
"""

import logging
import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
    )

    # Request metrics
    REQUEST_COUNT = Counter(
        "ampl_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    REQUEST_LATENCY = Histogram(
        "ampl_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    ACTIVE_REQUESTS = Gauge(
        "ampl_http_active_requests",
        "Currently active HTTP requests",
    )

    # Business metrics
    INTENT_COUNT = Counter(
        "ampl_intent_classification_total",
        "Intent classifications",
        ["intent"],
    )
    LEAD_SCORE_HIST = Histogram(
        "ampl_lead_score",
        "Lead score distribution",
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    RETRIEVAL_LATENCY = Histogram(
        "ampl_retrieval_duration_seconds",
        "Vector retrieval latency",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    LLM_LATENCY = Histogram(
        "ampl_llm_duration_seconds",
        "LLM generation latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    CACHE_HITS = Counter("ampl_cache_hits_total", "Cache hits", ["cache_type"])
    CACHE_MISSES = Counter("ampl_cache_misses_total", "Cache misses", ["cache_type"])

    PROMETHEUS_AVAILABLE = True

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed, metrics disabled")


def record_intent(intent: str):
    """Record an intent classification event."""
    if PROMETHEUS_AVAILABLE:
        INTENT_COUNT.labels(intent=intent).inc()


def record_lead_score(score: float):
    """Record a lead score."""
    if PROMETHEUS_AVAILABLE:
        LEAD_SCORE_HIST.observe(score)


def record_retrieval_latency(seconds: float):
    """Record retrieval latency."""
    if PROMETHEUS_AVAILABLE:
        RETRIEVAL_LATENCY.observe(seconds)


def record_llm_latency(seconds: float):
    """Record LLM generation latency."""
    if PROMETHEUS_AVAILABLE:
        LLM_LATENCY.observe(seconds)


def record_cache_hit(cache_type: str):
    """Record a cache hit."""
    if PROMETHEUS_AVAILABLE:
        CACHE_HITS.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record a cache miss."""
    if PROMETHEUS_AVAILABLE:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records HTTP request metrics."""

    async def dispatch(self, request: Request, call_next):
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)

        ACTIVE_REQUESTS.inc()
        start = time.time()

        try:
            response = await call_next(request)
        except Exception:
            ACTIVE_REQUESTS.dec()
            raise

        duration = time.time() - start
        endpoint = request.url.path

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)
        ACTIVE_REQUESTS.dec()

        return response


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="prometheus-client not installed",
            status_code=503,
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
