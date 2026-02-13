"""
Admin API Routes for AMPL Chatbot.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..services import get_services

logger = logging.getLogger(__name__)

router = APIRouter()

system_start_time = datetime.utcnow()


# ── Models ────────────────────────────────────────────

class SystemHealth(BaseModel):
    status: str
    services: Dict[str, Any]
    uptime_seconds: float
    timestamp: str


class IndexStats(BaseModel):
    total_vectors: int
    dimension: int
    namespaces: Dict[str, int]
    index_fullness: float


class IngestionStatus(BaseModel):
    total_documents: int
    documents_by_type: Dict[str, int]
    last_ingestion: Optional[str]
    pending_documents: int


class AnalyticsSummary(BaseModel):
    total_conversations: int
    total_messages: int
    average_response_time_ms: float
    intent_distribution: Dict[str, int]
    lead_conversion_rate: float
    top_queries: List[Dict[str, Any]]


class ConfigUpdate(BaseModel):
    key: str
    value: Any


# ── Endpoints ─────────────────────────────────────────

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Detailed system health including all sub-services."""
    services = get_services()
    uptime = (datetime.utcnow() - system_start_time).total_seconds()

    svc_health = services.health() if services else {}
    status = "healthy" if services.is_ready else "degraded"

    return SystemHealth(
        status=status,
        services=svc_health,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/index/stats", response_model=IndexStats)
async def get_index_stats():
    """Vector index statistics from Pinecone."""
    services = get_services()
    if services.pinecone_client:
        try:
            stats = services.pinecone_client.get_stats()
            return IndexStats(
                total_vectors=stats.get("total_vector_count", 0),
                dimension=stats.get("dimension", 1024),
                namespaces=stats.get("namespaces", {}),
                index_fullness=stats.get("index_fullness", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")

    return IndexStats(total_vectors=0, dimension=1024, namespaces={}, index_fullness=0.0)


@router.get("/ingestion/status", response_model=IngestionStatus)
async def get_ingestion_status():
    """Current ingestion status."""
    return IngestionStatus(
        total_documents=0,
        documents_by_type={"inventory": 0, "sales": 0, "insurance": 0, "faq": 0},
        last_ingestion=None,
        pending_documents=0,
    )


@router.post("/ingestion/trigger")
async def trigger_ingestion(
    document_type: str = Query(..., description="Type: inventory|sales|insurance|faq|all"),
    source_path: Optional[str] = Query(None, description="Source file path"),
):
    """Trigger document ingestion (placeholder — use CLI for now)."""
    valid = ("inventory", "sales", "insurance", "faq", "all")
    if document_type not in valid:
        raise HTTPException(status_code=400, detail=f"Must be one of: {valid}")

    return {
        "message": f"Ingestion triggered for {document_type}",
        "status": "queued",
        "hint": "Use `python -m ingest_ampl.main` CLI for actual ingestion",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(days: int = Query(7, ge=1, le=90)):
    """Analytics summary."""
    services = get_services()
    total_convos = 0
    total_msgs = 0

    if services.is_ready:
        convos = services.orchestrator._conversations
        total_convos = len(convos)
        total_msgs = sum(len(m) for m in convos.values())

    return AnalyticsSummary(
        total_conversations=total_convos,
        total_messages=total_msgs,
        average_response_time_ms=0,
        intent_distribution={"buy": 0, "finance": 0, "test_drive": 0, "service": 0, "info": 0},
        lead_conversion_rate=0.0,
        top_queries=[],
    )


@router.get("/config")
async def get_config():
    """Current runtime configuration (safe subset)."""
    services = get_services()
    s = services.settings
    if s:
        return {
            "llm_provider": s.llm_provider,
            "llm_model_id": s.llm_model_id,
            "max_tokens": s.max_tokens,
            "temperature": s.temperature,
            "top_k": s.top_k,
            "similarity_threshold": s.similarity_threshold,
            "lead_score_threshold_hot": s.lead_score_threshold_hot,
            "lead_score_threshold_warm": s.lead_score_threshold_warm,
            "auto_route_leads": s.auto_route_leads,
        }
    return {}


@router.patch("/config")
async def update_config(update: ConfigUpdate):
    """Update a runtime config value."""
    services = get_services()
    allowed = ("lead_score_threshold_hot", "lead_score_threshold_warm", "auto_route_leads")

    if update.key not in allowed:
        raise HTTPException(status_code=400, detail=f"Updatable keys: {allowed}")

    if services.lead_scorer and update.key.startswith("lead_score_threshold"):
        if update.key == "lead_score_threshold_hot":
            services.lead_scorer.adjust_thresholds(hot=int(update.value))
        elif update.key == "lead_score_threshold_warm":
            services.lead_scorer.adjust_thresholds(warm=int(update.value))

    return {"message": "Updated", "key": update.key, "value": update.value}


@router.post("/cache/clear")
async def clear_cache(cache_type: str = Query("all")):
    """Clear caches."""
    valid = ("embeddings", "responses", "sessions", "all")
    if cache_type not in valid:
        raise HTTPException(status_code=400, detail=f"Must be one of: {valid}")
    return {"message": f"Cache cleared: {cache_type}", "timestamp": datetime.utcnow().isoformat()}
