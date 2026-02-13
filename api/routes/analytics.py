"""
Analytics Dashboard API routes for AMPL Chatbot (Gap 14.5).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import require_role
from database.session import get_db
from database.models import Conversation, Message, Lead, Feedback

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    dependencies=[Depends(require_role("admin", "manager"))],
)


@router.get("/conversations")
async def conversation_stats(
    days: int = Query(30, ge=1, le=365),
    tenant_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Conversation metrics: count, avg messages, by date range."""
    since = datetime.utcnow() - timedelta(days=days)
    q = select(func.count(Conversation.id)).where(Conversation.started_at >= since)
    if tenant_id:
        q = q.where(Conversation.tenant_id == tenant_id)
    total = (await db.execute(q)).scalar() or 0

    msg_q = (
        select(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.started_at >= since)
    )
    if tenant_id:
        msg_q = msg_q.where(Conversation.tenant_id == tenant_id)
    total_messages = (await db.execute(msg_q)).scalar() or 0

    return {
        "period_days": days,
        "total_conversations": total,
        "total_messages": total_messages,
        "avg_messages_per_conversation": round(total_messages / total, 1) if total else 0,
    }


@router.get("/leads")
async def lead_funnel(
    days: int = Query(30, ge=1, le=365),
    tenant_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Lead funnel: total, warm, hot, routed, conversion rates."""
    since = datetime.utcnow() - timedelta(days=days)
    base = select(func.count(Lead.id)).where(Lead.created_at >= since)
    if tenant_id:
        base = base.where(Lead.tenant_id == tenant_id)

    total = (await db.execute(base)).scalar() or 0
    warm = (await db.execute(base.where(Lead.temperature.in_(["warm", "hot"])))).scalar() or 0
    hot = (await db.execute(base.where(Lead.temperature == "hot"))).scalar() or 0
    routed = (await db.execute(base.where(Lead.routed_at.isnot(None)))).scalar() or 0

    return {
        "period_days": days,
        "total": total,
        "warm": warm,
        "hot": hot,
        "routed": routed,
        "warm_rate": round(warm / total * 100, 1) if total else 0,
        "hot_rate": round(hot / total * 100, 1) if total else 0,
    }


@router.get("/intents")
async def intent_distribution(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Distribution of classified intents."""
    since = datetime.utcnow() - timedelta(days=days)
    q = (
        select(Message.intent, func.count(Message.id))
        .where(Message.created_at >= since, Message.intent.isnot(None))
        .group_by(Message.intent)
        .order_by(func.count(Message.id).desc())
    )
    result = await db.execute(q)
    return {
        "period_days": days,
        "intents": [{"intent": row[0], "count": row[1]} for row in result.all()],
    }


@router.get("/feedback")
async def feedback_stats(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Feedback aggregates: avg rating, NPS, count."""
    since = datetime.utcnow() - timedelta(days=days)
    q = select(
        func.count(Feedback.id),
        func.avg(Feedback.rating),
    ).where(Feedback.created_at >= since)
    row = (await db.execute(q)).one()

    # NPS calculation: promoters (4-5) - detractors (1-2)
    promoters_q = select(func.count(Feedback.id)).where(
        Feedback.created_at >= since, Feedback.rating >= 4
    )
    detractors_q = select(func.count(Feedback.id)).where(
        Feedback.created_at >= since, Feedback.rating <= 2
    )
    promoters = (await db.execute(promoters_q)).scalar() or 0
    detractors = (await db.execute(detractors_q)).scalar() or 0
    total_fb = row[0] or 0
    nps = round((promoters - detractors) / total_fb * 100, 1) if total_fb else 0

    return {
        "period_days": days,
        "count": total_fb,
        "avg_rating": round(float(row[1]), 2) if row[1] else None,
        "nps": nps,
        "promoters": promoters,
        "detractors": detractors,
    }
