"""
Export & Compliance (GDPR) routes for AMPL Chatbot (Gap 14.11).
"""

import csv
import io
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import require_role, get_current_user
from database.session import get_db
from database.models import Conversation, Message, Lead, Feedback, AuditLog
from database.repositories import AuditLogRepository

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/compliance",
    tags=["compliance"],
    dependencies=[Depends(require_role("admin"))],
)


@router.get("/export/{conversation_id}")
async def export_conversation(
    conversation_id: str,
    format: str = Query("json", regex="^(json|csv)$"),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a full conversation as JSON or CSV."""
    # Log the export action
    audit = AuditLogRepository(db)
    await audit.log(
        action="data_export",
        user_id=user.get("sub"),
        resource_type="conversation",
        resource_id=conversation_id,
    )

    # Fetch messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "role", "content", "intent", "lead_score"])
        for msg in messages:
            writer.writerow([
                msg.created_at.isoformat(),
                msg.role,
                msg.content,
                msg.intent or "",
                msg.lead_score or "",
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=conversation_{conversation_id}.csv"},
        )

    # JSON format
    data = {
        "conversation_id": conversation_id,
        "messages": [
            {
                "timestamp": msg.created_at.isoformat(),
                "role": msg.role,
                "content": msg.content,
                "intent": msg.intent,
                "lead_score": msg.lead_score,
            }
            for msg in messages
        ],
    }
    return data


@router.delete("/erase/{customer_identifier}")
async def right_to_erasure(
    customer_identifier: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Right to erasure: delete all data for a customer (phone or email).

    This permanently removes all conversations, messages, leads,
    and feedback associated with the customer.
    """
    audit = AuditLogRepository(db)
    await audit.log(
        action="data_erase",
        user_id=user.get("sub"),
        resource_type="customer",
        resource_id=customer_identifier,
    )

    # Find conversations by phone or email
    convs = await db.execute(
        select(Conversation.id).where(
            (Conversation.customer_phone == customer_identifier) |
            (Conversation.customer_email == customer_identifier)
        )
    )
    conv_ids = [row[0] for row in convs.all()]

    if not conv_ids:
        raise HTTPException(status_code=404, detail="No data found for this customer")

    # Delete feedback, messages, leads, conversations (cascade should handle most)
    for conv_id in conv_ids:
        await db.execute(delete(Feedback).where(Feedback.conversation_id == conv_id))
        await db.execute(delete(Message).where(Message.conversation_id == conv_id))
        await db.execute(delete(Lead).where(Lead.conversation_id == conv_id))
        await db.execute(delete(Conversation).where(Conversation.id == conv_id))

    await db.flush()

    return {
        "status": "erased",
        "customer_identifier": customer_identifier,
        "conversations_deleted": len(conv_ids),
    }


@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """List recent audit log entries."""
    audit = AuditLogRepository(db)
    entries = await audit.get_recent(limit=limit)
    return {
        "entries": [
            {
                "id": e.id,
                "action": e.action,
                "user_id": e.user_id,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]
    }
