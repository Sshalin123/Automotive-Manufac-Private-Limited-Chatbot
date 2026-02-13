"""
Scheduled Message Dispatcher for AMPL Chatbot.

Manages time-based follow-ups, service reminders, and SLA checks.
In production, this would be backed by a task queue (Celery, APScheduler, etc.).
Currently provides API endpoints to trigger and query scheduled messages.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import get_settings
from llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────

class ScheduledMessageType(str, Enum):
    FOLLOWUP = "followup"
    SERVICE_REMINDER = "service_reminder"
    RC_FOLLOWUP = "rc_followup"
    SLA_CHECK = "sla_check"
    WELCOME_CALL = "welcome_call"


class ScheduledMessage(BaseModel):
    id: Optional[str] = None
    customer_id: str
    conversation_id: Optional[str] = None
    message_type: ScheduledMessageType
    scheduled_at: str
    message: str
    status: str = "pending"  # pending, sent, cancelled
    metadata: Dict[str, Any] = {}


class ScheduleFollowUpRequest(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    event_type: str = Field(..., description="enquiry, delivery, service_complete")
    vehicle_model: Optional[str] = None
    event_date: Optional[str] = None  # ISO format, defaults to now


class ScheduleServiceReminderRequest(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    vehicle_model: str
    delivery_date: str  # ISO format
    current_km: int = 0


# ── In-Memory Store (replace with DB in production) ──────────────

_scheduled_messages: List[ScheduledMessage] = []


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/scheduled/followup")
async def schedule_followup(request: ScheduleFollowUpRequest):
    """
    Schedule follow-up messages based on event type.
    Enquiry: N+0 (immediate), N+1, N+15
    Delivery: Day-0, Day-1 thank you, Day-1 welcome, Day-15 RC
    Service: Day-0 feedback request
    """
    settings = get_settings()
    event_date = datetime.fromisoformat(request.event_date) if request.event_date else datetime.utcnow()
    messages_scheduled = []

    if request.event_type == "enquiry":
        intervals = settings.followup_intervals_list  # [0, 1, 15]
        templates = [
            "Thank you for your enquiry at AMPL! Your Relationship Manager will be in touch shortly.",
            "Hi! Just following up on your enquiry. Have you had a chance to think about your vehicle preferences? We're here to help!",
            "We noticed your enquiry from 15 days ago. We'd love to assist you further — any questions about our vehicles or offers?",
        ]
        for i, day_offset in enumerate(intervals):
            if i < len(templates):
                msg = ScheduledMessage(
                    id=f"followup-{request.customer_id}-{day_offset}d",
                    customer_id=request.customer_id,
                    conversation_id=request.conversation_id,
                    message_type=ScheduledMessageType.FOLLOWUP,
                    scheduled_at=(event_date + timedelta(days=day_offset)).isoformat(),
                    message=templates[i],
                    metadata={"event_type": "enquiry", "day_offset": day_offset},
                )
                _scheduled_messages.append(msg)
                messages_scheduled.append(msg.id)

    elif request.event_type == "delivery":
        delivery_messages = [
            (0, "Your vehicle is ready for delivery! We look forward to handing you the keys."),
            (1, f"Thank you for choosing AMPL for your {request.vehicle_model or 'new vehicle'}! We hope you're enjoying the drive."),
            (1, f"Welcome to the AMPL Family! Your {request.vehicle_model or 'vehicle'} journey starts now. We're always here for you."),
            (15, "Quick check-in: How's your RC registration going? If you need any help with the process, we're here to assist."),
        ]
        for day_offset, template in delivery_messages:
            msg_id = f"delivery-{request.customer_id}-{day_offset}d-{len(messages_scheduled)}"
            msg = ScheduledMessage(
                id=msg_id,
                customer_id=request.customer_id,
                conversation_id=request.conversation_id,
                message_type=ScheduledMessageType.FOLLOWUP if day_offset < 15 else ScheduledMessageType.RC_FOLLOWUP,
                scheduled_at=(event_date + timedelta(days=day_offset)).isoformat(),
                message=template,
                metadata={"event_type": "delivery", "day_offset": day_offset},
            )
            _scheduled_messages.append(msg)
            messages_scheduled.append(msg.id)

    elif request.event_type == "service_complete":
        msg = ScheduledMessage(
            id=f"service-followup-{request.customer_id}",
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            message_type=ScheduledMessageType.FOLLOWUP,
            scheduled_at=event_date.isoformat(),
            message="Your recent service is complete. We'd love to hear about your experience — please rate us!",
            metadata={"event_type": "service_complete"},
        )
        _scheduled_messages.append(msg)
        messages_scheduled.append(msg.id)

    return {
        "status": "scheduled",
        "customer_id": request.customer_id,
        "messages_scheduled": len(messages_scheduled),
        "message_ids": messages_scheduled,
    }


@router.post("/scheduled/service-reminders")
async def schedule_service_reminders(request: ScheduleServiceReminderRequest):
    """
    Schedule service reminders based on delivery date.
    Milestones: 7-day welcome, 1000km/30d, 5000km/180d, 10000km/365d.
    """
    settings = get_settings()
    delivery_date = datetime.fromisoformat(request.delivery_date)
    messages_scheduled = []

    for milestone in settings.service_milestones_list:
        name = milestone["name"]
        days = milestone.get("days", 0)
        km = milestone.get("km")

        reminder_date = delivery_date + timedelta(days=days)

        km_text = f" or {km:,} km" if km else ""
        message = PromptTemplates.get_user_prompt(
            "service_reminder",
            customer_id=request.customer_id,
            vehicle_model=request.vehicle_model,
            milestone_name=name,
            due_at=f"{days} days{km_text} after delivery",
            toll_free=settings.toll_free_number,
        )

        msg = ScheduledMessage(
            id=f"service-{request.customer_id}-{name.lower().replace(' ', '-')}",
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            message_type=ScheduledMessageType.SERVICE_REMINDER,
            scheduled_at=reminder_date.isoformat(),
            message=message,
            metadata={"milestone": name, "km": km, "days": days},
        )
        _scheduled_messages.append(msg)
        messages_scheduled.append(msg.id)

    return {
        "status": "scheduled",
        "customer_id": request.customer_id,
        "vehicle_model": request.vehicle_model,
        "reminders_scheduled": len(messages_scheduled),
        "message_ids": messages_scheduled,
    }


@router.post("/scheduled/sla-check")
async def trigger_sla_check():
    """
    Check for complaints/escalations older than 15 days without resolution.
    In production, this would run on a cron schedule.
    """
    services = get_services()
    stale = []

    if services.is_ready:
        now = datetime.utcnow()
        for conv_id, created_at in services.orchestrator._conversation_created.items():
            stage = services.orchestrator._conversation_stages.get(conv_id, "enquiry")
            age_days = (now - created_at).days
            if age_days >= 15 and stage in ("enquiry", "escalation"):
                stale.append({
                    "conversation_id": conv_id,
                    "stage": stage,
                    "age_days": age_days,
                    "created_at": created_at.isoformat(),
                })

    return {
        "checked_at": datetime.utcnow().isoformat(),
        "stale_conversations": len(stale),
        "conversations": stale,
    }


@router.get("/scheduled/messages")
async def list_scheduled_messages(
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
):
    """List scheduled messages with optional filters."""
    results = _scheduled_messages

    if customer_id:
        results = [m for m in results if m.customer_id == customer_id]
    if status:
        results = [m for m in results if m.status == status]

    return {
        "total": len(results),
        "messages": [m.model_dump() for m in results],
    }


@router.delete("/scheduled/messages/{message_id}")
async def cancel_scheduled_message(message_id: str):
    """Cancel a scheduled message."""
    for msg in _scheduled_messages:
        if msg.id == message_id:
            if msg.status == "sent":
                raise HTTPException(status_code=400, detail="Message already sent")
            msg.status = "cancelled"
            return {"status": "cancelled", "message_id": message_id}

    raise HTTPException(status_code=404, detail="Scheduled message not found")


# ── Import for SLA check ─────────────────────────────────────────

from ..services import get_services
