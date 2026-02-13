"""
Multi-Channel Notification Dispatcher for AMPL Chatbot.

Routes messages to the appropriate channel:
- Chat widget (default)
- WhatsApp API
- Email

In production, this would integrate with WhatsApp Business API
(e.g., Gupshup, Twilio, Meta Cloud API) and an email service
(e.g., SendGrid, SES). Currently provides the routing layer
and API endpoints.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────

class Channel(str, Enum):
    WIDGET = "widget"
    WHATSAPP = "whatsapp"
    EMAIL = "email"


class NotificationRequest(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    channel: Channel = Channel.WIDGET
    message: str = Field(..., min_length=1)
    subject: Optional[str] = None  # For email
    template_id: Optional[str] = None  # For WhatsApp template messages
    template_vars: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}


class BulkNotificationRequest(BaseModel):
    customer_ids: List[str]
    channel: Channel = Channel.WIDGET
    message: str = Field(..., min_length=1)
    subject: Optional[str] = None
    template_id: Optional[str] = None
    template_vars: Dict[str, str] = {}


class NotificationLog(BaseModel):
    id: str
    customer_id: str
    channel: str
    status: str  # queued, sent, delivered, failed
    message_preview: str
    sent_at: str
    metadata: Dict[str, Any] = {}


# ── In-Memory Log (replace with DB in production) ────────────────

_notification_log: List[NotificationLog] = []
_log_counter = 0


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/notifications/send")
async def send_notification(
    request: NotificationRequest, background_tasks: BackgroundTasks
):
    """Send a notification through the specified channel."""
    global _log_counter
    _log_counter += 1
    log_id = f"notif-{_log_counter}"

    background_tasks.add_task(
        _send_via_channel,
        log_id=log_id,
        customer_id=request.customer_id,
        channel=request.channel,
        message=request.message,
        subject=request.subject,
        template_id=request.template_id,
        template_vars=request.template_vars,
    )

    log_entry = NotificationLog(
        id=log_id,
        customer_id=request.customer_id,
        channel=request.channel.value,
        status="queued",
        message_preview=request.message[:100],
        sent_at=datetime.utcnow().isoformat(),
        metadata=request.metadata,
    )
    _notification_log.append(log_entry)

    return {
        "status": "queued",
        "notification_id": log_id,
        "channel": request.channel.value,
        "customer_id": request.customer_id,
    }


@router.post("/notifications/bulk")
async def send_bulk_notification(
    request: BulkNotificationRequest, background_tasks: BackgroundTasks
):
    """Send the same notification to multiple customers."""
    global _log_counter
    notification_ids = []

    for customer_id in request.customer_ids:
        _log_counter += 1
        log_id = f"notif-{_log_counter}"
        notification_ids.append(log_id)

        background_tasks.add_task(
            _send_via_channel,
            log_id=log_id,
            customer_id=customer_id,
            channel=request.channel,
            message=request.message,
            subject=request.subject,
            template_id=request.template_id,
            template_vars=request.template_vars,
        )

        log_entry = NotificationLog(
            id=log_id,
            customer_id=customer_id,
            channel=request.channel.value,
            status="queued",
            message_preview=request.message[:100],
            sent_at=datetime.utcnow().isoformat(),
        )
        _notification_log.append(log_entry)

    return {
        "status": "queued",
        "total_recipients": len(request.customer_ids),
        "channel": request.channel.value,
        "notification_ids": notification_ids,
    }


@router.get("/notifications/log")
async def get_notification_log(
    customer_id: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = 50,
):
    """Get notification log with optional filters."""
    results = _notification_log

    if customer_id:
        results = [n for n in results if n.customer_id == customer_id]
    if channel:
        results = [n for n in results if n.channel == channel]

    results = results[-limit:]

    return {
        "total": len(results),
        "notifications": [n.model_dump() for n in results],
    }


# ── Channel Dispatchers (Background Tasks) ───────────────────────

def _send_via_channel(
    log_id: str,
    customer_id: str,
    channel: Channel,
    message: str,
    subject: Optional[str] = None,
    template_id: Optional[str] = None,
    template_vars: Dict[str, str] = {},
):
    """Route message to the appropriate channel."""
    try:
        if channel == Channel.WHATSAPP:
            _send_whatsapp(customer_id, message, template_id, template_vars)
        elif channel == Channel.EMAIL:
            _send_email(customer_id, message, subject)
        else:
            _send_widget(customer_id, message)

        # Update log status
        for entry in _notification_log:
            if entry.id == log_id:
                entry.status = "sent"
                break

        logger.info(f"Notification {log_id} sent via {channel.value} to {customer_id}")

    except Exception as e:
        for entry in _notification_log:
            if entry.id == log_id:
                entry.status = "failed"
                entry.metadata["error"] = str(e)
                break
        logger.error(f"Notification {log_id} failed: {e}")


def _send_whatsapp(
    customer_id: str,
    message: str,
    template_id: Optional[str],
    template_vars: Dict[str, str],
):
    """
    Send via WhatsApp Business API.
    In production, integrate with Gupshup / Twilio / Meta Cloud API.
    """
    logger.info(
        f"WhatsApp message to {customer_id}",
        extra={"template_id": template_id, "message_length": len(message)},
    )
    # TODO: Integrate with WhatsApp Business API provider


def _send_email(customer_id: str, message: str, subject: Optional[str]):
    """
    Send via email.
    In production, integrate with SendGrid / SES / SMTP.
    """
    logger.info(
        f"Email to {customer_id}",
        extra={"subject": subject, "message_length": len(message)},
    )
    # TODO: Integrate with email service provider


def _send_widget(customer_id: str, message: str):
    """
    Send via chat widget (push to active conversation).
    In production, this would push via WebSocket or SSE.
    """
    logger.info(
        f"Widget message to {customer_id}",
        extra={"message_length": len(message)},
    )
    # TODO: Push via WebSocket to frontend widget
