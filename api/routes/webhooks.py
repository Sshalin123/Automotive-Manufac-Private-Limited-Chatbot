"""
Webhook Routes for AMPL Chatbot.

Receives DMS events (payment, delivery, service-complete, job-card)
and triggers chatbot messages to customers.
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..services import get_services
from config.settings import get_settings
from llm.prompt_templates import PromptTemplates, PromptType

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Webhook Request Models ────────────────────────────────────────

class PaymentWebhook(BaseModel):
    customer_id: str
    booking_id: str
    amount: float
    payment_mode: Optional[str] = None
    receipt_url: Optional[str] = None
    conversation_id: Optional[str] = None


class DeliveryWebhook(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    vehicle_model: str
    vehicle_variant: Optional[str] = None
    vehicle_colour: Optional[str] = None
    delivery_date: str
    photo_url: Optional[str] = None
    delay: bool = False
    updated_eta: Optional[str] = None
    message_type: str = Field(
        default="confirmation",
        description="confirmation, thank_you, welcome_to_family",
    )


class ServiceCompleteWebhook(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    vehicle_model: str
    job_card_id: str
    service_type: Optional[str] = None
    work_done: Optional[str] = None


class JobCardWebhook(BaseModel):
    customer_id: str
    conversation_id: Optional[str] = None
    vehicle_model: str
    job_card_id: str
    event: str = Field(..., description="opened, in_progress, closed")
    notes: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/webhooks/payment")
async def payment_webhook(payload: PaymentWebhook, background_tasks: BackgroundTasks):
    """
    Receive payment event from DMS.
    Sends 'Rs.XXX received — Yes/No?' confirmation to customer.
    """
    settings = get_settings()

    confirmation_msg = PromptTemplates.get_user_prompt(
        "payment_confirmation",
        customer_id=payload.customer_id,
        booking_id=payload.booking_id,
        amount=f"{payload.amount:,.2f}",
        payment_mode=payload.payment_mode or "N/A",
        receipt_url=payload.receipt_url or "N/A",
    )

    background_tasks.add_task(
        _dispatch_message,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        message=confirmation_msg,
        event_type="payment_confirmation",
    )

    return {
        "status": "accepted",
        "event": "payment",
        "customer_id": payload.customer_id,
        "booking_id": payload.booking_id,
    }


@router.post("/webhooks/delivery")
async def delivery_webhook(payload: DeliveryWebhook, background_tasks: BackgroundTasks):
    """
    Receive delivery event from DMS.
    Sends confirmation, delay notification, or welcome message.
    """
    if payload.delay:
        message = (
            f"We regret to inform you that the delivery of your {payload.vehicle_model} "
            f"has been delayed. Updated expected date: {payload.updated_eta or 'TBD'}. "
            f"We sincerely apologize for the inconvenience."
        )
    else:
        message = PromptTemplates.get_user_prompt(
            "delivery_notification",
            customer_id=payload.customer_id,
            vehicle_model=payload.vehicle_model,
            vehicle_variant=payload.vehicle_variant or "",
            vehicle_colour=payload.vehicle_colour or "",
            delivery_date=payload.delivery_date,
            photo_url=payload.photo_url or "N/A",
            message_type=payload.message_type,
        )

    background_tasks.add_task(
        _dispatch_message,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        message=message,
        event_type=f"delivery_{payload.message_type}",
    )

    # Schedule post-delivery follow-ups (Day-1 thank you, Day-1 welcome)
    if payload.message_type == "confirmation" and not payload.delay:
        background_tasks.add_task(
            _schedule_post_delivery,
            customer_id=payload.customer_id,
            conversation_id=payload.conversation_id,
            vehicle_model=payload.vehicle_model,
        )

    return {
        "status": "accepted",
        "event": "delivery",
        "customer_id": payload.customer_id,
        "message_type": payload.message_type,
        "delay": payload.delay,
    }


@router.post("/webhooks/service-complete")
async def service_complete_webhook(
    payload: ServiceCompleteWebhook, background_tasks: BackgroundTasks
):
    """
    Receive service-complete event from DMS.
    Sends follow-up with feedback request + escalation matrix.
    """
    settings = get_settings()

    message = (
        f"Your {payload.vehicle_model} (Job Card: {payload.job_card_id}) "
        f"service has been completed."
    )
    if payload.work_done:
        message += f"\n\nWork done: {payload.work_done}"
    message += (
        f"\n\nWe'd love your feedback! Please rate your experience: "
        f"Poor / Fair / Very Good / Excellent"
        f"\n\nFor any concerns, call our toll-free number: {settings.toll_free_number}"
    )

    background_tasks.add_task(
        _dispatch_message,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        message=message,
        event_type="service_complete",
    )

    return {
        "status": "accepted",
        "event": "service_complete",
        "customer_id": payload.customer_id,
        "job_card_id": payload.job_card_id,
    }


@router.post("/webhooks/job-card")
async def job_card_webhook(payload: JobCardWebhook, background_tasks: BackgroundTasks):
    """
    Receive job card events (opened, in_progress, closed).
    """
    event_messages = {
        "opened": (
            f"A service job card ({payload.job_card_id}) has been opened "
            f"for your {payload.vehicle_model}. We'll keep you updated on progress."
        ),
        "in_progress": (
            f"Your {payload.vehicle_model} service is in progress "
            f"(Job Card: {payload.job_card_id})."
        ),
        "closed": (
            f"Service for your {payload.vehicle_model} (Job Card: {payload.job_card_id}) "
            f"is now complete. Please share your feedback!"
        ),
    }

    message = event_messages.get(payload.event, f"Job card update: {payload.event}")
    if payload.notes:
        message += f"\n\nNotes: {payload.notes}"

    background_tasks.add_task(
        _dispatch_message,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        message=message,
        event_type=f"job_card_{payload.event}",
    )

    return {
        "status": "accepted",
        "event": f"job_card_{payload.event}",
        "customer_id": payload.customer_id,
        "job_card_id": payload.job_card_id,
    }


# ── Helpers ───────────────────────────────────────────────────────

def _dispatch_message(
    customer_id: str,
    conversation_id: Optional[str],
    message: str,
    event_type: str,
):
    """Dispatch message to customer (background task)."""
    logger.info(
        f"Dispatching {event_type} message",
        extra={
            "customer_id": customer_id,
            "conversation_id": conversation_id,
            "event_type": event_type,
            "message_length": len(message),
        },
    )
    # In production, this would route through notifications.py
    # to send via WhatsApp API, email, or chat widget.


def _schedule_post_delivery(
    customer_id: str,
    conversation_id: Optional[str],
    vehicle_model: str,
):
    """Schedule post-delivery follow-up messages (background task)."""
    logger.info(
        "Scheduling post-delivery follow-ups",
        extra={
            "customer_id": customer_id,
            "vehicle_model": vehicle_model,
            "follow_ups": ["day_1_thank_you", "day_1_welcome_to_family", "day_15_rc_followup"],
        },
    )
    # In production, this would enqueue tasks in scheduled.py
    # for Day-0 confirmation (already sent), Day-1 thank you,
    # Day-1 welcome, Day-15 RC follow-up.
