"""
Human handoff API routes for AMPL Chatbot (Gap 14.6).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/handoff", tags=["handoff"])


class HandoffResolveRequest(BaseModel):
    notes: str = ""


@router.get("/active")
async def list_active_handoffs(user=Depends(get_current_user)):
    """List all active handoff sessions."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "handoff_manager") or not services.handoff_manager:
        return {"sessions": []}
    sessions = services.handoff_manager.get_active_sessions()
    return {
        "sessions": [
            {
                "conversation_id": s.conversation_id,
                "trigger": s.trigger.value,
                "assigned_agent_id": s.assigned_agent_id,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]
    }


@router.post("/resolve/{conversation_id}")
async def resolve_handoff(
    conversation_id: str,
    request: HandoffResolveRequest,
    user=Depends(get_current_user),
):
    """Resolve a handoff session (agent marks it done)."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "handoff_manager") or not services.handoff_manager:
        raise HTTPException(status_code=404, detail="Handoff not found")
    session = services.handoff_manager.resolve_handoff(conversation_id, request.notes)
    if not session:
        raise HTTPException(status_code=404, detail="No active handoff for this conversation")
    return {"status": "resolved", "conversation_id": conversation_id}
