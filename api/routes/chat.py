"""
Chat API Routes for AMPL Chatbot.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..services import get_services
from llm.orchestrator import ChatRequest as OrchestratorRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    include_sources: bool = True
    max_chunks: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    id: str
    source: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    query: str
    sources: List[Source] = []
    lead_score: Optional[int] = None
    lead_priority: Optional[str] = None
    intent: Optional[str] = None
    suggested_actions: List[str] = []
    processing_time_ms: float
    timestamp: str


class ConversationHistoryItem(BaseModel):
    role: str
    content: str
    timestamp: str


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ConversationHistoryItem]
    turn_count: int
    created_at: str


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Process a chat message through the full RAG pipeline.

    1. Classify intent  2. Embed query  3. Retrieve context
    4. Generate LLM response  5. Score lead  6. Return response
    """
    services = get_services()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # If orchestrator is initialised, run the real pipeline
    if services.is_ready:
        try:
            orch_request = OrchestratorRequest(
                conversation_id=conversation_id,
                message=request.message,
                user_context=request.user_context,
                max_chunks=request.max_chunks,
                include_sources=request.include_sources,
            )
            result = await services.orchestrator.process(orch_request)

            sources = [
                Source(id=s["id"], source=s["source"], score=s["score"], preview=s["preview"])
                for s in result.sources
            ]

            background_tasks.add_task(
                _log_chat_analytics,
                result.conversation_id,
                request.message,
                result.intent,
                result.lead_score,
            )

            return ChatResponse(
                response=result.response,
                conversation_id=result.conversation_id,
                query=result.query,
                sources=sources,
                lead_score=result.lead_score,
                lead_priority=result.lead_priority,
                intent=result.intent,
                suggested_actions=result.suggested_actions,
                processing_time_ms=result.processing_time_ms,
                timestamp=result.timestamp,
            )

        except Exception as e:
            logger.error(f"Orchestrator error, falling back to mock: {e}")

    # Fallback: mock response when services are not ready
    start_time = datetime.utcnow()
    mock = _generate_mock_response(request.message)
    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    background_tasks.add_task(
        _log_chat_analytics, conversation_id, request.message, mock["intent"], mock["lead_score"]
    )

    return ChatResponse(
        response=mock["response"],
        conversation_id=conversation_id,
        query=request.message,
        sources=mock["sources"],
        lead_score=mock["lead_score"],
        lead_priority=mock["lead_priority"],
        intent=mock["intent"],
        suggested_actions=mock["suggested_actions"],
        processing_time_ms=round(processing_time, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/chat/{conversation_id}/history", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """Get conversation history."""
    services = get_services()
    messages_raw: List[Dict[str, Any]] = []

    if services.is_ready:
        messages_raw = services.orchestrator.get_conversation(conversation_id)

    if not messages_raw:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = [
        ConversationHistoryItem(
            role=m["role"], content=m["content"], timestamp=m.get("timestamp", "")
        )
        for m in messages_raw
    ]

    return ConversationHistory(
        conversation_id=conversation_id,
        messages=messages,
        turn_count=len([m for m in messages if m.role == "user"]),
        created_at=messages[0].timestamp if messages else datetime.utcnow().isoformat(),
    )


@router.delete("/chat/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a conversation."""
    services = get_services()
    if services.is_ready:
        services.orchestrator.clear_conversation(conversation_id)
        return {"message": "Conversation cleared", "conversation_id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/chat/stats")
async def get_chat_stats():
    """Get chat statistics."""
    services = get_services()
    if services.is_ready:
        convos = services.orchestrator._conversations
        return {
            "total_conversations": len(convos),
            "total_messages": sum(len(msgs) for msgs in convos.values()),
            "active_conversations": len(convos),
        }
    return {"total_conversations": 0, "total_messages": 0, "active_conversations": 0}


# ── Helpers ───────────────────────────────────────────────────────

def _generate_mock_response(message: str) -> Dict[str, Any]:
    """Fallback mock response when orchestrator is unavailable."""
    ml = message.lower()

    if any(w in ml for w in ("buy", "purchase", "book", "price")):
        intent, score, priority = "buy", 75, "hot"
        actions = ["Book test drive", "Get quote", "Talk to sales"]
    elif any(w in ml for w in ("emi", "loan", "finance")):
        intent, score, priority = "finance", 65, "warm"
        actions = ["Calculate EMI", "Check eligibility"]
    elif any(w in ml for w in ("test drive", "demo")):
        intent, score, priority = "test_drive", 70, "hot"
        actions = ["Schedule test drive", "Home test drive"]
    else:
        intent, score, priority = "info", 30, "cold"
        actions = ["View vehicles", "Browse FAQs"]

    responses = {
        "buy": "I'd be happy to help you with your purchase! Could you tell me which vehicle you're interested in and your budget range?",
        "finance": "We offer competitive EMI options starting from 7.5% interest rate with tenures up to 7 years. Want me to calculate an EMI for a specific vehicle?",
        "test_drive": "I can help you schedule a test drive! Which vehicle would you like to experience, and what's your preferred date and time?",
        "info": "Hello! I'm here to help you explore our vehicle lineup. Ask me about specific models, pricing, financing, or schedule a test drive.",
    }

    return {
        "response": responses[intent],
        "intent": intent,
        "lead_score": score,
        "lead_priority": priority,
        "suggested_actions": actions,
        "sources": [
            Source(id="mock-1", source="vehicle_catalog", score=0.85, preview="Vehicle specifications and pricing...")
        ] if intent != "info" else [],
    }


def _log_chat_analytics(conversation_id: str, message: str, intent: str, lead_score: int):
    """Log chat analytics (background task)."""
    logger.info(
        "Chat analytics",
        extra={
            "conversation_id": conversation_id,
            "message_length": len(message),
            "intent": intent,
            "lead_score": lead_score,
        },
    )
