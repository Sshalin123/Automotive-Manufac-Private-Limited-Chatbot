"""
Real-time communication routes (WebSocket + SSE) for AMPL Chatbot (Gap 14.3).
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import StreamingResponse

from api.realtime.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["realtime"])

# Singleton connection manager
manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    return manager


@router.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time chat.

    Receives: {"message": "user text", "metadata": {...}}
    Sends: {"type": "token"|"complete"|"error", "data": "..."}
    """
    await manager.connect(websocket, conversation_id)
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            if not user_message:
                await websocket.send_json({"type": "error", "data": "Empty message"})
                continue

            # Process via orchestrator (injected at runtime)
            try:
                from api.services import get_services
                services = get_services()
                if not services.is_ready:
                    await websocket.send_json({"type": "error", "data": "Service not ready"})
                    continue

                from llm.orchestrator import ChatRequest
                request = ChatRequest(
                    message=user_message,
                    conversation_id=conversation_id,
                )
                result = await services.orchestrator.process(request)

                # Send complete response
                await websocket.send_json({
                    "type": "complete",
                    "data": result.response,
                    "metadata": {
                        "intent": result.intent,
                        "confidence": result.confidence,
                        "lead_score": result.lead_score,
                    }
                })

            except Exception as e:
                logger.error(f"WS processing error: {e}")
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id)


@router.get("/sse/chat/{conversation_id}")
async def sse_chat(
    request: Request,
    conversation_id: str,
    message: str = Query(..., description="User message"),
):
    """
    SSE fallback for clients that can't use WebSocket.

    Streams response tokens as server-sent events.
    """
    async def event_generator():
        try:
            from api.services import get_services
            services = get_services()
            if not services.is_ready:
                yield f"data: {json.dumps({'type': 'error', 'data': 'Service not ready'})}\n\n"
                return

            from llm.orchestrator import ChatRequest
            chat_request = ChatRequest(
                message=message,
                conversation_id=conversation_id,
            )
            result = await services.orchestrator.process(chat_request)

            yield f"data: {json.dumps({'type': 'complete', 'data': result.response})}\n\n"

        except Exception as e:
            logger.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
