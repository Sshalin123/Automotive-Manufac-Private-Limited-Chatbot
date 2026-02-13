"""
WebSocket Connection Manager for AMPL Chatbot (Gap 14.3).

Tracks active WebSocket connections and broadcasts messages.
"""

import logging
from typing import Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections by conversation_id."""

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if conversation_id not in self._connections:
            self._connections[conversation_id] = []
        self._connections[conversation_id].append(websocket)
        logger.info(f"WS connected: {conversation_id} (total: {self.active_count})")

    def disconnect(self, websocket: WebSocket, conversation_id: str):
        """Remove a WebSocket connection."""
        if conversation_id in self._connections:
            self._connections[conversation_id] = [
                ws for ws in self._connections[conversation_id] if ws != websocket
            ]
            if not self._connections[conversation_id]:
                del self._connections[conversation_id]
        logger.info(f"WS disconnected: {conversation_id}")

    async def send_message(self, conversation_id: str, message: dict):
        """Send a message to all connections for a conversation."""
        connections = self._connections.get(conversation_id, [])
        dead = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws, conversation_id)

    async def broadcast(self, message: dict):
        """Broadcast to all connected clients."""
        for conv_id in list(self._connections.keys()):
            await self.send_message(conv_id, message)

    @property
    def active_count(self) -> int:
        return sum(len(conns) for conns in self._connections.values())

    def is_connected(self, conversation_id: str) -> bool:
        return bool(self._connections.get(conversation_id))
