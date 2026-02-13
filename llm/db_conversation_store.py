"""
Database-backed ConversationStore for AMPL Chatbot.

Implements the ConversationStore protocol using the repository layer.
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from database.repositories import ConversationRepository

logger = logging.getLogger(__name__)


class DbConversationStore:
    """Persistent conversation store backed by PostgreSQL."""

    def __init__(self, session: AsyncSession):
        self._repo = ConversationRepository(session)

    async def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get message history as list of role/content dicts."""
        messages = await self._repo.get_messages(conversation_id, limit=50)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities_json: Optional[Dict] = None,
        lead_score: Optional[float] = None,
    ) -> None:
        """Save a message to the database."""
        # Ensure conversation exists
        conv = await self._repo.get_by_id(conversation_id)
        if not conv:
            await self._repo.create(conversation_id=conversation_id)

        await self._repo.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            intent=intent,
            entities_json=entities_json,
            lead_score=lead_score,
        )

    async def clear(self, conversation_id: str) -> None:
        """Clear is a no-op for DB store (messages are retained)."""
        logger.debug(f"Clear called for conversation {conversation_id} (no-op in DB mode)")
