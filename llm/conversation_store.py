"""
ConversationStore protocol for AMPL Chatbot.

Abstracts conversation storage so the orchestrator can work
with either in-memory dicts or a database backend.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for conversation persistence."""

    async def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get message history for a conversation."""
        ...

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities_json: Optional[Dict] = None,
        lead_score: Optional[float] = None,
    ) -> None:
        """Save a message to the conversation."""
        ...

    async def clear(self, conversation_id: str) -> None:
        """Clear conversation history."""
        ...
