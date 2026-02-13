"""
Human Agent Handoff Manager for AMPL Chatbot (Gap 14.6).

Manages escalation triggers, agent assignment, and handback.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HandoffTrigger(Enum):
    """Reasons for handoff to human agent."""
    USER_REQUEST = "user_request"
    LOW_CONFIDENCE = "low_confidence"
    HOT_LEAD = "hot_lead"
    COMPLAINT = "complaint"
    REPEATED_FAILURE = "repeated_failure"


@dataclass
class HandoffSession:
    """Active handoff session."""
    conversation_id: str
    trigger: HandoffTrigger
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    notes: str = ""


class HandoffManager:
    """
    Manages human agent handoff.

    Trigger conditions:
    1. User explicitly asks ("talk to agent", "human", "manager")
    2. 3+ consecutive low-confidence responses
    3. Hot lead with complex query
    4. Complaint escalation
    """

    HANDOFF_KEYWORDS = {
        "talk to agent", "talk to human", "speak to manager",
        "real person", "human agent", "transfer to agent",
        "connect me to", "i want to talk", "agent se baat",
        "manager se baat", "insaan se baat",
    }

    LOW_CONFIDENCE_THRESHOLD = 0.4
    LOW_CONFIDENCE_STREAK = 3

    def __init__(self):
        self._active_sessions: Dict[str, HandoffSession] = {}
        self._low_confidence_counts: Dict[str, int] = {}
        self._agent_queue: List[str] = []  # agent_ids
        self._agent_index = 0

    def check_trigger(
        self,
        conversation_id: str,
        message: str,
        intent: Optional[str] = None,
        confidence: float = 1.0,
        lead_score: float = 0.0,
    ) -> Optional[HandoffTrigger]:
        """
        Check if a handoff should be triggered.

        Returns trigger reason or None.
        """
        # Already in handoff?
        if conversation_id in self._active_sessions:
            return None

        # 1. User request
        message_lower = message.lower()
        if any(kw in message_lower for kw in self.HANDOFF_KEYWORDS):
            return HandoffTrigger.USER_REQUEST

        # 2. Low confidence streak
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            self._low_confidence_counts[conversation_id] = (
                self._low_confidence_counts.get(conversation_id, 0) + 1
            )
            if self._low_confidence_counts[conversation_id] >= self.LOW_CONFIDENCE_STREAK:
                return HandoffTrigger.LOW_CONFIDENCE
        else:
            self._low_confidence_counts[conversation_id] = 0

        # 3. Complaint escalation
        if intent in ("complaint", "escalation"):
            return HandoffTrigger.COMPLAINT

        # 4. Hot lead
        if lead_score >= 70 and confidence < 0.5:
            return HandoffTrigger.HOT_LEAD

        return None

    def initiate_handoff(
        self,
        conversation_id: str,
        trigger: HandoffTrigger,
    ) -> HandoffSession:
        """Create a handoff session and assign an agent."""
        agent_id = self._assign_agent()
        session = HandoffSession(
            conversation_id=conversation_id,
            trigger=trigger,
            assigned_agent_id=agent_id,
        )
        self._active_sessions[conversation_id] = session
        logger.info(f"Handoff initiated: {conversation_id} -> agent {agent_id} ({trigger.value})")
        return session

    def resolve_handoff(self, conversation_id: str, notes: str = "") -> Optional[HandoffSession]:
        """Resolve a handoff session (agent marks it done)."""
        session = self._active_sessions.pop(conversation_id, None)
        if session:
            session.resolved_at = datetime.utcnow()
            session.notes = notes
            logger.info(f"Handoff resolved: {conversation_id}")
        return session

    def is_in_handoff(self, conversation_id: str) -> bool:
        return conversation_id in self._active_sessions

    def get_session(self, conversation_id: str) -> Optional[HandoffSession]:
        return self._active_sessions.get(conversation_id)

    def set_agents(self, agent_ids: List[str]):
        """Set available agent IDs for round-robin assignment."""
        self._agent_queue = agent_ids

    def _assign_agent(self) -> Optional[str]:
        """Round-robin agent assignment."""
        if not self._agent_queue:
            return None
        agent = self._agent_queue[self._agent_index % len(self._agent_queue)]
        self._agent_index += 1
        return agent

    def get_active_sessions(self) -> List[HandoffSession]:
        return list(self._active_sessions.values())
