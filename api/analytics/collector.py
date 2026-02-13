"""
Analytics Collector for AMPL Chatbot (Gap 14.5).

Records events to the database for analytics queries.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnalyticsCollector:
    """Collects and records analytics events."""

    def record_chat(
        self,
        conversation_id: str,
        intent: str,
        confidence: float,
        lead_score: float,
        latency_ms: float,
        tenant_id: Optional[str] = None,
    ):
        """Record a chat interaction event (for async DB write)."""
        # In production, this would write to DB or a queue.
        # For now, metrics are recorded via Prometheus middleware.
        logger.debug(
            f"Analytics: conv={conversation_id} intent={intent} "
            f"conf={confidence:.2f} score={lead_score:.1f} latency={latency_ms:.0f}ms"
        )

    def record_lead_event(
        self,
        lead_id: str,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        logger.debug(f"Analytics: lead={lead_id} event={event_type}")

    def record_feedback(
        self,
        conversation_id: str,
        rating: int,
        comment: Optional[str] = None,
    ):
        logger.debug(f"Analytics: feedback conv={conversation_id} rating={rating}")
