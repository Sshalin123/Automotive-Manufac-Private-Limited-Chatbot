"""
Cumulative Lead Scorer for AMPL Chatbot (Gap 6.3).

Wraps LeadScorer to maintain per-conversation score history.
Uses 60% cumulative average + 40% peak score blend.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .scoring_model import LeadScorer

logger = logging.getLogger(__name__)


@dataclass
class CumulativeScoreResult:
    """Result of cumulative scoring."""
    current_score: float  # This message's raw score
    cumulative_score: float  # Blended score
    peak_score: float  # Highest score in conversation
    message_count: int
    temperature: str  # hot, warm, cold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_score": round(self.current_score, 2),
            "cumulative_score": round(self.cumulative_score, 2),
            "peak_score": round(self.peak_score, 2),
            "message_count": self.message_count,
            "temperature": self.temperature,
        }


class CumulativeScorer:
    """
    Cumulative lead scorer that tracks per-conversation scoring history.

    Formula: final = 0.6 * cumulative_avg + 0.4 * peak_score
    Decay: messages with no buying signals reduce cumulative by 5%.
    """

    BLEND_CUMULATIVE = 0.6
    BLEND_PEAK = 0.4
    DECAY_FACTOR = 0.95  # 5% decay for low-signal messages
    LOW_SIGNAL_THRESHOLD = 15  # Scores below this are "no signal"

    def __init__(
        self,
        base_scorer: LeadScorer,
        hot_threshold: int = 70,
        warm_threshold: int = 50,
    ):
        self.base_scorer = base_scorer
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        # Per-conversation state: {conv_id: [score1, score2, ...]}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._peak: Dict[str, float] = defaultdict(float)

    def score(
        self,
        message: str,
        conversation_id: str,
        intent_result: Any = None,
        entities: Any = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> CumulativeScoreResult:
        """
        Score a message with cumulative context.

        Args:
            message: Current user message
            conversation_id: Conversation identifier
            intent_result: IntentResult from classifier
            entities: ExtractedEntities
            history: Conversation history (for base scorer)

        Returns:
            CumulativeScoreResult with blended score
        """
        # Get raw score from base scorer
        raw_result = self.base_scorer.score(
            intent_result=intent_result,
            entities=entities,
        )
        current_score = raw_result.score

        # Update history
        scores = self._history[conversation_id]
        scores.append(current_score)

        # Update peak
        if current_score > self._peak[conversation_id]:
            self._peak[conversation_id] = current_score

        peak = self._peak[conversation_id]

        # Calculate cumulative average with decay
        if current_score < self.LOW_SIGNAL_THRESHOLD and len(scores) > 1:
            # Apply decay to previous scores
            decayed_avg = sum(scores[:-1]) / len(scores[:-1]) * self.DECAY_FACTOR
            cumulative_avg = (decayed_avg * (len(scores) - 1) + current_score) / len(scores)
        else:
            cumulative_avg = sum(scores) / len(scores)

        # Blend
        blended = (self.BLEND_CUMULATIVE * cumulative_avg) + (self.BLEND_PEAK * peak)
        blended = min(100, max(0, blended))

        # Determine temperature
        if blended >= self.hot_threshold:
            temperature = "hot"
        elif blended >= self.warm_threshold:
            temperature = "warm"
        else:
            temperature = "cold"

        return CumulativeScoreResult(
            current_score=current_score,
            cumulative_score=blended,
            peak_score=peak,
            message_count=len(scores),
            temperature=temperature,
        )

    def get_score(self, conversation_id: str) -> float:
        """Get current cumulative score for a conversation."""
        scores = self._history.get(conversation_id, [])
        if not scores:
            return 0.0
        peak = self._peak.get(conversation_id, 0.0)
        avg = sum(scores) / len(scores)
        return (self.BLEND_CUMULATIVE * avg) + (self.BLEND_PEAK * peak)

    def reset(self, conversation_id: str):
        """Reset scoring history for a conversation."""
        self._history.pop(conversation_id, None)
        self._peak.pop(conversation_id, None)
