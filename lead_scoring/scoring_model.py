"""
Lead Scoring Model for AMPL Chatbot.

Implements a rule-based scoring system with optional ML enhancement
for achieving 90%+ lead qualification accuracy.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .intent_classifier import Intent, IntentResult, Timeline, ContactWillingness
from .entity_extractor import ExtractedEntities

logger = logging.getLogger(__name__)


class LeadPriority(Enum):
    """Lead priority levels."""
    HOT = "hot"          # Score >= 70 - Immediate follow-up
    WARM = "warm"        # Score 50-69 - Standard follow-up
    COLD = "cold"        # Score < 50 - Nurture campaign
    DISQUALIFIED = "disqualified"  # Not a valid lead


@dataclass
class LeadScore:
    """Lead score result."""
    score: int  # 0-100
    priority: LeadPriority
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)
    disqualification_reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "priority": self.priority.value,
            "score_breakdown": self.score_breakdown,
            "confidence": self.confidence,
            "signals": self.signals,
            "disqualification_reasons": self.disqualification_reasons,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class LeadScorer:
    """
    Scores leads based on intent, entities, and conversation signals.

    Scoring Rules (0-100):
    - BUY intent: +30
    - FINANCE intent: +25
    - TEST_DRIVE intent: +20
    - Specific model mentioned: +15
    - Budget mentioned: +10
    - Timeline IMMEDIATE: +15
    - Timeline THIS_MONTH: +10
    - Contact willingness YES: +10
    - Repeat engagement (3+ turns): +10
    - Trade-in mentioned: +5
    - Location provided: +5
    - Features specified: +3

    Thresholds:
    - Score >= 70: Hot Lead
    - Score 50-69: Warm Lead
    - Score < 50: Cold Lead
    """

    # Scoring weights
    SCORING_RULES = {
        # Intent scores
        "intent_buy": 30,
        "intent_finance": 25,
        "intent_test_drive": 20,
        "intent_exchange": 15,
        "intent_service": 5,
        "intent_insurance": 10,
        "intent_info": 0,
        "intent_complaint": -20,

        # Entity scores
        "model_mentioned": 15,
        "budget_mentioned": 10,
        "budget_high": 5,  # Budget > 15 lakh
        "trade_in_mentioned": 5,
        "location_provided": 5,
        "contact_provided": 10,
        "features_specified": 3,
        "color_preference": 2,

        # Timeline scores
        "timeline_immediate": 15,
        "timeline_this_month": 10,
        "timeline_this_quarter": 5,
        "timeline_exploring": 0,

        # Engagement scores
        "repeat_engagement": 10,  # 3+ conversation turns
        "returning_visitor": 8,   # Previous conversation
        "contact_willing": 10,

        # Negative signals
        "just_browsing": -5,
        "competitor_mention": -3,
        "price_objection": -5,
    }

    # Priority thresholds
    HOT_THRESHOLD = 70
    WARM_THRESHOLD = 50

    def __init__(self, custom_rules: Optional[Dict[str, int]] = None):
        """
        Initialize the lead scorer.

        Args:
            custom_rules: Optional custom scoring rules to override defaults
        """
        self.rules = self.SCORING_RULES.copy()
        if custom_rules:
            self.rules.update(custom_rules)

    def score(
        self,
        intent_result: IntentResult,
        entities: ExtractedEntities,
        conversation_turns: int = 1,
        is_returning: bool = False,
        additional_signals: Optional[Dict[str, Any]] = None
    ) -> LeadScore:
        """
        Calculate lead score based on intent, entities, and signals.

        Args:
            intent_result: Intent classification result
            entities: Extracted entities from messages
            conversation_turns: Number of conversation turns
            is_returning: Whether this is a returning visitor
            additional_signals: Optional additional signals

        Returns:
            LeadScore with score and breakdown
        """
        score = 0
        breakdown: Dict[str, int] = {}
        signals: List[str] = []
        disqualifications: List[str] = []
        recommendations: List[str] = []

        # Score based on primary intent
        intent_score = self._score_intent(intent_result.primary_intent)
        if intent_score != 0:
            score += intent_score
            breakdown[f"intent_{intent_result.primary_intent.value}"] = intent_score
            signals.append(f"Primary intent: {intent_result.primary_intent.value}")

        # Bonus for secondary high-value intents
        for secondary in intent_result.secondary_intents:
            if secondary in [Intent.BUY, Intent.FINANCE, Intent.TEST_DRIVE]:
                bonus = self._score_intent(secondary) // 3
                score += bonus
                breakdown[f"secondary_{secondary.value}"] = bonus
                signals.append(f"Secondary intent: {secondary.value}")

        # Score based on timeline
        timeline_score = self._score_timeline(intent_result.timeline)
        if timeline_score != 0:
            score += timeline_score
            breakdown["timeline"] = timeline_score
            signals.append(f"Timeline: {intent_result.timeline.value}")

        # Score based on contact willingness
        if intent_result.contact_willingness == ContactWillingness.YES:
            score += self.rules["contact_willing"]
            breakdown["contact_willing"] = self.rules["contact_willing"]
            signals.append("Customer willing to be contacted")

        # Score based on extracted entities
        entity_scores = self._score_entities(entities)
        for key, value in entity_scores.items():
            score += value
            breakdown[key] = value
            if value > 0:
                signals.append(f"Entity detected: {key}")

        # Engagement scoring
        if conversation_turns >= 3:
            score += self.rules["repeat_engagement"]
            breakdown["repeat_engagement"] = self.rules["repeat_engagement"]
            signals.append(f"Engaged conversation ({conversation_turns} turns)")

        if is_returning:
            score += self.rules["returning_visitor"]
            breakdown["returning_visitor"] = self.rules["returning_visitor"]
            signals.append("Returning visitor")

        # Process additional signals
        if additional_signals:
            additional = self._process_additional_signals(additional_signals)
            for key, value in additional.items():
                score += value
                breakdown[key] = value

        # Check for disqualification signals
        disqualifications = self._check_disqualification(
            intent_result, entities, additional_signals
        )

        # Ensure score is within bounds
        score = max(0, min(100, score))

        # Determine priority
        if disqualifications:
            priority = LeadPriority.DISQUALIFIED
        elif score >= self.HOT_THRESHOLD:
            priority = LeadPriority.HOT
            recommendations.append("Immediate follow-up recommended")
            recommendations.append("Route to sales team")
        elif score >= self.WARM_THRESHOLD:
            priority = LeadPriority.WARM
            recommendations.append("Schedule follow-up within 24 hours")
            recommendations.append("Continue qualification in chat")
        else:
            priority = LeadPriority.COLD
            recommendations.append("Add to nurture campaign")
            recommendations.append("Send relevant content")

        # Add specific recommendations based on signals
        recommendations.extend(self._get_recommendations(intent_result, entities))

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            intent_result, entities, conversation_turns
        )

        return LeadScore(
            score=score,
            priority=priority,
            score_breakdown=breakdown,
            confidence=confidence,
            signals=signals,
            disqualification_reasons=disqualifications,
            recommendations=recommendations,
        )

    def _score_intent(self, intent: Intent) -> int:
        """Get score for an intent."""
        intent_key = f"intent_{intent.value}"
        return self.rules.get(intent_key, 0)

    def _score_timeline(self, timeline: Timeline) -> int:
        """Get score for a timeline."""
        timeline_key = f"timeline_{timeline.value}"
        return self.rules.get(timeline_key, 0)

    def _score_entities(self, entities: ExtractedEntities) -> Dict[str, int]:
        """Score based on extracted entities."""
        scores: Dict[str, int] = {}

        if entities.models_mentioned:
            scores["model_mentioned"] = self.rules["model_mentioned"]

        if entities.budget_mentioned:
            scores["budget_mentioned"] = self.rules["budget_mentioned"]
            # Bonus for high budget
            if entities.budget_min and entities.budget_min >= 1500000:  # 15 lakh
                scores["budget_high"] = self.rules["budget_high"]

        if entities.trade_in_mentioned:
            scores["trade_in_mentioned"] = self.rules["trade_in_mentioned"]

        if entities.city or entities.pincode:
            scores["location_provided"] = self.rules["location_provided"]

        if entities.has_contact_info():
            scores["contact_provided"] = self.rules["contact_provided"]

        if entities.features_requested:
            scores["features_specified"] = self.rules["features_specified"]

        if entities.color_preference:
            scores["color_preference"] = self.rules["color_preference"]

        return scores

    def _process_additional_signals(
        self,
        signals: Dict[str, Any]
    ) -> Dict[str, int]:
        """Process additional signals into scores."""
        scores: Dict[str, int] = {}

        # Check for negative signals
        if signals.get("just_browsing"):
            scores["just_browsing"] = self.rules["just_browsing"]

        if signals.get("competitor_mentioned"):
            scores["competitor_mention"] = self.rules["competitor_mention"]

        if signals.get("price_objection"):
            scores["price_objection"] = self.rules["price_objection"]

        return scores

    def _check_disqualification(
        self,
        intent_result: IntentResult,
        entities: ExtractedEntities,
        additional_signals: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Check for disqualification signals."""
        reasons = []

        # Complaint without resolution intent
        if intent_result.primary_intent == Intent.COMPLAINT:
            reasons.append("Primary intent is complaint - route to support")

        # Explicit not interested signals
        if additional_signals:
            if additional_signals.get("not_interested"):
                reasons.append("Customer explicitly not interested")

            if additional_signals.get("spam_detected"):
                reasons.append("Spam or bot activity detected")

            if additional_signals.get("out_of_service_area"):
                reasons.append("Customer outside service area")

        return reasons

    def _calculate_confidence(
        self,
        intent_result: IntentResult,
        entities: ExtractedEntities,
        conversation_turns: int
    ) -> float:
        """Calculate confidence in the lead score."""
        confidence_factors = []

        # Intent confidence
        confidence_factors.append(intent_result.confidence)

        # Entity extraction confidence
        confidence_factors.append(entities.extraction_confidence)

        # Conversation length factor (more turns = more confident)
        turn_confidence = min(conversation_turns / 5, 1.0)
        confidence_factors.append(turn_confidence)

        # Contact info provides high confidence
        if entities.has_contact_info():
            confidence_factors.append(0.9)

        # Average all factors
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)

        return 0.5

    def _get_recommendations(
        self,
        intent_result: IntentResult,
        entities: ExtractedEntities
    ) -> List[str]:
        """Generate specific recommendations based on signals."""
        recommendations = []

        # Intent-based recommendations
        if intent_result.primary_intent == Intent.TEST_DRIVE:
            recommendations.append("Offer to schedule test drive")
            if entities.city:
                recommendations.append(f"Check test drive availability in {entities.city}")

        elif intent_result.primary_intent == Intent.FINANCE:
            recommendations.append("Share EMI calculator link")
            recommendations.append("Discuss financing options")

        elif intent_result.primary_intent == Intent.EXCHANGE:
            recommendations.append("Offer free car evaluation")
            recommendations.append("Share exchange bonus details")

        # Entity-based recommendations
        if entities.models_mentioned:
            models = ", ".join(entities.models_mentioned)
            recommendations.append(f"Focus discussion on: {models}")

        if entities.budget_mentioned and entities.budget_max:
            budget_lakh = entities.budget_max / 100000
            recommendations.append(f"Show vehicles under ₹{budget_lakh:.0f} lakh")

        # Missing information recommendations
        if not entities.has_contact_info():
            recommendations.append("Try to collect contact information")

        if not entities.budget_mentioned:
            recommendations.append("Understand budget requirements")

        if not entities.models_mentioned:
            recommendations.append("Identify preferred vehicle segment")

        return recommendations[:5]  # Limit to top 5

    def adjust_thresholds(self, hot: int = 70, warm: int = 50):
        """
        Adjust priority thresholds.

        Args:
            hot: Threshold for hot leads (default 70)
            warm: Threshold for warm leads (default 50)
        """
        self.HOT_THRESHOLD = hot
        self.WARM_THRESHOLD = warm

    def add_custom_rule(self, rule_name: str, score: int):
        """
        Add a custom scoring rule.

        Args:
            rule_name: Name of the rule
            score: Score value (positive or negative)
        """
        self.rules[rule_name] = score
