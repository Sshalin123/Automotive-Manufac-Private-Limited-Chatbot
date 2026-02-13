"""
Intent Classification for AMPL Chatbot.

Uses LLM-based classification to identify customer intent from messages.
"""

import json
import logging
import re
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Customer intent categories."""
    BUY = "buy"                      # Ready to purchase
    FINANCE = "finance"              # Interested in financing/EMI
    TEST_DRIVE = "test_drive"        # Wants to schedule test drive
    SERVICE = "service"              # Service/maintenance inquiry
    INFO = "info"                    # General information seeking
    EXCHANGE = "exchange"            # Car exchange inquiry
    INSURANCE = "insurance"          # Insurance related
    COMPLAINT = "complaint"          # Complaint or issue
    UNKNOWN = "unknown"              # Unable to determine


class Timeline(Enum):
    """Purchase timeline urgency."""
    IMMEDIATE = "immediate"          # Within a week
    THIS_MONTH = "this_month"        # Within a month
    THIS_QUARTER = "this_quarter"    # Within 3 months
    EXPLORING = "exploring"          # Just exploring, no timeline
    UNKNOWN = "unknown"


class ContactWillingness(Enum):
    """Customer's willingness to be contacted."""
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary_intent: Intent
    secondary_intents: List[Intent] = field(default_factory=list)
    confidence: float = 0.0
    timeline: Timeline = Timeline.UNKNOWN
    contact_willingness: ContactWillingness = ContactWillingness.UNKNOWN
    raw_analysis: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class IntentClassifier:
    """
    Classifies customer intent from messages using LLM.

    This classifier uses both rule-based pattern matching for quick
    classification and LLM analysis for complex cases.
    """

    # Intent keywords for rule-based classification
    INTENT_KEYWORDS = {
        Intent.BUY: [
            "buy", "purchase", "book", "booking", "order", "ready to buy",
            "want to buy", "looking to buy", "interested in buying",
            "finalize", "confirm order", "place order", "take delivery",
            "best price", "final price", "negotiate", "discount for buying",
        ],
        Intent.FINANCE: [
            "emi", "loan", "finance", "financing", "interest rate",
            "down payment", "monthly payment", "installment", "credit",
            "bank loan", "nbfc", "car loan", "auto loan", "tenure",
            "pre-approved", "loan eligibility", "emi calculator",
        ],
        Intent.TEST_DRIVE: [
            "test drive", "test-drive", "demo", "try the car",
            "drive the car", "experience", "feel of the car",
            "book test drive", "schedule test drive", "home test drive",
            "want to drive", "can i drive",
        ],
        Intent.SERVICE: [
            "service", "servicing", "maintenance", "repair", "fix",
            "service center", "workshop", "oil change", "tyre",
            "brake", "ac service", "free service", "paid service",
            "service cost", "service schedule", "recall",
        ],
        Intent.EXCHANGE: [
            "exchange", "trade-in", "trade in", "old car", "sell my car",
            "exchange value", "exchange bonus", "replace my car",
            "part exchange", "exchange offer",
        ],
        Intent.INSURANCE: [
            "insurance", "insure", "coverage", "claim", "ncb",
            "zero dep", "depreciation", "premium", "policy",
            "renew insurance", "transfer insurance",
        ],
        Intent.COMPLAINT: [
            "complaint", "issue", "problem", "not working", "defect",
            "unhappy", "dissatisfied", "disappointed", "bad experience",
            "escalate", "manager", "consumer court",
        ],
    }

    # Timeline keywords
    TIMELINE_KEYWORDS = {
        Timeline.IMMEDIATE: [
            "today", "tomorrow", "this week", "asap", "immediately",
            "urgent", "right away", "as soon as", "next day",
        ],
        Timeline.THIS_MONTH: [
            "this month", "within a month", "next week", "soon",
            "in a few days", "this weekend",
        ],
        Timeline.THIS_QUARTER: [
            "next month", "in 2 months", "next quarter", "few months",
            "planning", "thinking about",
        ],
        Timeline.EXPLORING: [
            "just looking", "exploring", "researching", "comparing",
            "not sure", "sometime", "in the future", "maybe",
            "just checking", "curious",
        ],
    }

    # Contact willingness keywords
    CONTACT_YES = [
        "call me", "contact me", "reach out", "give me a call",
        "my number is", "my phone", "my email", "yes please contact",
        "send me", "whatsapp",
    ]

    CONTACT_NO = [
        "don't call", "do not call", "no calls", "don't contact",
        "i'll call", "i will contact", "not interested in calls",
        "just information", "no salesperson",
    ]

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the intent classifier.

        Args:
            llm_client: Optional LLM client for advanced classification
        """
        self.llm_client = llm_client

    def classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = False
    ) -> IntentResult:
        """
        Classify the intent of a customer message.

        Args:
            message: The customer message to classify
            conversation_history: Optional previous messages for context
            use_llm: Whether to use LLM for classification

        Returns:
            IntentResult with classification details
        """
        message_lower = message.lower()

        # Rule-based classification first
        primary_intent, secondary_intents, confidence = self._rule_based_classify(message_lower)

        # Determine timeline
        timeline = self._extract_timeline(message_lower)

        # Determine contact willingness
        contact_willingness = self._extract_contact_willingness(message_lower)

        # Use LLM for low confidence or complex cases
        if use_llm and self.llm_client and confidence < 0.7:
            llm_result = self._llm_classify(message, conversation_history)
            if llm_result:
                return llm_result

        return IntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            timeline=timeline,
            contact_willingness=contact_willingness,
        )

    def _rule_based_classify(
        self,
        message_lower: str
    ) -> Tuple[Intent, List[Intent], float]:
        """
        Perform rule-based intent classification.

        Returns:
            Tuple of (primary_intent, secondary_intents, confidence)
        """
        intent_scores: Dict[Intent, float] = {}

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in message_lower:
                    # Longer keywords get higher scores
                    keyword_score = len(keyword.split()) * 0.2
                    score += keyword_score

                    # Exact phrase match gets bonus
                    if re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                        score += 0.1

            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            return Intent.INFO, [], 0.3

        # Sort by score
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        # Normalize confidence (cap at 1.0)
        confidence = min(primary_score / 2.0, 1.0)

        # Get secondary intents (with significant scores)
        secondary_intents = [
            intent for intent, score in sorted_intents[1:4]
            if score > primary_score * 0.3
        ]

        return primary_intent, secondary_intents, confidence

    def _extract_timeline(self, message_lower: str) -> Timeline:
        """Extract purchase timeline from message."""
        for timeline, keywords in self.TIMELINE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return timeline
        return Timeline.UNKNOWN

    def _extract_contact_willingness(self, message_lower: str) -> ContactWillingness:
        """Extract contact willingness from message."""
        for keyword in self.CONTACT_YES:
            if keyword in message_lower:
                return ContactWillingness.YES

        for keyword in self.CONTACT_NO:
            if keyword in message_lower:
                return ContactWillingness.NO

        # Check for phone/email patterns
        phone_pattern = r'\b\d{10}\b|\b\d{5}[\s-]?\d{5}\b'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        if re.search(phone_pattern, message_lower) or re.search(email_pattern, message_lower):
            return ContactWillingness.YES

        return ContactWillingness.UNKNOWN

    def _llm_classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[IntentResult]:
        """
        Use LLM for intent classification.

        Args:
            message: Customer message
            conversation_history: Previous messages

        Returns:
            IntentResult or None if LLM call fails
        """
        if not self.llm_client:
            return None

        prompt = self._build_classification_prompt(message, conversation_history)

        try:
            # This would call the LLM client
            # response = self.llm_client.generate(prompt)
            # result = self._parse_llm_response(response)
            # return result
            pass
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None

        return None

    def _build_classification_prompt(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build the prompt for LLM classification."""
        prompt = """Analyze the customer message and extract the following information.
Return a JSON object with these fields:

{
  "primary_intent": "BUY | FINANCE | TEST_DRIVE | SERVICE | EXCHANGE | INSURANCE | COMPLAINT | INFO",
  "secondary_intents": ["...", "..."],
  "confidence": 0.0-1.0,
  "timeline": "IMMEDIATE | THIS_MONTH | THIS_QUARTER | EXPLORING | UNKNOWN",
  "contact_willingness": "YES | NO | UNKNOWN",
  "reasoning": "Brief explanation"
}

Intent definitions:
- BUY: Ready to purchase or discussing purchase
- FINANCE: Interested in EMI, loans, financing options
- TEST_DRIVE: Wants to test drive or demo the vehicle
- SERVICE: Service, maintenance, repair inquiries
- EXCHANGE: Wants to exchange/trade-in old vehicle
- INSURANCE: Insurance related queries
- COMPLAINT: Complaints or issues
- INFO: General information seeking

"""
        if conversation_history:
            prompt += "Conversation context:\n"
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role}: {content}\n"

        prompt += f"\nCustomer message: {message}\n\nJSON response:"

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[IntentResult]:
        """Parse LLM response into IntentResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            primary_intent = Intent[data.get("primary_intent", "INFO").upper()]

            secondary_intents = [
                Intent[i.upper()]
                for i in data.get("secondary_intents", [])
                if i.upper() in Intent.__members__
            ]

            timeline = Timeline[data.get("timeline", "UNKNOWN").upper()]
            contact = ContactWillingness[data.get("contact_willingness", "UNKNOWN").upper()]

            return IntentResult(
                primary_intent=primary_intent,
                secondary_intents=secondary_intents,
                confidence=float(data.get("confidence", 0.5)),
                timeline=timeline,
                contact_willingness=contact,
                raw_analysis=data,
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None

    def get_intent_description(self, intent: Intent) -> str:
        """Get human-readable description of an intent."""
        descriptions = {
            Intent.BUY: "Customer is ready to purchase a vehicle",
            Intent.FINANCE: "Customer is interested in financing/EMI options",
            Intent.TEST_DRIVE: "Customer wants to schedule a test drive",
            Intent.SERVICE: "Customer has a service or maintenance inquiry",
            Intent.EXCHANGE: "Customer wants to exchange their old vehicle",
            Intent.INSURANCE: "Customer has insurance related queries",
            Intent.COMPLAINT: "Customer has a complaint or issue",
            Intent.INFO: "Customer is seeking general information",
            Intent.UNKNOWN: "Unable to determine customer intent",
        }
        return descriptions.get(intent, "Unknown intent")
