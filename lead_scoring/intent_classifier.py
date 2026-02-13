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
    BOOKING_CONFIRM = "booking_confirm"    # Booking/payment confirmation
    PAYMENT_CONFIRM = "payment_confirm"    # Payment receipt confirmation (Yes/No)
    SERVICE_REMINDER = "service_reminder"  # Service due reminder response
    FEEDBACK = "feedback"                  # Feedback / NPS response
    ESCALATION = "escalation"              # Escalation request
    DELIVERY_UPDATE = "delivery_update"    # Delivery status inquiry
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

    # Intent keywords for rule-based classification (includes Hinglish — Gap 6.2)
    INTENT_KEYWORDS = {
        Intent.BUY: [
            "buy", "purchase", "book", "booking", "order", "ready to buy",
            "want to buy", "looking to buy", "interested in buying",
            "finalize", "confirm order", "place order", "take delivery",
            "best price", "final price", "negotiate", "discount for buying",
            # Hinglish (Gap 6.2)
            "khareedna", "kharidna", "lena chahiye", "lena hai", "book karo",
            "book karni", "gaadi leni", "gaadi chahiye", "le lunga", "lena chahunga",
            "kab mil sakti", "ready hu", "order kardo",
        ],
        Intent.FINANCE: [
            "emi", "loan", "finance", "financing", "interest rate",
            "down payment", "monthly payment", "installment", "credit",
            "bank loan", "nbfc", "car loan", "auto loan", "tenure",
            "pre-approved", "loan eligibility", "emi calculator",
            # Hinglish
            "emi kitna", "emi kitni", "loan chahiye", "kist", "byaj",
            "byaj dar", "down payment kitna", "mahine ki kist",
            "loan milega", "finance karwana",
        ],
        Intent.TEST_DRIVE: [
            "test drive", "test-drive", "demo", "try the car",
            "drive the car", "experience", "feel of the car",
            "book test drive", "schedule test drive", "home test drive",
            "want to drive", "can i drive",
            # Hinglish
            "td book", "gaadi chalani hai", "gaadi chala ke dekh",
            "test drive kab", "ghar pe test drive", "demo dikhao",
            "chalake dekhna hai",
        ],
        Intent.SERVICE: [
            "service", "servicing", "maintenance", "repair", "fix",
            "service center", "workshop", "oil change", "tyre",
            "brake", "ac service", "free service", "paid service",
            "service cost", "service schedule", "recall",
            # Hinglish
            "service kab", "service karwani", "repair karwana",
            "gaadi kharab", "service center kahan", "free service kab",
            "oil change karwana", "tyre badalna",
        ],
        Intent.EXCHANGE: [
            "exchange", "trade-in", "trade in", "old car", "sell my car",
            "exchange value", "exchange bonus", "replace my car",
            "part exchange", "exchange offer",
            # Hinglish
            "purani gaadi dena", "purani gaadi exchange", "badli karni",
            "exchange mein dena", "purani car ki value",
        ],
        Intent.INSURANCE: [
            "insurance", "insure", "coverage", "claim", "ncb",
            "zero dep", "depreciation", "premium", "policy",
            "renew insurance", "transfer insurance",
            # Hinglish
            "bima", "insurance kitna", "claim kaise", "insurance renew",
            "policy transfer",
        ],
        Intent.COMPLAINT: [
            "complaint", "issue", "problem", "not working", "defect",
            "unhappy", "dissatisfied", "disappointed", "bad experience",
            "escalate", "manager", "consumer court",
            # Hinglish
            "shikayat", "dikkat", "pareshani", "kharab experience",
            "manager se baat", "complaint karna", "theek nahi",
        ],
        Intent.BOOKING_CONFIRM: [
            "booking confirmed", "confirm booking", "booking done",
            "booked", "booking status", "my booking", "booking details",
            "advance paid", "token paid",
            # Hinglish
            "booking ho gayi", "booking confirm hai", "advance diya",
            "token de diya", "booking ka status",
        ],
        Intent.PAYMENT_CONFIRM: [
            "payment received", "amount received", "yes received",
            "no not received", "payment mismatch", "wrong amount",
            "receipt", "payment status", "payment confirmation",
            # Hinglish
            "payment mil gaya", "paisa mila", "rashi nahi mili",
            "galat amount", "payment ka status",
        ],
        Intent.SERVICE_REMINDER: [
            "service due", "service reminder", "when is my service",
            "next service", "free service", "service booking",
            "book service", "schedule service",
            # Hinglish
            "service kab hai", "agla service", "free service kab",
            "service book karo", "service yaad dilao",
        ],
        Intent.FEEDBACK: [
            "feedback", "rating", "review", "experience was",
            "satisfied", "happy with", "recommend", "nps",
            "poor", "fair", "very good", "excellent",
            # Hinglish
            "kaisa laga", "experience kaisa tha", "accha laga",
            "bura laga", "rating dena", "review dena",
        ],
        Intent.ESCALATION: [
            "escalate", "escalation", "speak to manager",
            "higher authority", "not resolved", "unresolved",
            "need help urgently", "still waiting",
            # Hinglish
            "manager se baat karo", "upar wale se baat",
            "abhi tak nahi hua", "jaldi karo", "bahut wait ho gaya",
        ],
        Intent.DELIVERY_UPDATE: [
            "delivery status", "when will i get", "delivery date",
            "car ready", "delivery delay", "dispatch status",
            "allotment status", "when delivery",
            # Hinglish
            "delivery kab", "gaadi kab milegi", "kab tak aayegi",
            "allotment hua", "dispatch hua", "gaadi ready hai",
        ],
    }

    # Timeline keywords (includes Hinglish — Gap 6.2)
    TIMELINE_KEYWORDS = {
        Timeline.IMMEDIATE: [
            "today", "tomorrow", "this week", "asap", "immediately",
            "urgent", "right away", "as soon as", "next day",
            # Hinglish
            "aaj", "kal", "abhi", "turant", "jaldi", "fatafat",
        ],
        Timeline.THIS_MONTH: [
            "this month", "within a month", "next week", "soon",
            "in a few days", "this weekend",
            # Hinglish
            "is mahine", "agla hafta", "kuch din mein", "jaldi hi",
        ],
        Timeline.THIS_QUARTER: [
            "next month", "in 2 months", "next quarter", "few months",
            "planning", "thinking about",
            # Hinglish
            "agla mahina", "do mahine", "soch raha", "plan hai",
        ],
        Timeline.EXPLORING: [
            "just looking", "exploring", "researching", "comparing",
            "not sure", "sometime", "in the future", "maybe",
            "just checking", "curious",
            # Hinglish
            "bas dekh raha", "compare kar raha", "pata nahi",
            "baad mein", "kabhi", "sochu", "dekhte hain",
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

    def __init__(self, llm_client: Optional[Any] = None, llm_model_id: Optional[str] = None):
        """
        Initialize the intent classifier.

        Args:
            llm_client: Optional LLM client for advanced classification
            llm_model_id: Model ID for LLM calls
        """
        self.llm_client = llm_client
        self._llm_model_id = llm_model_id

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
        system = "You are an automotive dealership intent classification engine. Return only valid JSON."

        try:
            response_text = None

            # Detect client type and call accordingly
            client_type = type(self.llm_client).__name__

            if client_type == "OpenAI":
                # OpenAI client
                response = self.llm_client.chat.completions.create(
                    model=self._llm_model_id or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=256,
                    temperature=0.1,
                )
                response_text = response.choices[0].message.content

            else:
                # Bedrock client
                import json as _json
                body = _json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 256,
                    "temperature": 0.1,
                    "system": system,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                })
                response = self.llm_client.invoke_model(
                    modelId=self._llm_model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = _json.loads(response["body"].read())
                if "content" in response_body and response_body["content"]:
                    response_text = response_body["content"][0]["text"]

            if response_text:
                result = self._parse_llm_response(response_text)
                if result:
                    logger.debug(f"LLM classified intent: {result.primary_intent.value} (conf={result.confidence})")
                return result

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
  "primary_intent": "BUY | FINANCE | TEST_DRIVE | SERVICE | EXCHANGE | INSURANCE | COMPLAINT | BOOKING_CONFIRM | PAYMENT_CONFIRM | SERVICE_REMINDER | FEEDBACK | ESCALATION | DELIVERY_UPDATE | INFO",
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
- BOOKING_CONFIRM: Booking or payment confirmation
- PAYMENT_CONFIRM: Payment receipt confirmation (Yes/No response)
- SERVICE_REMINDER: Service due or service scheduling
- FEEDBACK: Feedback, rating, or NPS response
- ESCALATION: Requesting escalation to higher authority
- DELIVERY_UPDATE: Delivery status or allotment inquiry
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
            Intent.BOOKING_CONFIRM: "Customer confirming or inquiring about booking",
            Intent.PAYMENT_CONFIRM: "Customer confirming payment receipt",
            Intent.SERVICE_REMINDER: "Customer responding to service reminder",
            Intent.FEEDBACK: "Customer providing feedback or rating",
            Intent.ESCALATION: "Customer requesting escalation",
            Intent.DELIVERY_UPDATE: "Customer inquiring about delivery status",
            Intent.INFO: "Customer is seeking general information",
            Intent.UNKNOWN: "Unable to determine customer intent",
        }
        return descriptions.get(intent, "Unknown intent")
