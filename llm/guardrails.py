"""
Response Verification & Guardrails for AMPL Chatbot (Gap 9.2).

Post-LLM safety checks to prevent hallucinations, unauthorized pricing,
competitor promotion, and harmful content.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of response verification."""
    passed: bool = True
    flags: List[str] = field(default_factory=list)
    sanitized_response: str = ""
    original_response: str = ""


class ResponseVerifier:
    """
    Verifies LLM responses before returning to user.

    Checks:
    1. Unauthorized pricing (fabricated price numbers)
    2. Competitor promotion
    3. Hallucinated models (mentions models not in context)
    4. Harmful / inappropriate content
    """

    # Competitor brands to flag if promoted
    COMPETITOR_BRANDS = {
        "hyundai", "tata", "mahindra", "kia", "toyota", "honda",
        "mg", "skoda", "volkswagen", "renault", "nissan", "jeep",
        "citroen", "bmw", "mercedes", "audi", "volvo",
    }

    # Known Maruti/Suzuki models (primary brand)
    KNOWN_MODELS = {
        "alto", "s-presso", "wagon r", "wagonr", "celerio", "swift",
        "dzire", "baleno", "ignis", "ciaz", "ertiga", "xl6",
        "brezza", "vitara brezza", "grand vitara", "fronx", "jimny",
        "invicto", "eeco", "super carry",
    }

    # Patterns that suggest fabricated pricing
    PRICE_PATTERN = re.compile(
        r'(?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d{2})?(?:\s*(?:lakh|lac|cr|crore))?',
        re.IGNORECASE,
    )

    # Basic harmful content patterns
    HARMFUL_PATTERNS = [
        re.compile(r'\b(?:kill|murder|harm|weapon|bomb|terrorist)\b', re.IGNORECASE),
        re.compile(r'\b(?:hack|exploit|bypass security)\b', re.IGNORECASE),
    ]

    def __init__(self, brand_name: str = "Maruti Suzuki"):
        self.brand_name = brand_name

    def verify(
        self,
        response: str,
        context_text: Optional[str] = None,
        query: Optional[str] = None,
    ) -> VerificationResult:
        """
        Run all verification checks on LLM response.

        Args:
            response: LLM-generated response text
            context_text: RAG context chunks that were provided to LLM
            query: Original user query

        Returns:
            VerificationResult with flags and sanitized response
        """
        result = VerificationResult(
            original_response=response,
            sanitized_response=response,
        )

        self._check_unauthorized_pricing(result, context_text)
        self._check_competitor_promotion(result)
        self._check_hallucinated_models(result, context_text)
        self._check_harmful_content(result)

        result.passed = len(result.flags) == 0

        if not result.passed:
            logger.warning(f"Response verification flags: {result.flags}")

        return result

    def _check_unauthorized_pricing(
        self, result: VerificationResult, context_text: Optional[str]
    ):
        """Flag if response contains prices not found in context."""
        prices_in_response = self.PRICE_PATTERN.findall(result.sanitized_response)
        if not prices_in_response:
            return

        if not context_text:
            # No context provided but response has prices — flag
            result.flags.append("pricing_without_context")
            # Append disclaimer
            result.sanitized_response += (
                "\n\n*Please verify pricing with your nearest dealership as "
                "prices may vary by location and variant.*"
            )
            return

        # Check if prices appear in context
        context_lower = context_text.lower()
        for price in prices_in_response:
            price_clean = price.strip().lower()
            if price_clean not in context_lower:
                result.flags.append("unverified_pricing")
                result.sanitized_response += (
                    "\n\n*Prices mentioned are indicative. Please confirm "
                    "with your nearest dealership for exact pricing.*"
                )
                break

    def _check_competitor_promotion(self, result: VerificationResult):
        """Flag if response promotes competitor brands."""
        response_lower = result.sanitized_response.lower()
        for competitor in self.COMPETITOR_BRANDS:
            # Check for promotional language near competitor mentions
            if competitor in response_lower:
                promo_patterns = [
                    rf'{competitor}\s+(?:is|are)\s+(?:better|superior|best)',
                    rf'(?:recommend|suggest|try)\s+{competitor}',
                    rf'{competitor}\s+(?:offers?|provides?)\s+(?:more|better)',
                ]
                for pattern in promo_patterns:
                    if re.search(pattern, response_lower):
                        result.flags.append(f"competitor_promotion:{competitor}")
                        # Remove the promotional sentence
                        break

    def _check_hallucinated_models(
        self, result: VerificationResult, context_text: Optional[str]
    ):
        """Flag if response mentions vehicle models not in context."""
        if not context_text:
            return

        response_lower = result.sanitized_response.lower()
        context_lower = context_text.lower()

        for model in self.KNOWN_MODELS:
            if model in response_lower and model not in context_lower:
                # Model mentioned in response but not in provided context
                result.flags.append(f"model_not_in_context:{model}")

    def _check_harmful_content(self, result: VerificationResult):
        """Flag harmful or inappropriate content."""
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(result.sanitized_response):
                result.flags.append("harmful_content")
                result.sanitized_response = (
                    "I apologize, but I can only assist with automotive-related queries. "
                    "How can I help you with your vehicle needs?"
                )
                break
