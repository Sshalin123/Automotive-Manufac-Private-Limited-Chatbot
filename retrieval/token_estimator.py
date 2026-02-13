"""
Token Estimator for AMPL Chatbot.

Provides accurate token counting for OpenAI (tiktoken) and
language-aware heuristic for Bedrock/Claude models.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Hindi Unicode range
_HINDI_RANGE = re.compile(r'[\u0900-\u097F]')


class TokenEstimator:
    """
    Estimates token count for text, with provider-aware strategies.

    - OpenAI: uses tiktoken (cl100k_base) for exact counting
    - Bedrock/Other: language-aware heuristic (Hindi ~2 chars/token, English ~4 chars/token)
    """

    def __init__(self, provider: str = "bedrock"):
        """
        Args:
            provider: "openai" or "bedrock" (default)
        """
        self.provider = provider.lower()
        self._tiktoken_enc = None

        if self.provider == "openai":
            try:
                import tiktoken
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                logger.info("TokenEstimator: using tiktoken (cl100k_base)")
            except ImportError:
                logger.warning("tiktoken not installed, falling back to heuristic")

    def estimate(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        if self._tiktoken_enc:
            return len(self._tiktoken_enc.encode(text))

        return self._heuristic_estimate(text)

    def _heuristic_estimate(self, text: str) -> int:
        """Language-aware heuristic token estimation."""
        if not text:
            return 0

        hindi_chars = len(_HINDI_RANGE.findall(text))
        total_chars = len(text)
        non_hindi_chars = total_chars - hindi_chars

        # Hindi: ~2 chars per token, English/Latin: ~4 chars per token
        hindi_tokens = hindi_chars / 2 if hindi_chars else 0
        english_tokens = non_hindi_chars / 4 if non_hindi_chars else 0

        return max(1, int(hindi_tokens + english_tokens))
