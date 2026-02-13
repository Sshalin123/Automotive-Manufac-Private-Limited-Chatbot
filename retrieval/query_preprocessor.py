"""
Query Preprocessor for AMPL Chatbot.

Normalizes, cleans, and expands user queries before embedding
to improve vector search quality for vague, messy, and
multilingual (Hinglish) queries.
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Preprocesses raw user queries for better embedding quality.

    Pipeline:
    1. Unicode normalization + lowercase
    2. Remove filler words / conversational noise
    3. Expand Hinglish (Roman Hindi) terms to English equivalents
    4. Expand common abbreviations
    """

    # Filler words to remove (low semantic value for embedding)
    FILLER_WORDS = {
        # English fillers
        "please", "plz", "pls", "kindly", "can you", "could you",
        "tell me about", "tell me", "i want to know", "i want to",
        "what is", "what are", "how about", "let me know",
        "just", "actually", "basically", "like", "umm", "hmm",
        "ok so", "okay so", "well",
        # Hindi fillers (Roman script)
        "kya hai", "bataiye", "batao", "zara", "thoda",
        "ji", "haan", "accha",
    }

    # Hinglish → English expansion map (augments, not replaces)
    HINGLISH_MAP = {
        # Vehicle terms
        "gaadi": "car vehicle",
        "gadi": "car vehicle",
        "car": "car",
        "bike": "bike motorcycle",
        "scooter": "scooter",
        # Actions
        "khareedna": "buy purchase",
        "kharidna": "buy purchase",
        "lena": "buy take",
        "dikhao": "show display",
        "dikha": "show",
        "chahiye": "want need",
        "chaiye": "want need",
        "book": "book reserve",
        "test": "test",
        # Price / Finance
        "kitna": "how much price cost",
        "kitne": "how much price cost",
        "kimat": "price cost",
        "keemat": "price cost",
        "rate": "rate price",
        "budget": "budget",
        "emi": "emi installment",
        "loan": "loan finance",
        "down payment": "down payment",
        "finance": "finance loan",
        # Features
        "sunroof": "sunroof",
        "automatic": "automatic transmission",
        "manual": "manual transmission",
        "diesel": "diesel fuel",
        "petrol": "petrol fuel",
        "cng": "cng fuel",
        "electric": "electric ev",
        "hybrid": "hybrid fuel",
        "mileage": "mileage fuel efficiency",
        "average": "mileage fuel efficiency",
        # Colors
        "lal": "red color",
        "neela": "blue color",
        "safed": "white color",
        "kala": "black color",
        # Service
        "service": "service maintenance",
        "servicing": "service maintenance",
        "repair": "repair service",
        "problem": "problem issue complaint",
        "kharab": "broken defective problem",
        # Comparison
        "compare": "compare comparison",
        "better": "better comparison",
        "acha": "good better",
        "accha": "good better",
        "best": "best top",
        "konsa": "which one",
        "kaun": "which one",
        # Time
        "abhi": "now immediate",
        "jaldi": "soon quick",
        "aaj": "today",
        "kal": "tomorrow",
        # Misc
        "naya": "new latest",
        "purana": "old used exchange",
        "exchange": "exchange trade-in",
        "insurance": "insurance",
        "warranty": "warranty guarantee",
        "offer": "offer discount deal",
        "discount": "discount offer",
    }

    # Abbreviation expansions
    ABBREVIATION_MAP = {
        "td": "test drive",
        "sr": "sunroof",
        "amt": "automatic manual transmission",
        "cvt": "continuously variable transmission",
        "abs": "anti-lock braking system",
        "ac": "air conditioning",
        "cng": "compressed natural gas",
        "ev": "electric vehicle",
        "suv": "sport utility vehicle suv",
        "mpv": "multi purpose vehicle mpv",
        "bs6": "bs6 emission",
        "bs vi": "bs6 emission",
        "rc": "registration certificate",
        "rto": "regional transport office",
        "dl": "driving licence",
    }

    def __init__(
        self,
        enable_hinglish: bool = True,
        enable_abbreviations: bool = True,
    ):
        self.enable_hinglish = enable_hinglish
        self.enable_abbreviations = enable_abbreviations

        # Pre-compile filler patterns (sorted long→short to match greedily)
        sorted_fillers = sorted(self.FILLER_WORDS, key=len, reverse=True)
        self._filler_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(f) for f in sorted_fillers) + r')\b',
            re.IGNORECASE,
        )

    def preprocess(self, raw_query: str) -> str:
        """
        Full preprocessing pipeline.

        Args:
            raw_query: Raw user message

        Returns:
            Preprocessed query optimized for embedding
        """
        if not raw_query or not raw_query.strip():
            return raw_query

        query = self._normalize(raw_query)
        query = self._remove_fillers(query)

        if self.enable_hinglish:
            query = self._expand_hinglish(query)

        if self.enable_abbreviations:
            query = self._expand_abbreviations(query)

        # Clean up extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        if query != raw_query.strip().lower():
            logger.debug(f"Query preprocessed: '{raw_query}' → '{query}'")

        return query

    def _normalize(self, text: str) -> str:
        """Unicode normalization, lowercase, strip."""
        text = unicodedata.normalize("NFC", text)
        text = text.lower().strip()
        # Remove excessive punctuation but keep hyphens in model names
        text = re.sub(r'[!?.,:;]+', ' ', text)
        return text

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words and conversational noise."""
        text = self._filler_pattern.sub('', text)
        return text

    def _expand_hinglish(self, text: str) -> str:
        """Expand Hinglish words by appending English equivalents."""
        words = text.split()
        expanded = []

        for word in words:
            expanded.append(word)
            expansion = self.HINGLISH_MAP.get(word)
            if expansion and expansion != word:
                # Append expansion terms (avoid duplicating the original word)
                for term in expansion.split():
                    if term != word and term not in expanded:
                        expanded.append(term)

        return ' '.join(expanded)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        words = text.split()
        expanded = []

        for word in words:
            expanded.append(word)
            expansion = self.ABBREVIATION_MAP.get(word)
            if expansion:
                for term in expansion.split():
                    if term != word and term not in expanded:
                        expanded.append(term)

        return ' '.join(expanded)
