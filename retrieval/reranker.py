"""
Reranker for AMPL Chatbot.

Reranks search results for improved relevance.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .pinecone_client import SearchResult

logger = logging.getLogger(__name__)


class RerankerType(Enum):
    """Supported reranker types."""
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"


@dataclass
class RerankedResult:
    """Reranked search result."""
    original: SearchResult
    rerank_score: float
    combined_score: float
    boosted: bool = False
    boost_reason: Optional[str] = None


class Reranker:
    """
    Reranks search results for improved relevance.

    Supports:
    - Rule-based reranking (keyword boosting, recency)
    - Cross-encoder reranking (with external model)
    - LLM-based reranking
    """

    # Keyword boost factors
    BOOST_KEYWORDS = {
        "price": 1.2,
        "cost": 1.2,
        "emi": 1.3,
        "loan": 1.2,
        "finance": 1.2,
        "test drive": 1.3,
        "booking": 1.3,
        "available": 1.1,
        "feature": 1.1,
        "safety": 1.1,
        "mileage": 1.1,
        "warranty": 1.2,
        "service": 1.1,
        "exchange": 1.2,
    }

    def __init__(
        self,
        reranker_type: RerankerType = RerankerType.RULE_BASED,
        combine_weight: float = 0.5,
        cross_encoder_model: Optional[str] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the reranker.

        Args:
            reranker_type: Type of reranking to use
            combine_weight: Weight for combining original and rerank scores
            cross_encoder_model: Model name for cross-encoder reranking
            llm_client: LLM client for LLM-based reranking
        """
        self.reranker_type = reranker_type
        self.combine_weight = combine_weight
        self.cross_encoder_model = cross_encoder_model
        self.llm_client = llm_client

        self._cross_encoder = None
        if reranker_type == RerankerType.CROSS_ENCODER and cross_encoder_model:
            self._initialize_cross_encoder()

    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
            logger.info(f"Cross-encoder initialized: {self.cross_encoder_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to rule-based")
            self.reranker_type = RerankerType.RULE_BASED
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.reranker_type = RerankerType.RULE_BASED

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """
        Rerank search results.

        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        if self.reranker_type == RerankerType.CROSS_ENCODER:
            reranked = self._rerank_cross_encoder(query, results)
        elif self.reranker_type == RerankerType.LLM_BASED:
            reranked = self._rerank_llm(query, results)
        else:
            reranked = self._rerank_rule_based(query, results)

        # Sort by combined score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _rerank_rule_based(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[RerankedResult]:
        """Rule-based reranking with keyword boosting."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        reranked = []

        for result in results:
            boost = 1.0
            boost_reasons = []

            text_lower = result.text.lower()
            metadata = result.metadata

            # Keyword matching boost
            for keyword, boost_factor in self.BOOST_KEYWORDS.items():
                if keyword in query_lower and keyword in text_lower:
                    boost *= boost_factor
                    boost_reasons.append(f"keyword:{keyword}")

            # Query word overlap boost
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            overlap_ratio = overlap / len(query_words) if query_words else 0

            if overlap_ratio > 0.5:
                boost *= (1 + overlap_ratio * 0.3)
                boost_reasons.append(f"overlap:{overlap_ratio:.2f}")

            # Document type priority
            doc_type = metadata.get("document_type", "")
            if doc_type == "faq":
                boost *= 1.2
                boost_reasons.append("type:faq")
            elif doc_type == "inventory":
                boost *= 1.1
                boost_reasons.append("type:inventory")

            # Recency boost (if timestamp available)
            ingested_at = metadata.get("ingested_at")
            if ingested_at:
                # Simple recency check - newer is better
                boost *= 1.05
                boost_reasons.append("recent")

            # Calculate scores
            rerank_score = result.score * boost
            combined_score = (
                self.combine_weight * result.score +
                (1 - self.combine_weight) * rerank_score
            )

            reranked.append(RerankedResult(
                original=result,
                rerank_score=rerank_score,
                combined_score=combined_score,
                boosted=boost > 1.0,
                boost_reason=", ".join(boost_reasons) if boost_reasons else None
            ))

        return reranked

    def _rerank_cross_encoder(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[RerankedResult]:
        """Cross-encoder based reranking."""
        if not self._cross_encoder:
            return self._rerank_rule_based(query, results)

        try:
            # Prepare pairs
            pairs = [(query, result.text) for result in results]

            # Get cross-encoder scores
            scores = self._cross_encoder.predict(pairs)

            reranked = []
            for result, ce_score in zip(results, scores):
                # Normalize cross-encoder score to 0-1
                normalized_ce = (ce_score + 1) / 2  # Assuming scores in [-1, 1]

                combined_score = (
                    self.combine_weight * result.score +
                    (1 - self.combine_weight) * normalized_ce
                )

                reranked.append(RerankedResult(
                    original=result,
                    rerank_score=normalized_ce,
                    combined_score=combined_score,
                    boosted=normalized_ce > result.score
                ))

            return reranked

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return self._rerank_rule_based(query, results)

    def _rerank_llm(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[RerankedResult]:
        """LLM-based reranking."""
        if not self.llm_client:
            return self._rerank_rule_based(query, results)

        # For now, fall back to rule-based
        # LLM reranking would involve sending query + results to LLM
        # and getting relevance scores back
        logger.warning("LLM reranking not fully implemented, using rule-based")
        return self._rerank_rule_based(query, results)

    def add_boost_keyword(self, keyword: str, factor: float):
        """
        Add a custom boost keyword.

        Args:
            keyword: Keyword to boost
            factor: Boost factor (> 1.0 to boost, < 1.0 to demote)
        """
        self.BOOST_KEYWORDS[keyword.lower()] = factor

    def remove_boost_keyword(self, keyword: str):
        """Remove a boost keyword."""
        self.BOOST_KEYWORDS.pop(keyword.lower(), None)
