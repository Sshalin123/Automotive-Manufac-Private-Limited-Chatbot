"""
Context Builder for AMPL Chatbot.

Assembles retrieved chunks into context for LLM consumption.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .pinecone_client import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source citation information."""
    id: str
    source: str
    chunk_index: int
    score: float
    text_preview: str
    document_type: Optional[str] = None


@dataclass
class Context:
    """Assembled context for LLM."""
    text: str
    sources: List[Source] = field(default_factory=list)
    total_tokens: int = 0
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sources": [
                {
                    "id": s.id,
                    "source": s.source,
                    "chunk_index": s.chunk_index,
                    "score": s.score,
                    "text_preview": s.text_preview,
                    "document_type": s.document_type,
                }
                for s in self.sources
            ],
            "total_tokens": self.total_tokens,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
        }


class ContextBuilder:
    """
    Builds context from retrieved search results.

    Features:
    - Deduplication of similar chunks
    - Token counting
    - Source attribution
    - Priority-based ordering
    """

    # Document type priorities (higher = more important)
    TYPE_PRIORITY = {
        "faq": 5,      # FAQs are most authoritative
        "inventory": 4, # Vehicle specs are important
        "booking": 4,  # Booking details are high priority
        "insurance": 3,
        "service": 3,  # Service info
        "delivery": 3, # Delivery info
        "sales": 2,
        "general": 1,
    }

    def __init__(
        self,
        max_tokens: int = 4000,
        max_chunks: int = 10,
        include_sources: bool = True,
        deduplicate: bool = True,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize the context builder.

        Args:
            max_tokens: Maximum tokens in context
            max_chunks: Maximum number of chunks
            include_sources: Include source citations
            deduplicate: Remove duplicate/similar chunks
            similarity_threshold: Threshold for deduplication
        """
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.include_sources = include_sources
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold

    def build(
        self,
        results: List[SearchResult],
        query: Optional[str] = None,
        prioritize_types: Optional[List[str]] = None
    ) -> Context:
        """
        Build context from search results.

        Args:
            results: Search results from Pinecone
            query: Original query (for metadata)
            prioritize_types: Document types to prioritize

        Returns:
            Assembled Context object
        """
        if not results:
            return Context(
                text="No relevant information found.",
                sources=[],
                chunk_count=0,
                metadata={"query": query} if query else {}
            )

        # Deduplicate if enabled
        if self.deduplicate:
            results = self._deduplicate(results)

        # Sort by priority and score
        results = self._sort_results(results, prioritize_types)

        # Build context text
        context_parts = []
        sources = []
        total_tokens = 0

        for i, result in enumerate(results[:self.max_chunks]):
            text = result.text.strip()
            if not text:
                continue

            # Estimate tokens (rough: 1 token ≈ 4 characters)
            text_tokens = len(text) // 4

            # Check if adding this would exceed limit
            if total_tokens + text_tokens > self.max_tokens:
                # Try to add truncated version
                remaining_tokens = self.max_tokens - total_tokens
                if remaining_tokens > 100:
                    text = text[:remaining_tokens * 4]
                    text_tokens = remaining_tokens
                else:
                    break

            # Add to context
            source_label = f"[Source {i + 1}]"
            context_parts.append(f"{source_label}: {text}")
            total_tokens += text_tokens

            # Create source citation
            if self.include_sources:
                sources.append(Source(
                    id=result.id,
                    source=result.source or result.metadata.get("source", "Unknown"),
                    chunk_index=result.metadata.get("chunk_index", 0),
                    score=round(result.score, 4),
                    text_preview=text[:200] + "..." if len(text) > 200 else text,
                    document_type=result.metadata.get("document_type"),
                ))

        context_text = "\n\n".join(context_parts)

        return Context(
            text=context_text,
            sources=sources,
            total_tokens=total_tokens,
            chunk_count=len(context_parts),
            metadata={
                "query": query,
                "original_results": len(results),
                "included_results": len(context_parts),
            }
        )

    def _deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or highly similar chunks."""
        if len(results) <= 1:
            return results

        deduplicated = [results[0]]

        for result in results[1:]:
            is_duplicate = False

            for existing in deduplicated:
                # Check if texts are too similar
                if self._is_similar(result.text, existing.text):
                    is_duplicate = True
                    break

                # Check if same source and adjacent chunks
                if (result.source == existing.source and
                    abs(result.metadata.get("chunk_index", 0) -
                        existing.metadata.get("chunk_index", 0)) <= 1):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(result)

        logger.debug(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        return deduplicated

    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are highly similar."""
        # Simple overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        smaller = min(len(words1), len(words2))

        similarity = overlap / smaller if smaller > 0 else 0
        return similarity > self.similarity_threshold

    def _sort_results(
        self,
        results: List[SearchResult],
        prioritize_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Sort results by priority and score."""
        def get_priority(result: SearchResult) -> tuple:
            doc_type = result.metadata.get("document_type", "general")

            # Type priority
            type_priority = self.TYPE_PRIORITY.get(doc_type, 0)

            # Check if in prioritized types
            if prioritize_types and doc_type in prioritize_types:
                type_priority += 10

            return (type_priority, result.score)

        return sorted(results, key=get_priority, reverse=True)

    def format_for_prompt(self, context: Context) -> str:
        """
        Format context for LLM prompt.

        Args:
            context: Context object

        Returns:
            Formatted string for prompt
        """
        if not context.text:
            return ""

        return f"""<context>
{context.text}
</context>

Sources referenced: {len(context.sources)} documents
"""

    def get_citation_text(self, context: Context) -> str:
        """
        Generate citation text for response.

        Args:
            context: Context object

        Returns:
            Formatted citation string
        """
        if not context.sources:
            return ""

        citations = []
        for i, source in enumerate(context.sources[:5], 1):
            source_name = source.source.split("/")[-1] if "/" in source.source else source.source
            citations.append(f"[{i}] {source_name}")

        return "\n".join(citations)
