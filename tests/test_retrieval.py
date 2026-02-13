"""Tests for retrieval components."""

import pytest
from retrieval.context_builder import ContextBuilder
from retrieval.reranker import Reranker


# ── Context Builder ───────────────────────────────────

class TestContextBuilder:
    def test_build_with_results(self):
        builder = ContextBuilder(max_tokens=500, max_chunks=3)
        results = [
            type("R", (), {"id": "1", "text": "The Nexon costs 10.5 lakhs.", "score": 0.9, "metadata": {"source": "inventory"}, "source": "inventory"})(),
            type("R", (), {"id": "2", "text": "EMI starts at 12,999/month.", "score": 0.85, "metadata": {"source": "faq"}, "source": "faq"})(),
        ]
        ctx = builder.build(results=results, query="Nexon price and EMI")
        assert ctx.text  # Non-empty context
        assert ctx.chunk_count <= 3

    def test_empty_results(self):
        builder = ContextBuilder(max_tokens=500, max_chunks=3)
        ctx = builder.build(results=[], query="test")
        assert ctx.chunk_count == 0


# ── Reranker ──────────────────────────────────────────

class TestReranker:
    def test_rule_based_rerank(self):
        reranker = Reranker()
        # Create mock SearchResult objects
        results = [
            type("SearchResult", (), {"id": "1", "text": "General information about cars.", "score": 0.8, "metadata": {}, "source": "", "text_preview": "General information about cars."})(),
            type("SearchResult", (), {"id": "2", "text": "The price of this vehicle is 15 lakhs with EMI options.", "score": 0.75, "metadata": {}, "source": "", "text_preview": "The price of this vehicle is 15 lakhs with EMI options."})(),
        ]
        reranked = reranker.rerank(query="What is the price?", results=results)
        assert len(reranked) == 2
        # The result mentioning "price" should be boosted
        assert reranked[0].original.id == "2"
