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
        results = [
            {"id": "1", "text": "General information about cars.", "score": 0.8},
            {"id": "2", "text": "The price of this vehicle is 15 lakhs with EMI options.", "score": 0.75},
        ]
        reranked = reranker.rerank(results, query="What is the price?", method="rule_based")
        assert len(reranked) == 2
        # The result mentioning "price" should be boosted
        assert reranked[0]["id"] == "2"
