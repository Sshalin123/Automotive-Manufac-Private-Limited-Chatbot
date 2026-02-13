"""
Retrieval Module for AMPL Chatbot.

This module provides RAG retrieval capabilities:
- Embedding generation (Bedrock/OpenAI)
- Pinecone vector operations
- Context building with citations
- Reranking support
"""

from .embedder import EmbeddingService, EmbeddingProvider
from .pinecone_client import PineconeClient, SearchResult
from .context_builder import ContextBuilder, Context
from .reranker import Reranker, RerankedResult

__all__ = [
    "EmbeddingService",
    "EmbeddingProvider",
    "PineconeClient",
    "SearchResult",
    "ContextBuilder",
    "Context",
    "Reranker",
    "RerankedResult",
]
