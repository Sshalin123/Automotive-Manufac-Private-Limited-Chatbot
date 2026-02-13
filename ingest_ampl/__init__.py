"""
AMPL Data Ingestion Module.

This module handles ingestion of AMPL-specific data sources:
- Vehicle inventory (CSV/JSON/API)
- Sales documents and brochures
- Insurance and financing documents
- FAQ and knowledge base
"""

from .inventory_loader import InventoryLoader
from .sales_docs_loader import SalesDocsLoader
from .insurance_loader import InsuranceFinanceLoader
from .faq_loader import FAQLoader
from .chunking_strategies import ChunkingStrategy, SemanticChunker, FixedSizeChunker

__all__ = [
    "InventoryLoader",
    "SalesDocsLoader",
    "InsuranceFinanceLoader",
    "FAQLoader",
    "ChunkingStrategy",
    "SemanticChunker",
    "FixedSizeChunker",
]
