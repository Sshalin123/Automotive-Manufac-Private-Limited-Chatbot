"""Tests for data ingestion loaders."""

import json
import csv
import tempfile
import os
import pytest
from pathlib import Path

from ingest_ampl.inventory_loader import InventoryLoader
from ingest_ampl.faq_loader import FAQLoader
from ingest_ampl.chunking_strategies import FixedSizeChunker


# ── Inventory Loader ──────────────────────────────────

class TestInventoryLoader:
    def test_load_from_csv(self, sample_inventory_row, tmp_path):
        csv_file = tmp_path / "test_inventory.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sample_inventory_row.keys())
            writer.writeheader()
            writer.writerow(sample_inventory_row)

        loader = InventoryLoader()
        docs = loader.load_from_csv(str(csv_file))
        assert len(docs) == 1
        assert "Tata Nexon" in docs[0].content
        assert docs[0].metadata["category"] == "SUV"

    def test_load_from_json(self, tmp_path):
        data = [
            {
                "model_name": "Hyundai Creta",
                "variant": "SX(O)",
                "price": 1825000,
                "category": "SUV",
                "fuel_type": "Petrol",
                "transmission": "Automatic",
                "year": 2024,
                "available": True,
                "description": "Premium mid-size SUV.",
            }
        ]
        json_file = tmp_path / "test_inventory.json"
        json_file.write_text(json.dumps(data))

        loader = InventoryLoader()
        docs = loader.load_from_json(str(json_file))
        assert len(docs) == 1
        assert "Hyundai Creta" in docs[0].content

    def test_empty_csv(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("model_name,price\n")
        loader = InventoryLoader()
        docs = loader.load_from_csv(str(csv_file))
        assert len(docs) == 0


# ── FAQ Loader ────────────────────────────────────────

class TestFAQLoader:
    def test_load_from_json(self, sample_faq, tmp_path):
        faq_file = tmp_path / "test_faqs.json"
        faq_file.write_text(json.dumps([sample_faq]))

        loader = FAQLoader()
        docs = loader.load_from_json(str(faq_file))
        assert len(docs) == 1
        assert "financing" in docs[0].content.lower() or "loan" in docs[0].content.lower()

    def test_load_default_faqs(self):
        loader = FAQLoader()
        docs = loader.load_default_automotive_faqs()
        assert len(docs) > 5  # Should have multiple defaults

    def test_add_single_faq(self):
        loader = FAQLoader()
        loader.add_faq("What is the warranty?", "3 years / 1 lakh km", category="warranty")
        docs = loader.get_documents()
        assert len(docs) >= 1


# ── Chunking Strategies ──────────────────────────────

class TestChunking:
    def test_fixed_size_chunker(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        text = "word " * 200  # ~1000 chars
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size + some tolerance
        for c in chunks:
            assert len(c.content) <= 150  # chunk_size + overlap tolerance

    def test_single_chunk_for_short_text(self):
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        text = "This is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_metadata(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        text = "word " * 100
        meta = {"source": "test"}
        chunks = chunker.chunk(text, meta)
        for c in chunks:
            assert c.metadata.get("source") == "test"
