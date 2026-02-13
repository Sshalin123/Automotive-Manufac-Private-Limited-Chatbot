"""
AMPL Data Ingestion CLI.

Usage:
    python -m ingest_ampl.main --source inventory --file data/inventory.csv
    python -m ingest_ampl.main --source faq --file data/faqs.json
    python -m ingest_ampl.main --source all --dir data/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from config.settings import get_settings
from retrieval.embedder import EmbeddingService, EmbeddingConfig, EmbeddingProvider
from retrieval.pinecone_client import PineconeClient, PineconeConfig

from .inventory_loader import InventoryLoader
from .faq_loader import FAQLoader
from .sales_docs_loader import SalesDocsLoader
from .insurance_loader import InsuranceFinanceLoader
from .chunking_strategies import FixedSizeChunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end data ingestion pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = self._create_embedding_service()
        self.pinecone_client = self._create_pinecone_client()
        self.chunker = FixedSizeChunker(chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)

    def _create_embedding_service(self) -> EmbeddingService:
        s = self.settings
        if s.is_openai:
            provider = EmbeddingProvider.OPENAI
            model_id = s.openai_embed_model
        else:
            provider = EmbeddingProvider.BEDROCK_TITAN
            model_id = s.bedrock_embed_model_id

        return EmbeddingService(EmbeddingConfig(
            provider=provider,
            model_id=model_id,
            aws_region=s.aws_region,
            openai_api_key=s.openai_api_key,
        ))

    def _create_pinecone_client(self) -> PineconeClient:
        s = self.settings
        if not s.pinecone_api_key:
            logger.error("PINECONE_API_KEY not set")
            sys.exit(1)

        return PineconeClient(PineconeConfig(
            api_key=s.pinecone_api_key,
            index_name=s.pinecone_index_name,
            cloud=s.pinecone_cloud,
            region=s.pinecone_region,
            dimension=self.embedding_service.get_dimension(),
        ))

    async def ingest_inventory(self, file_path: str):
        """Ingest vehicle inventory from CSV or JSON."""
        path = Path(file_path)
        loader = InventoryLoader()

        if path.suffix == ".csv":
            docs = loader.load_from_csv(file_path)
        elif path.suffix == ".json":
            docs = loader.load_from_json(file_path)
        else:
            logger.error(f"Unsupported format: {path.suffix}")
            return

        logger.info(f"Loaded {len(docs)} vehicle documents")
        await self._embed_and_upsert(docs, namespace="inventory")

    async def ingest_faqs(self, file_path: str):
        """Ingest FAQs from JSON or CSV."""
        path = Path(file_path)
        loader = FAQLoader()

        if path.suffix == ".json":
            docs = loader.load_from_json(file_path)
        elif path.suffix == ".csv":
            docs = loader.load_from_csv(file_path)
        else:
            logger.error(f"Unsupported format: {path.suffix}")
            return

        logger.info(f"Loaded {len(docs)} FAQ documents")
        await self._embed_and_upsert(docs, namespace="faq")

    async def ingest_sales_docs(self, directory: str):
        """Ingest sales documents from a directory."""
        loader = SalesDocsLoader()
        docs = loader.load_directory(directory)
        logger.info(f"Loaded {len(docs)} sales documents")
        await self._embed_and_upsert(docs, namespace="sales")

    async def ingest_insurance(self, file_path: str):
        """Ingest insurance/finance documents."""
        loader = InsuranceFinanceLoader()
        docs = loader.load_pdf(file_path)
        logger.info(f"Loaded {len(docs)} insurance documents")
        await self._embed_and_upsert(docs, namespace="insurance")

    async def ingest_default_faqs(self):
        """Ingest built-in default automotive FAQs."""
        loader = FAQLoader()
        docs = loader.load_default_automotive_faqs()
        logger.info(f"Loaded {len(docs)} default FAQ documents")
        await self._embed_and_upsert(docs, namespace="faq")

    async def ingest_all(self, data_dir: str):
        """Ingest all data from a directory."""
        d = Path(data_dir)
        if not d.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return

        # Inventory files
        for f in d.glob("inventory*.csv"):
            logger.info(f"Ingesting inventory: {f}")
            await self.ingest_inventory(str(f))
        for f in d.glob("inventory*.json"):
            logger.info(f"Ingesting inventory: {f}")
            await self.ingest_inventory(str(f))

        # FAQ files
        for f in d.glob("faq*.json"):
            logger.info(f"Ingesting FAQs: {f}")
            await self.ingest_faqs(str(f))
        for f in d.glob("faq*.csv"):
            logger.info(f"Ingesting FAQs: {f}")
            await self.ingest_faqs(str(f))

        # Sales docs
        sales_dir = d / "sales"
        if sales_dir.exists():
            logger.info(f"Ingesting sales docs from: {sales_dir}")
            await self.ingest_sales_docs(str(sales_dir))

        # Default FAQs
        await self.ingest_default_faqs()

        logger.info("All ingestion complete")

    async def _embed_and_upsert(self, docs: List[Any], namespace: str):
        """Embed documents and upsert to Pinecone."""
        if not docs:
            logger.warning("No documents to process")
            return

        texts = [doc.content for doc in docs]
        ids = [doc.id for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        # Chunk long texts
        chunked_texts = []
        chunked_ids = []
        chunked_meta = []

        for i, text in enumerate(texts):
            chunks = self.chunker.chunk(text, metadatas[i])
            if len(chunks) <= 1:
                chunked_texts.append(text)
                chunked_ids.append(ids[i])
                chunked_meta.append(metadatas[i])
            else:
                for j, chunk in enumerate(chunks):
                    chunked_texts.append(chunk.content)
                    chunked_ids.append(f"{ids[i]}_chunk_{j}")
                    meta = {**metadatas[i], "chunk_index": j, "parent_id": ids[i]}
                    chunked_meta.append(meta)

        logger.info(f"Embedding {len(chunked_texts)} chunks for namespace '{namespace}'")

        # Embed in batches
        batch_size = 25
        all_embeddings = []
        for start in range(0, len(chunked_texts), batch_size):
            batch = chunked_texts[start : start + batch_size]
            embeddings = await self.embedding_service.embed_texts(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"  Embedded batch {start // batch_size + 1}")

        # Prepare vectors for Pinecone
        vectors = []
        for idx in range(len(chunked_texts)):
            meta = {**chunked_meta[idx], "text": chunked_texts[idx][:1000]}
            vectors.append({
                "id": chunked_ids[idx],
                "values": all_embeddings[idx],
                "metadata": meta,
            })

        # Upsert to Pinecone
        self.pinecone_client.upsert(vectors, namespace=namespace)
        logger.info(f"Upserted {len(vectors)} vectors to namespace '{namespace}'")


def main():
    parser = argparse.ArgumentParser(description="AMPL Chatbot Data Ingestion")
    parser.add_argument(
        "--source",
        choices=["inventory", "faq", "sales", "insurance", "defaults", "all"],
        required=True,
        help="Data source type to ingest",
    )
    parser.add_argument("--file", help="Path to data file")
    parser.add_argument("--dir", default="./data", help="Data directory (for --source all)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = IngestionPipeline()

    if args.source == "all":
        asyncio.run(pipeline.ingest_all(args.dir))
    elif args.source == "inventory":
        if not args.file:
            logger.error("--file required for inventory ingestion")
            sys.exit(1)
        asyncio.run(pipeline.ingest_inventory(args.file))
    elif args.source == "faq":
        if not args.file:
            logger.error("--file required for FAQ ingestion")
            sys.exit(1)
        asyncio.run(pipeline.ingest_faqs(args.file))
    elif args.source == "sales":
        asyncio.run(pipeline.ingest_sales_docs(args.dir))
    elif args.source == "insurance":
        if not args.file:
            logger.error("--file required for insurance ingestion")
            sys.exit(1)
        asyncio.run(pipeline.ingest_insurance(args.file))
    elif args.source == "defaults":
        asyncio.run(pipeline.ingest_default_faqs())

    logger.info("Ingestion pipeline finished")


if __name__ == "__main__":
    main()
