"""
Embedding Service for AMPL Chatbot.

Generates embeddings using AWS Bedrock or OpenAI.
"""

import json
import logging
from enum import Enum
from typing import List, Optional, Any
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    BEDROCK_TITAN = "bedrock_titan"
    OPENAI = "openai"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    provider: EmbeddingProvider = EmbeddingProvider.BEDROCK_TITAN
    model_id: str = "amazon.titan-embed-text-v2:0"
    dimension: int = 1024  # Titan v2 default
    batch_size: int = 25  # Max texts per batch
    aws_region: str = "us-east-1"
    openai_api_key: Optional[str] = None


class EmbeddingService:
    """
    Service for generating text embeddings.

    Supports:
    - AWS Bedrock Titan embeddings
    - OpenAI embeddings
    - Batch processing
    - Caching (optional)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding service.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self._client = None
        self._openai_client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == EmbeddingProvider.BEDROCK_TITAN:
            self._initialize_bedrock()
        elif self.config.provider == EmbeddingProvider.OPENAI:
            self._initialize_openai()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.config.aws_region
            )
            logger.info(f"Bedrock client initialized in {self.config.aws_region}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.config.provider == EmbeddingProvider.BEDROCK_TITAN:
            return await self._embed_bedrock(text)
        elif self.config.provider == EmbeddingProvider.OPENAI:
            return await self._embed_openai(text)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            if self.config.provider == EmbeddingProvider.OPENAI:
                # OpenAI supports batch embedding natively
                batch_embeddings = await self._embed_openai_batch(batch)
                embeddings.extend(batch_embeddings)
            else:
                # Bedrock: process individually with concurrency
                batch_embeddings = await asyncio.gather(
                    *[self._embed_bedrock(text) for text in batch]
                )
                embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_bedrock(self, text: str) -> List[float]:
        """Generate embedding using Bedrock Titan."""
        try:
            # Truncate text if too long (Titan has 8K token limit)
            if len(text) > 25000:  # Rough character estimate
                text = text[:25000]

            response = self._client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]

            logger.debug(f"Generated Bedrock embedding, dim={len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Bedrock embedding failed: {e}")
            raise

    async def _embed_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            response = self._openai_client.embeddings.create(
                model=self.config.model_id,
                input=text
            )
            embedding = response.data[0].embedding

            logger.debug(f"Generated OpenAI embedding, dim={len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch using OpenAI."""
        try:
            response = self._openai_client.embeddings.create(
                model=self.config.model_id,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]

            logger.debug(f"Generated {len(embeddings)} OpenAI embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        dimensions = {
            "amazon.titan-embed-text-v2:0": 1024,
            "amazon.titan-embed-text-v1": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.config.model_id, self.config.dimension)

    def switch_provider(self, provider: EmbeddingProvider, model_id: str):
        """
        Switch to a different embedding provider.

        Args:
            provider: New provider
            model_id: New model ID
        """
        self.config.provider = provider
        self.config.model_id = model_id
        self._initialize_client()
        logger.info(f"Switched to {provider.value} with model {model_id}")

    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.

        Returns:
            True if service is operational
        """
        try:
            embedding = await self.embed_text("test")
            return len(embedding) > 0
        except Exception as e:
            logger.warning(f"Embedding health check failed: {e}")
            return False
