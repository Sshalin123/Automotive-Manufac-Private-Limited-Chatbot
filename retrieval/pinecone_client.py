"""
Pinecone Client for AMPL Chatbot.

Handles all vector database operations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a vector search."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


@dataclass
class PineconeConfig:
    """Configuration for Pinecone client."""
    api_key: str
    environment: str = ""  # Not needed for serverless
    index_name: str = "ampl-chatbot"
    dimension: int = 1024
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class PineconeClient:
    """
    Client for Pinecone vector database operations.

    Supports:
    - Index management
    - Vector upsert (single and batch)
    - Similarity search with filtering
    - Namespace management
    """

    # Namespaces for different document types
    NAMESPACES = {
        "inventory": "inventory",
        "sales": "sales",
        "insurance": "insurance",
        "faq": "faq",
        "general": "",
    }

    def __init__(self, config: PineconeConfig):
        """
        Initialize the Pinecone client.

        Args:
            config: Pinecone configuration
        """
        self.config = config
        self._client = None
        self._index = None

        self._initialize()

    def _initialize(self):
        """Initialize Pinecone client and index."""
        try:
            self._client = Pinecone(api_key=self.config.api_key)

            # Check if index exists
            existing_indexes = [idx.name for idx in self._client.list_indexes()]

            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.config.index_name}")
                self._create_index()
            else:
                logger.info(f"Using existing Pinecone index: {self.config.index_name}")

            self._index = self._client.Index(self.config.index_name)

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def _create_index(self):
        """Create a new Pinecone index."""
        self._client.create_index(
            name=self.config.index_name,
            dimension=self.config.dimension,
            metric=self.config.metric,
            spec=ServerlessSpec(
                cloud=self.config.cloud,
                region=self.config.region
            )
        )
        logger.info(f"Created Pinecone index: {self.config.index_name}")

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = ""
    ) -> int:
        """
        Upsert vectors to the index.

        Args:
            vectors: List of vectors with id, values, and metadata
            namespace: Target namespace

        Returns:
            Number of vectors upserted
        """
        if not vectors:
            return 0

        try:
            # Format vectors for Pinecone
            formatted = [
                (v["id"], v["values"], v.get("metadata", {}))
                for v in vectors
            ]

            # Upsert in batches of 100
            batch_size = 100
            upserted = 0

            for i in range(0, len(formatted), batch_size):
                batch = formatted[i:i + batch_size]
                self._index.upsert(vectors=batch, namespace=namespace)
                upserted += len(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1}, total: {upserted}")

            logger.info(f"Upserted {upserted} vectors to namespace '{namespace}'")
            return upserted

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise

    def upsert_single(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        namespace: str = ""
    ) -> bool:
        """
        Upsert a single vector.

        Args:
            id: Vector ID
            embedding: Embedding vector
            metadata: Vector metadata
            namespace: Target namespace

        Returns:
            True if successful
        """
        try:
            self._index.upsert(
                vectors=[(id, embedding, metadata)],
                namespace=namespace
            )
            return True
        except Exception as e:
            logger.error(f"Single upsert failed: {e}")
            return False

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Query for similar vectors.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            namespace: Namespace to search
            filter: Metadata filter
            include_metadata: Include metadata in results
            min_score: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        try:
            response = self._index.query(
                vector=embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=include_metadata
            )

            results = []
            for match in response.matches:
                if match.score < min_score:
                    continue

                metadata = match.metadata or {}
                results.append(SearchResult(
                    id=match.id,
                    score=match.score,
                    text=metadata.get("text", ""),
                    metadata=metadata,
                    source=metadata.get("source")
                ))

            logger.debug(f"Query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query_all_namespaces(
        self,
        embedding: List[float],
        top_k: int = 5,
        namespaces: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Query across multiple namespaces.

        Args:
            embedding: Query embedding
            top_k: Results per namespace
            namespaces: Namespaces to search (default: all)
            filter: Metadata filter
            min_score: Minimum similarity score

        Returns:
            Combined and sorted results
        """
        if namespaces is None:
            namespaces = list(self.NAMESPACES.values())

        all_results = []

        for namespace in namespaces:
            try:
                results = self.query(
                    embedding=embedding,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter,
                    min_score=min_score
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Query failed for namespace '{namespace}': {e}")
                continue

        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete vectors from the index.

        Args:
            ids: Vector IDs to delete
            delete_all: Delete all vectors in namespace
            namespace: Target namespace
            filter: Metadata filter for deletion

        Returns:
            True if successful
        """
        try:
            if delete_all:
                self._index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted all vectors from namespace '{namespace}'")
            elif ids:
                self._index.delete(ids=ids, namespace=namespace)
                logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
            elif filter:
                self._index.delete(filter=filter, namespace=namespace)
                logger.info(f"Deleted vectors matching filter from namespace '{namespace}'")

            return True

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats
        """
        try:
            stats = self._index.describe_index_stats()

            return {
                "total_vector_count": getattr(stats, 'total_vector_count', 0),
                "dimension": getattr(stats, 'dimension', 0),
                "index_fullness": getattr(stats, 'index_fullness', 0.0),
                "namespaces": {
                    name: {
                        "vector_count": getattr(ns, 'vector_count', 0)
                    }
                    for name, ns in (getattr(stats, 'namespaces', {}) or {}).items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_namespace_count(self, namespace: str = "") -> int:
        """
        Get vector count for a namespace.

        Args:
            namespace: Target namespace

        Returns:
            Number of vectors
        """
        stats = self.get_stats()
        namespaces = stats.get("namespaces", {})

        if namespace in namespaces:
            return namespaces[namespace].get("vector_count", 0)

        return stats.get("total_vector_count", 0)

    def fetch(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch vectors by ID.

        Args:
            ids: Vector IDs to fetch
            namespace: Target namespace

        Returns:
            Dictionary mapping IDs to vector data
        """
        try:
            response = self._index.fetch(ids=ids, namespace=namespace)
            return {
                id: {
                    "values": vec.values,
                    "metadata": vec.metadata
                }
                for id, vec in response.vectors.items()
            }
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return {}

    async def health_check(self) -> bool:
        """
        Check if Pinecone is healthy.

        Returns:
            True if operational
        """
        try:
            stats = self.get_stats()
            return "total_vector_count" in stats
        except:
            return False
