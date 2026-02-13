"""
Knowledge Base Version Tracker for AMPL Chatbot (Gap 14.9).

Tracks document versions, hashes, and staleness.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """Version info for a knowledge base document."""
    doc_id: str
    filename: str
    content_hash: str
    chunk_count: int
    namespace: str
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: int = 1


class VersionTracker:
    """Tracks knowledge base document versions."""

    def __init__(self, stale_days: int = 90):
        self._documents: Dict[str, DocumentVersion] = {}
        self.stale_days = stale_days

    def register(
        self,
        doc_id: str,
        filename: str,
        content: str,
        chunk_count: int,
        namespace: str,
    ) -> DocumentVersion:
        """Register or update a document version."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        existing = self._documents.get(doc_id)
        if existing and existing.content_hash == content_hash:
            # No change
            return existing

        version = (existing.version + 1) if existing else 1
        doc = DocumentVersion(
            doc_id=doc_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=chunk_count,
            namespace=namespace,
            version=version,
        )
        self._documents[doc_id] = doc
        logger.info(f"Document registered: {filename} v{version} ({chunk_count} chunks)")
        return doc

    def get(self, doc_id: str) -> Optional[DocumentVersion]:
        return self._documents.get(doc_id)

    def list_all(self) -> List[DocumentVersion]:
        return list(self._documents.values())

    def get_stale(self) -> List[DocumentVersion]:
        """Get documents older than stale_days."""
        cutoff = datetime.utcnow() - timedelta(days=self.stale_days)
        return [d for d in self._documents.values() if d.last_updated < cutoff]

    def remove(self, doc_id: str) -> bool:
        return self._documents.pop(doc_id, None) is not None

    def is_changed(self, doc_id: str, content: str) -> bool:
        """Check if document content has changed."""
        existing = self._documents.get(doc_id)
        if not existing:
            return True
        new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return new_hash != existing.content_hash
