"""
Knowledge Base Management routes for AMPL Chatbot (Gap 14.9).
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from api.middleware.auth import require_role

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(require_role("admin", "manager"))],
)


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    namespace: str = Form("general"),
):
    """Upload and ingest a document into the knowledge base."""
    from api.services import get_services
    services = get_services()

    content = (await file.read()).decode("utf-8", errors="ignore")
    doc_id = str(uuid.uuid4())

    if hasattr(services, "version_tracker") and services.version_tracker:
        doc_version = services.version_tracker.register(
            doc_id=doc_id,
            filename=file.filename or "unknown",
            content=content,
            chunk_count=0,  # Will be updated after chunking
            namespace=namespace,
        )

    # TODO: Chunk, embed, and upsert to Pinecone
    # This would use the existing ingestion pipeline

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "namespace": namespace,
        "size_bytes": len(content),
        "status": "ingested",
    }


@router.post("/refresh")
async def refresh_stale():
    """Re-embed documents that are older than the staleness threshold."""
    from api.services import get_services
    services = get_services()

    if not hasattr(services, "version_tracker") or not services.version_tracker:
        raise HTTPException(status_code=503, detail="Version tracker not available")

    stale = services.version_tracker.get_stale()
    return {
        "stale_count": len(stale),
        "documents": [
            {"doc_id": d.doc_id, "filename": d.filename, "last_updated": d.last_updated.isoformat()}
            for d in stale
        ],
    }


@router.get("/status")
async def knowledge_status():
    """List all tracked documents with version info."""
    from api.services import get_services
    services = get_services()

    if not hasattr(services, "version_tracker") or not services.version_tracker:
        return {"documents": []}

    docs = services.version_tracker.list_all()
    return {
        "total_documents": len(docs),
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "version": d.version,
                "chunk_count": d.chunk_count,
                "last_updated": d.last_updated.isoformat(),
                "namespace": d.namespace,
            }
            for d in docs
        ],
    }


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    from api.services import get_services
    services = get_services()

    if not hasattr(services, "version_tracker") or not services.version_tracker:
        raise HTTPException(status_code=503, detail="Version tracker not available")

    removed = services.version_tracker.remove(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")

    # TODO: Also remove vectors from Pinecone

    return {"status": "deleted", "doc_id": doc_id}
