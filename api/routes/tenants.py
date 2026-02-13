"""
Multi-Tenancy API routes for AMPL Chatbot (Gap 14.10).
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.middleware.auth import require_role

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/tenants",
    tags=["tenants"],
    dependencies=[Depends(require_role("admin"))],
)


class CreateTenantRequest(BaseModel):
    name: str
    namespace_prefix: str
    brand_name: str = "AMPL"
    rm_name: str = ""
    rm_phone: str = ""
    rm_email: str = ""
    website_url: str = ""


@router.post("/")
async def create_tenant(request: CreateTenantRequest):
    """Create a new tenant (dealership)."""
    from api.services import get_services
    from api.tenants.manager import TenantConfig
    import secrets

    services = get_services()
    if not hasattr(services, "tenant_manager") or not services.tenant_manager:
        raise HTTPException(status_code=503, detail="Multi-tenancy not available")

    config = TenantConfig(
        id=str(uuid.uuid4()),
        name=request.name,
        namespace_prefix=request.namespace_prefix,
        brand_name=request.brand_name,
        rm_name=request.rm_name,
        rm_phone=request.rm_phone,
        rm_email=request.rm_email,
        website_url=request.website_url,
        api_keys=[secrets.token_hex(32)],
    )
    services.tenant_manager.register(config)
    return {
        "id": config.id,
        "name": config.name,
        "api_key": config.api_keys[0],
    }


@router.get("/")
async def list_tenants():
    """List all tenants."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "tenant_manager") or not services.tenant_manager:
        return {"tenants": []}
    tenants = services.tenant_manager.list_all()
    return {
        "tenants": [
            {"id": t.id, "name": t.name, "namespace_prefix": t.namespace_prefix, "is_active": t.is_active}
            for t in tenants
        ]
    }


@router.delete("/{tenant_id}")
async def delete_tenant(tenant_id: str):
    """Remove a tenant."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "tenant_manager") or not services.tenant_manager:
        raise HTTPException(status_code=503, detail="Multi-tenancy not available")
    removed = services.tenant_manager.remove(tenant_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"status": "deleted"}
