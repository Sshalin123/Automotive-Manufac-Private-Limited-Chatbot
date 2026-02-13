"""
Multi-Tenancy Manager for AMPL Chatbot (Gap 14.10).

Manages per-dealership configuration, namespace prefixing, and branding.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for a single tenant (dealership)."""
    id: str
    name: str
    namespace_prefix: str  # Prepended to Pinecone namespaces
    brand_name: str = "AMPL"
    rm_name: str = ""
    rm_phone: str = ""
    rm_email: str = ""
    website_url: str = ""
    toll_free_number: str = ""
    pinecone_index: Optional[str] = None  # Override index if needed
    custom_system_prompt: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    is_active: bool = True


class TenantManager:
    """Manages multi-dealership tenant configurations."""

    def __init__(self):
        self._tenants: Dict[str, TenantConfig] = {}
        self._api_key_map: Dict[str, str] = {}  # api_key -> tenant_id

    def register(self, config: TenantConfig) -> TenantConfig:
        self._tenants[config.id] = config
        for key in config.api_keys:
            self._api_key_map[key] = config.id
        logger.info(f"Tenant registered: {config.name} ({config.id})")
        return config

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        return self._tenants.get(tenant_id)

    def get_by_api_key(self, api_key: str) -> Optional[TenantConfig]:
        tenant_id = self._api_key_map.get(api_key)
        return self._tenants.get(tenant_id) if tenant_id else None

    def list_all(self) -> List[TenantConfig]:
        return list(self._tenants.values())

    def get_namespace_prefix(self, tenant_id: str) -> str:
        """Get namespace prefix for a tenant."""
        config = self._tenants.get(tenant_id)
        return config.namespace_prefix if config else ""

    def get_service_config(self, tenant_id: str) -> Dict:
        """Get service config overrides for a tenant."""
        config = self._tenants.get(tenant_id)
        if not config:
            return {}
        return {
            "rm_name": config.rm_name,
            "rm_phone": config.rm_phone,
            "rm_email": config.rm_email,
            "website_url": config.website_url,
            "toll_free_number": config.toll_free_number,
            "brand_name": config.brand_name,
        }

    def remove(self, tenant_id: str) -> bool:
        config = self._tenants.pop(tenant_id, None)
        if config:
            for key in config.api_keys:
                self._api_key_map.pop(key, None)
        return config is not None
