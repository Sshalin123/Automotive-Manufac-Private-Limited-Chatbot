"""
Lead Router for AMPL Chatbot.

Routes qualified leads to CRM systems via webhooks.
"""

import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import asyncio

import httpx

from .scoring_model import LeadScore, LeadPriority

logger = logging.getLogger(__name__)


class LeadStatus(Enum):
    """Lead status in the pipeline."""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    NEGOTIATING = "negotiating"
    WON = "won"
    LOST = "lost"
    NURTURING = "nurturing"


class CRMProvider(Enum):
    """Supported CRM providers."""
    ZOHO = "zoho"
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"
    CUSTOM = "custom"


@dataclass
class Lead:
    """Lead data structure."""

    # Core identifiers
    lead_id: str
    conversation_id: str

    # Score information
    score: int
    priority: LeadPriority
    status: LeadStatus = LeadStatus.NEW

    # Customer information
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None

    # Interest details
    primary_intent: Optional[str] = None
    models_interested: List[str] = field(default_factory=list)
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    timeline: Optional[str] = None

    # Engagement
    conversation_summary: Optional[str] = None
    conversation_turns: int = 0
    last_message: Optional[str] = None

    # Trade-in
    trade_in_model: Optional[str] = None
    trade_in_year: Optional[int] = None

    # Routing
    assigned_to: Optional[str] = None
    source: str = "chatbot"

    # Metadata
    signals: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    crm_id: Optional[str] = None
    routed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        data = asdict(self)
        # Convert enums to strings
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        # Convert datetime to ISO strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.routed_at:
            data["routed_at"] = self.routed_at.isoformat()
        return data

    def to_crm_payload(self, provider: CRMProvider) -> Dict[str, Any]:
        """
        Convert to CRM-specific payload format.

        Args:
            provider: Target CRM provider

        Returns:
            Dictionary formatted for the specific CRM
        """
        if provider == CRMProvider.HUBSPOT:
            return self._to_hubspot_format()
        elif provider == CRMProvider.ZOHO:
            return self._to_zoho_format()
        elif provider == CRMProvider.SALESFORCE:
            return self._to_salesforce_format()
        else:
            return self.to_dict()

    def _to_hubspot_format(self) -> Dict[str, Any]:
        """Format for HubSpot CRM."""
        properties = {
            "firstname": self.name.split()[0] if self.name else "",
            "lastname": " ".join(self.name.split()[1:]) if self.name and len(self.name.split()) > 1 else "",
            "email": self.email or "",
            "phone": self.phone or "",
            "city": self.city or "",
            "lead_source": self.source,
            "lead_score": str(self.score),
            "lead_priority": self.priority.value,
            "interested_models": ", ".join(self.models_interested),
            "budget_range": f"{self.budget_min or 0} - {self.budget_max or 0}",
            "timeline": self.timeline or "",
            "conversation_summary": self.conversation_summary or "",
        }
        return {"properties": properties}

    def _to_zoho_format(self) -> Dict[str, Any]:
        """Format for Zoho CRM."""
        return {
            "data": [{
                "Last_Name": self.name.split()[-1] if self.name else "Unknown",
                "First_Name": self.name.split()[0] if self.name else "",
                "Email": self.email or "",
                "Phone": self.phone or "",
                "City": self.city or "",
                "Lead_Source": "Website Chatbot",
                "Lead_Status": self.status.value.title(),
                "Rating": self.priority.value.upper(),
                "Description": self.conversation_summary or "",
                "Interested_Models": ", ".join(self.models_interested),
                "Budget_Min": self.budget_min,
                "Budget_Max": self.budget_max,
                "Timeline": self.timeline,
            }]
        }

    def _to_salesforce_format(self) -> Dict[str, Any]:
        """Format for Salesforce."""
        return {
            "Name": self.name or "Unknown",
            "Email": self.email or "",
            "Phone": self.phone or "",
            "City": self.city or "",
            "LeadSource": "Website Chatbot",
            "Status": self.status.value.title(),
            "Rating": self.priority.value.title(),
            "Description": self.conversation_summary or "",
            "Custom_Models__c": ", ".join(self.models_interested),
            "Custom_Budget_Min__c": self.budget_min,
            "Custom_Budget_Max__c": self.budget_max,
        }


class LeadRouter:
    """
    Routes qualified leads to CRM systems.

    Supports:
    - Webhook-based routing
    - Multiple CRM integrations
    - Retry logic for failed deliveries
    - Lead queue management
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        crm_provider: CRMProvider = CRMProvider.CUSTOM,
        api_key: Optional[str] = None,
        min_score_for_routing: int = 50,
        auto_route_hot_leads: bool = True
    ):
        """
        Initialize the lead router.

        Args:
            webhook_url: URL for webhook delivery
            crm_provider: Target CRM provider
            api_key: API key for CRM authentication
            min_score_for_routing: Minimum score to route leads
            auto_route_hot_leads: Automatically route hot leads
        """
        self.webhook_url = webhook_url
        self.crm_provider = crm_provider
        self.api_key = api_key
        self.min_score_for_routing = min_score_for_routing
        self.auto_route_hot_leads = auto_route_hot_leads

        # Lead queue
        self._lead_queue: List[Lead] = []
        self._routed_leads: Dict[str, Lead] = {}

    def should_route(self, lead: Lead) -> bool:
        """
        Check if a lead should be routed to CRM.

        Args:
            lead: Lead to check

        Returns:
            True if lead should be routed
        """
        # Don't route disqualified leads
        if lead.priority == LeadPriority.DISQUALIFIED:
            return False

        # Route hot leads immediately if auto-routing is enabled
        if self.auto_route_hot_leads and lead.priority == LeadPriority.HOT:
            return True

        # Check score threshold
        return lead.score >= self.min_score_for_routing

    async def route_lead(self, lead: Lead) -> bool:
        """
        Route a lead to the configured CRM.

        Args:
            lead: Lead to route

        Returns:
            True if routing was successful
        """
        if not self.webhook_url:
            logger.warning("No webhook URL configured, lead not routed")
            self._lead_queue.append(lead)
            return False

        try:
            # Prepare payload based on CRM provider
            payload = lead.to_crm_payload(self.crm_provider)

            headers = {
                "Content-Type": "application/json",
            }

            # Add authentication if configured
            if self.api_key:
                if self.crm_provider == CRMProvider.HUBSPOT:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                elif self.crm_provider == CRMProvider.ZOHO:
                    headers["Authorization"] = f"Zoho-oauthtoken {self.api_key}"
                else:
                    headers["X-API-Key"] = self.api_key

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )

                if response.status_code in [200, 201, 202]:
                    lead.routed_at = datetime.utcnow()
                    lead.status = LeadStatus.NEW

                    # Try to extract CRM ID from response
                    try:
                        response_data = response.json()
                        if "id" in response_data:
                            lead.crm_id = str(response_data["id"])
                        elif "data" in response_data and response_data["data"]:
                            lead.crm_id = str(response_data["data"][0].get("id", ""))
                    except:
                        pass

                    self._routed_leads[lead.lead_id] = lead
                    logger.info(
                        f"Lead routed successfully",
                        lead_id=lead.lead_id,
                        score=lead.score,
                        crm_id=lead.crm_id
                    )
                    return True
                else:
                    logger.error(
                        f"Lead routing failed",
                        lead_id=lead.lead_id,
                        status_code=response.status_code,
                        response=response.text[:500]
                    )
                    self._lead_queue.append(lead)
                    return False

        except Exception as e:
            logger.error(f"Lead routing error: {e}", lead_id=lead.lead_id)
            self._lead_queue.append(lead)
            return False

    async def route_lead_with_retry(
        self,
        lead: Lead,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> bool:
        """
        Route a lead with retry logic.

        Args:
            lead: Lead to route
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if routing was successful
        """
        for attempt in range(max_retries):
            success = await self.route_lead(lead)
            if success:
                return True

            if attempt < max_retries - 1:
                logger.info(
                    f"Retrying lead routing",
                    lead_id=lead.lead_id,
                    attempt=attempt + 1,
                    max_retries=max_retries
                )
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        return False

    async def process_queue(self) -> int:
        """
        Process queued leads.

        Returns:
            Number of successfully routed leads
        """
        if not self._lead_queue:
            return 0

        success_count = 0
        remaining = []

        for lead in self._lead_queue:
            success = await self.route_lead(lead)
            if success:
                success_count += 1
            else:
                remaining.append(lead)

        self._lead_queue = remaining
        return success_count

    def queue_lead(self, lead: Lead):
        """
        Add a lead to the queue for later processing.

        Args:
            lead: Lead to queue
        """
        self._lead_queue.append(lead)
        logger.info(f"Lead queued", lead_id=lead.lead_id)

    def get_queue_size(self) -> int:
        """Get the number of leads in the queue."""
        return len(self._lead_queue)

    def get_queued_leads(self) -> List[Lead]:
        """Get all queued leads."""
        return self._lead_queue.copy()

    def get_routed_leads(self) -> Dict[str, Lead]:
        """Get all routed leads."""
        return self._routed_leads.copy()

    def get_lead(self, lead_id: str) -> Optional[Lead]:
        """Get a specific lead by ID."""
        return self._routed_leads.get(lead_id)

    def update_lead_status(self, lead_id: str, status: LeadStatus) -> bool:
        """
        Update the status of a routed lead.

        Args:
            lead_id: Lead ID
            status: New status

        Returns:
            True if lead was found and updated
        """
        if lead_id in self._routed_leads:
            self._routed_leads[lead_id].status = status
            self._routed_leads[lead_id].updated_at = datetime.utcnow()
            return True
        return False

    def create_lead_from_score(
        self,
        lead_id: str,
        conversation_id: str,
        score_result: LeadScore,
        customer_info: Optional[Dict[str, Any]] = None,
        conversation_summary: Optional[str] = None
    ) -> Lead:
        """
        Create a Lead object from scoring result.

        Args:
            lead_id: Unique lead identifier
            conversation_id: Conversation identifier
            score_result: Lead scoring result
            customer_info: Optional customer information
            conversation_summary: Optional conversation summary

        Returns:
            Lead object
        """
        info = customer_info or {}

        return Lead(
            lead_id=lead_id,
            conversation_id=conversation_id,
            score=score_result.score,
            priority=score_result.priority,
            name=info.get("name"),
            phone=info.get("phone"),
            email=info.get("email"),
            city=info.get("city"),
            primary_intent=info.get("primary_intent"),
            models_interested=info.get("models_interested", []),
            budget_min=info.get("budget_min"),
            budget_max=info.get("budget_max"),
            timeline=info.get("timeline"),
            conversation_summary=conversation_summary,
            conversation_turns=info.get("conversation_turns", 1),
            trade_in_model=info.get("trade_in_model"),
            trade_in_year=info.get("trade_in_year"),
            signals=score_result.signals,
            recommendations=score_result.recommendations,
        )
