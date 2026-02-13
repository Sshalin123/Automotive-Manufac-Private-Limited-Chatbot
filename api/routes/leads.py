"""
Lead Management API Routes for AMPL Chatbot.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# Enums
class LeadPriority(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    NEGOTIATING = "negotiating"
    WON = "won"
    LOST = "lost"
    NURTURING = "nurturing"


# Models
class LeadCreate(BaseModel):
    """Lead creation request."""
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None
    conversation_id: str
    primary_intent: Optional[str] = None
    models_interested: List[str] = []
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    timeline: Optional[str] = None
    source: str = "chatbot"


class Lead(BaseModel):
    """Lead model."""
    id: str
    score: int
    priority: LeadPriority
    status: LeadStatus
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    city: Optional[str]
    conversation_id: str
    primary_intent: Optional[str]
    models_interested: List[str]
    budget_min: Optional[float]
    budget_max: Optional[float]
    timeline: Optional[str]
    assigned_to: Optional[str]
    source: str
    signals: List[str] = []
    recommendations: List[str] = []
    created_at: str
    updated_at: str


class LeadUpdate(BaseModel):
    """Lead update request."""
    status: Optional[LeadStatus] = None
    assigned_to: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None


class LeadStats(BaseModel):
    """Lead statistics."""
    total: int
    by_priority: Dict[str, int]
    by_status: Dict[str, int]
    conversion_rate: float
    average_score: float


class LeadList(BaseModel):
    """Paginated lead list."""
    leads: List[Lead]
    total: int
    page: int
    page_size: int
    has_next: bool


# In-memory lead store (replace with database in production)
leads_db: Dict[str, Dict[str, Any]] = {}


@router.post("/leads", response_model=Lead)
async def create_lead(request: LeadCreate):
    """
    Create a new lead.

    This is typically called automatically by the chat system
    when a qualified lead is detected.
    """
    lead_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Calculate score based on available information
    score = _calculate_lead_score(request)
    priority = _determine_priority(score)

    lead = {
        "id": lead_id,
        "score": score,
        "priority": priority,
        "status": LeadStatus.NEW,
        "name": request.name,
        "phone": request.phone,
        "email": request.email,
        "city": request.city,
        "conversation_id": request.conversation_id,
        "primary_intent": request.primary_intent,
        "models_interested": request.models_interested,
        "budget_min": request.budget_min,
        "budget_max": request.budget_max,
        "timeline": request.timeline,
        "assigned_to": None,
        "source": request.source,
        "signals": [],
        "recommendations": _get_recommendations(request),
        "created_at": now,
        "updated_at": now
    }

    leads_db[lead_id] = lead

    logger.info(f"Lead created: {lead_id}, score: {score}, priority: {priority}")

    return Lead(**lead)


@router.get("/leads", response_model=LeadList)
async def list_leads(
    status: Optional[LeadStatus] = None,
    priority: Optional[LeadPriority] = None,
    min_score: Optional[int] = Query(None, ge=0, le=100),
    assigned_to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """
    List leads with filtering and pagination.
    """
    # Filter leads
    filtered = list(leads_db.values())

    if status:
        filtered = [l for l in filtered if l["status"] == status]

    if priority:
        filtered = [l for l in filtered if l["priority"] == priority]

    if min_score is not None:
        filtered = [l for l in filtered if l["score"] >= min_score]

    if assigned_to:
        filtered = [l for l in filtered if l.get("assigned_to") == assigned_to]

    # Sort by score descending
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Paginate
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]

    return LeadList(
        leads=[Lead(**l) for l in paginated],
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total
    )


@router.get("/leads/{lead_id}", response_model=Lead)
async def get_lead(lead_id: str):
    """Get a specific lead."""
    if lead_id not in leads_db:
        raise HTTPException(status_code=404, detail="Lead not found")

    return Lead(**leads_db[lead_id])


@router.patch("/leads/{lead_id}", response_model=Lead)
async def update_lead(lead_id: str, update: LeadUpdate):
    """Update a lead."""
    if lead_id not in leads_db:
        raise HTTPException(status_code=404, detail="Lead not found")

    lead = leads_db[lead_id]

    # Update fields
    if update.status:
        lead["status"] = update.status
    if update.assigned_to is not None:
        lead["assigned_to"] = update.assigned_to
    if update.name:
        lead["name"] = update.name
    if update.phone:
        lead["phone"] = update.phone
    if update.email:
        lead["email"] = update.email

    lead["updated_at"] = datetime.utcnow().isoformat()

    logger.info(f"Lead updated: {lead_id}")

    return Lead(**lead)


@router.post("/leads/{lead_id}/assign")
async def assign_lead(lead_id: str, sales_rep_id: str):
    """Assign a lead to a sales representative."""
    if lead_id not in leads_db:
        raise HTTPException(status_code=404, detail="Lead not found")

    leads_db[lead_id]["assigned_to"] = sales_rep_id
    leads_db[lead_id]["status"] = LeadStatus.CONTACTED
    leads_db[lead_id]["updated_at"] = datetime.utcnow().isoformat()

    return {
        "message": "Lead assigned successfully",
        "lead_id": lead_id,
        "assigned_to": sales_rep_id
    }


@router.get("/leads/stats/summary", response_model=LeadStats)
async def get_lead_stats():
    """Get lead statistics summary."""
    leads = list(leads_db.values())
    total = len(leads)

    if total == 0:
        return LeadStats(
            total=0,
            by_priority={},
            by_status={},
            conversion_rate=0.0,
            average_score=0.0
        )

    # Count by priority
    by_priority = {}
    for p in LeadPriority:
        by_priority[p.value] = len([l for l in leads if l["priority"] == p])

    # Count by status
    by_status = {}
    for s in LeadStatus:
        by_status[s.value] = len([l for l in leads if l["status"] == s])

    # Calculate conversion rate
    won = len([l for l in leads if l["status"] == LeadStatus.WON])
    closed = won + len([l for l in leads if l["status"] == LeadStatus.LOST])
    conversion_rate = (won / closed * 100) if closed > 0 else 0.0

    # Average score
    average_score = sum(l["score"] for l in leads) / total

    return LeadStats(
        total=total,
        by_priority=by_priority,
        by_status=by_status,
        conversion_rate=round(conversion_rate, 2),
        average_score=round(average_score, 2)
    )


@router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: str):
    """Delete a lead (soft delete in production)."""
    if lead_id not in leads_db:
        raise HTTPException(status_code=404, detail="Lead not found")

    del leads_db[lead_id]

    return {"message": "Lead deleted", "lead_id": lead_id}


# Helper functions
def _calculate_lead_score(request: LeadCreate) -> int:
    """Calculate lead score based on available information."""
    score = 20  # Base score

    # Contact information
    if request.phone:
        score += 15
    if request.email:
        score += 10
    if request.name:
        score += 5

    # Interest signals
    if request.models_interested:
        score += 15
    if request.budget_max:
        score += 10
    if request.timeline:
        if request.timeline in ["immediate", "this_month"]:
            score += 15
        elif request.timeline == "this_quarter":
            score += 10

    # Intent
    if request.primary_intent in ["buy", "test_drive"]:
        score += 20
    elif request.primary_intent == "finance":
        score += 15

    return min(score, 100)


def _determine_priority(score: int) -> LeadPriority:
    """Determine lead priority from score."""
    if score >= 70:
        return LeadPriority.HOT
    elif score >= 50:
        return LeadPriority.WARM
    else:
        return LeadPriority.COLD


def _get_recommendations(request: LeadCreate) -> List[str]:
    """Generate recommendations for lead follow-up."""
    recommendations = []

    if not request.phone and not request.email:
        recommendations.append("Collect contact information")

    if not request.budget_max:
        recommendations.append("Understand budget requirements")

    if not request.models_interested:
        recommendations.append("Identify vehicle preferences")

    if request.primary_intent == "test_drive":
        recommendations.append("Schedule test drive")
    elif request.primary_intent == "buy":
        recommendations.append("Prepare quote")
        recommendations.append("Check inventory availability")

    return recommendations[:4]
