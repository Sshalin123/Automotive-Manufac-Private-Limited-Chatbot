"""
A/B Testing API routes for AMPL Chatbot (Gap 14.7).
"""

import logging
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.middleware.auth import require_role

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(require_role("admin", "manager"))],
)


class VariantInput(BaseModel):
    name: str
    weight: float = 0.5
    overrides: Dict = {}


class CreateExperimentRequest(BaseModel):
    name: str
    description: str = ""
    variants: List[VariantInput]


@router.post("/")
async def create_experiment(request: CreateExperimentRequest):
    """Create a new A/B test experiment."""
    from api.services import get_services
    from api.experiments.engine import Experiment, Variant

    services = get_services()
    if not hasattr(services, "experiment_engine") or not services.experiment_engine:
        raise HTTPException(status_code=503, detail="Experiments not available")

    exp = Experiment(
        id=str(uuid.uuid4()),
        name=request.name,
        description=request.description,
        variants=[Variant(name=v.name, weight=v.weight, overrides=v.overrides) for v in request.variants],
    )
    services.experiment_engine.create_experiment(exp)
    return {"id": exp.id, "name": exp.name, "variants": len(exp.variants)}


@router.get("/")
async def list_experiments():
    """List all experiments."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "experiment_engine") or not services.experiment_engine:
        return {"experiments": []}
    experiments = services.experiment_engine.list_experiments()
    return {
        "experiments": [
            {"id": e.id, "name": e.name, "is_active": e.is_active, "variants": len(e.variants)}
            for e in experiments
        ]
    }


@router.delete("/{experiment_id}")
async def deactivate_experiment(experiment_id: str):
    """Deactivate an experiment."""
    from api.services import get_services
    services = get_services()
    if not hasattr(services, "experiment_engine") or not services.experiment_engine:
        raise HTTPException(status_code=503, detail="Experiments not available")
    services.experiment_engine.deactivate(experiment_id)
    return {"status": "deactivated"}
