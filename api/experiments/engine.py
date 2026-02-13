"""
A/B Testing Engine for AMPL Chatbot (Gap 14.7).

Manages experiments with weighted random variant assignment,
sticky per-conversation.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """A single variant in an experiment."""
    name: str
    weight: float = 0.5  # 0.0 - 1.0
    overrides: Dict[str, Any] = field(default_factory=dict)
    # overrides can include: temperature, top_k, system_prompt, similarity_threshold, etc.


@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    variants: List[Variant]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class ExperimentResult:
    """Metrics collected for a variant."""
    experiment_id: str
    variant_name: str
    conversation_count: int = 0
    avg_rating: float = 0.0
    lead_conversion_rate: float = 0.0


class ExperimentEngine:
    """Manages A/B testing experiments."""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        # Sticky assignments: {(experiment_id, conversation_id): variant_name}
        self._assignments: Dict[tuple, str] = {}

    def create_experiment(self, experiment: Experiment) -> Experiment:
        self._experiments[experiment.id] = experiment
        logger.info(f"Experiment created: {experiment.name} ({len(experiment.variants)} variants)")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        return list(self._experiments.values())

    def assign_variant(self, experiment_id: str, conversation_id: str) -> Optional[Variant]:
        """
        Assign a variant to a conversation (sticky).

        Uses weighted random selection on first call,
        returns same variant on subsequent calls.
        """
        exp = self._experiments.get(experiment_id)
        if not exp or not exp.is_active:
            return None

        key = (experiment_id, conversation_id)
        if key in self._assignments:
            variant_name = self._assignments[key]
            return next((v for v in exp.variants if v.name == variant_name), None)

        # Weighted random selection
        weights = [v.weight for v in exp.variants]
        total = sum(weights)
        weights = [w / total for w in weights]
        selected = random.choices(exp.variants, weights=weights, k=1)[0]

        self._assignments[key] = selected.name
        logger.debug(f"Experiment {experiment_id}: conv {conversation_id} -> {selected.name}")
        return selected

    def get_overrides(self, conversation_id: str) -> Dict[str, Any]:
        """Get all active experiment overrides for a conversation."""
        overrides = {}
        for exp in self._experiments.values():
            if not exp.is_active:
                continue
            variant = self.assign_variant(exp.id, conversation_id)
            if variant and variant.overrides:
                overrides.update(variant.overrides)
        return overrides

    def deactivate(self, experiment_id: str):
        exp = self._experiments.get(experiment_id)
        if exp:
            exp.is_active = False
