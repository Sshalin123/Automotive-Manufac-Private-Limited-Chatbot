"""
Conversation Flow Engine for AMPL Chatbot (Gap 14.8).

Guides structured conversations (enquiry, test drive booking, service booking)
with step-by-step field collection.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepType(Enum):
    PROMPT = "prompt"       # Ask user for information
    CONDITION = "condition"  # Branch based on entity/intent
    ACTION = "action"       # Execute an action (create lead, book TD)
    END = "end"


@dataclass
class FlowStep:
    """A single step in a conversation flow."""
    id: str
    step_type: StepType
    prompt_text: Optional[str] = None  # For PROMPT type
    entity_field: Optional[str] = None  # Entity field to check/collect
    condition_field: Optional[str] = None  # For CONDITION type
    condition_map: Dict[str, str] = field(default_factory=dict)  # value -> next_step_id
    next_step: Optional[str] = None  # Default next step
    action_name: Optional[str] = None  # For ACTION type


@dataclass
class FlowDefinition:
    """Complete flow definition."""
    id: str
    name: str
    trigger_intents: List[str]  # Intents that trigger this flow
    steps: Dict[str, FlowStep]  # step_id -> FlowStep
    start_step: str


@dataclass
class FlowState:
    """Current state of a flow for a conversation."""
    flow_id: str
    current_step: str
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


class FlowEngine:
    """
    Manages conversation flows with step-by-step execution.

    When an intent triggers a flow, the engine tracks which fields
    have been collected and prompts for missing ones.
    """

    def __init__(self):
        self._flows: Dict[str, FlowDefinition] = {}
        self._states: Dict[str, FlowState] = {}  # conversation_id -> state

    def register_flow(self, flow: FlowDefinition):
        self._flows[flow.id] = flow
        logger.info(f"Flow registered: {flow.name} ({len(flow.steps)} steps)")

    def check_trigger(self, intent: str) -> Optional[FlowDefinition]:
        """Check if an intent should trigger a flow."""
        for flow in self._flows.values():
            if intent in flow.trigger_intents:
                return flow
        return None

    def get_next_prompt(
        self,
        conversation_id: str,
        intent: str,
        entities: Any = None,
    ) -> Optional[str]:
        """
        Get the next prompt for the conversation flow.

        Returns None if no flow is active or all fields collected.
        """
        # Check if flow is already active
        state = self._states.get(conversation_id)

        if not state:
            # Try to start a new flow
            flow = self.check_trigger(intent)
            if not flow:
                return None
            state = FlowState(flow_id=flow.id, current_step=flow.start_step)
            self._states[conversation_id] = state

        flow = self._flows.get(state.flow_id)
        if not flow or state.completed:
            return None

        # Update collected fields from entities
        if entities:
            self._update_fields_from_entities(state, entities)

        # Get current step
        step = flow.steps.get(state.current_step)
        if not step:
            state.completed = True
            return None

        # Process based on step type
        if step.step_type == StepType.PROMPT:
            # Check if field already collected
            if step.entity_field and step.entity_field in state.collected_fields:
                # Move to next step
                state.current_step = step.next_step or ""
                if not state.current_step:
                    state.completed = True
                    return None
                return self.get_next_prompt(conversation_id, intent, entities)
            return step.prompt_text

        elif step.step_type == StepType.CONDITION:
            # Branch based on collected field value
            field_val = state.collected_fields.get(step.condition_field, "default")
            next_id = step.condition_map.get(str(field_val), step.next_step)
            if next_id:
                state.current_step = next_id
                return self.get_next_prompt(conversation_id, intent, entities)
            state.completed = True
            return None

        elif step.step_type == StepType.ACTION:
            # Action steps don't produce prompts
            state.current_step = step.next_step or ""
            if not state.current_step:
                state.completed = True
            return None

        elif step.step_type == StepType.END:
            state.completed = True
            return None

        return None

    def _update_fields_from_entities(self, state: FlowState, entities: Any):
        """Extract relevant fields from entities into flow state."""
        field_map = {
            "phone": "phone_numbers",
            "email": "email_addresses",
            "name": "names",
            "model": "models_mentioned",
            "city": "city",
            "budget": "budget_max",
            "fuel_type": "fuel_types_mentioned",
            "color": "color_preference",
        }
        for flow_field, entity_attr in field_map.items():
            if flow_field not in state.collected_fields:
                val = getattr(entities, entity_attr, None)
                if val:
                    if isinstance(val, list) and val:
                        state.collected_fields[flow_field] = val[0]
                    elif not isinstance(val, list):
                        state.collected_fields[flow_field] = val

    def get_state(self, conversation_id: str) -> Optional[FlowState]:
        return self._states.get(conversation_id)

    def reset(self, conversation_id: str):
        self._states.pop(conversation_id, None)

    def get_collected_fields(self, conversation_id: str) -> Dict[str, Any]:
        state = self._states.get(conversation_id)
        return state.collected_fields if state else {}
