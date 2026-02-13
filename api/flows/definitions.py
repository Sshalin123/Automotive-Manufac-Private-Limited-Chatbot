"""
Pre-built conversation flow definitions for AMPL Chatbot (Gap 14.8).
"""

from .engine import FlowDefinition, FlowStep, StepType


def get_enquiry_flow() -> FlowDefinition:
    """Enquiry flow: collect name, phone, model, variant, timeline."""
    return FlowDefinition(
        id="enquiry",
        name="Vehicle Enquiry",
        trigger_intents=["buy", "info"],
        start_step="ask_model",
        steps={
            "ask_model": FlowStep(
                id="ask_model",
                step_type=StepType.PROMPT,
                prompt_text="Which vehicle model are you interested in?",
                entity_field="model",
                next_step="ask_name",
            ),
            "ask_name": FlowStep(
                id="ask_name",
                step_type=StepType.PROMPT,
                prompt_text="May I have your name please?",
                entity_field="name",
                next_step="ask_phone",
            ),
            "ask_phone": FlowStep(
                id="ask_phone",
                step_type=StepType.PROMPT,
                prompt_text="Could you share your phone number so our team can assist you better?",
                entity_field="phone",
                next_step="ask_timeline",
            ),
            "ask_timeline": FlowStep(
                id="ask_timeline",
                step_type=StepType.PROMPT,
                prompt_text="When are you planning to make the purchase?",
                entity_field="timeline",
                next_step="end",
            ),
            "end": FlowStep(id="end", step_type=StepType.END),
        },
    )


def get_test_drive_flow() -> FlowDefinition:
    """Test drive booking flow."""
    return FlowDefinition(
        id="test_drive",
        name="Test Drive Booking",
        trigger_intents=["test_drive"],
        start_step="ask_model",
        steps={
            "ask_model": FlowStep(
                id="ask_model",
                step_type=StepType.PROMPT,
                prompt_text="Which model would you like to test drive?",
                entity_field="model",
                next_step="ask_name",
            ),
            "ask_name": FlowStep(
                id="ask_name",
                step_type=StepType.PROMPT,
                prompt_text="May I have your name for the booking?",
                entity_field="name",
                next_step="ask_phone",
            ),
            "ask_phone": FlowStep(
                id="ask_phone",
                step_type=StepType.PROMPT,
                prompt_text="Please share your phone number for the test drive confirmation.",
                entity_field="phone",
                next_step="ask_city",
            ),
            "ask_city": FlowStep(
                id="ask_city",
                step_type=StepType.PROMPT,
                prompt_text="Which city/location would you prefer for the test drive?",
                entity_field="city",
                next_step="end",
            ),
            "end": FlowStep(id="end", step_type=StepType.END),
        },
    )


def get_service_booking_flow() -> FlowDefinition:
    """Service booking flow."""
    return FlowDefinition(
        id="service_booking",
        name="Service Booking",
        trigger_intents=["service", "service_reminder"],
        start_step="ask_model",
        steps={
            "ask_model": FlowStep(
                id="ask_model",
                step_type=StepType.PROMPT,
                prompt_text="Which vehicle needs servicing? Please share the model name.",
                entity_field="model",
                next_step="ask_phone",
            ),
            "ask_phone": FlowStep(
                id="ask_phone",
                step_type=StepType.PROMPT,
                prompt_text="Please share your registered phone number.",
                entity_field="phone",
                next_step="ask_city",
            ),
            "ask_city": FlowStep(
                id="ask_city",
                step_type=StepType.PROMPT,
                prompt_text="Which city/service center location do you prefer?",
                entity_field="city",
                next_step="end",
            ),
            "end": FlowStep(id="end", step_type=StepType.END),
        },
    )


def register_all_flows(engine):
    """Register all pre-built flows with the engine."""
    engine.register_flow(get_enquiry_flow())
    engine.register_flow(get_test_drive_flow())
    engine.register_flow(get_service_booking_flow())
