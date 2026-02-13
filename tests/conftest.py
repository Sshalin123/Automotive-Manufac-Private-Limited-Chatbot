"""Shared fixtures for AMPL Chatbot tests."""

import os
import pytest
from fastapi.testclient import TestClient

# Ensure we use test/mock settings
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_inventory_row():
    """Single inventory CSV row as dict."""
    return {
        "model_name": "Tata Nexon",
        "variant": "XZ+ Dark",
        "price": "1050000",
        "ex_showroom_price": "980000",
        "on_road_price": "1050000",
        "category": "SUV",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "engine_cc": "1199",
        "power_hp": "120",
        "torque_nm": "170",
        "mileage_kmpl": "17.4",
        "seating_capacity": "5",
        "body_color": "Starlight",
        "year": "2024",
        "available": "true",
        "features": "Sunroof|360 Camera|Wireless Charging",
        "description": "The Tata Nexon XZ+ is a compact SUV.",
    }


@pytest.fixture
def sample_faq():
    return {
        "question": "What financing options are available?",
        "answer": "We offer bank loans at 7.5-9.5% interest rate.",
        "category": "financing",
        "tags": ["emi", "loan"],
    }
