"""Tests for the Chat API endpoints."""

import pytest


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "AMPL Automotive Chatbot"
    assert "version" in data


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")


def test_chat_basic(client):
    """Send a basic chat message and get a response."""
    resp = client.post("/api/v1/chat", json={"message": "Hello, what vehicles do you have?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "conversation_id" in data
    assert data["query"] == "Hello, what vehicles do you have?"
    assert isinstance(data["processing_time_ms"], (int, float))


def test_chat_buy_intent(client):
    """Buy-intent message should return high lead score."""
    resp = client.post("/api/v1/chat", json={"message": "I want to buy a Nexon"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["lead_score"] is not None
    assert data["lead_score"] >= 50


def test_chat_finance_intent(client):
    resp = client.post("/api/v1/chat", json={"message": "What EMI options do you have for a car loan?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] in ("finance", "buy", "info")


def test_chat_test_drive_intent(client):
    resp = client.post("/api/v1/chat", json={"message": "I want to schedule a test drive for Creta"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] in ("test_drive", "buy", "info")


def test_chat_conversation_continuity(client):
    """Messages with the same conversation_id share context."""
    r1 = client.post("/api/v1/chat", json={"message": "Hi"})
    cid = r1.json()["conversation_id"]

    r2 = client.post("/api/v1/chat", json={
        "message": "Show me SUVs",
        "conversation_id": cid,
    })
    assert r2.json()["conversation_id"] == cid


def test_chat_empty_message(client):
    """Empty message should fail validation."""
    resp = client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


def test_chat_long_message(client):
    """Message exceeding max length should fail."""
    resp = client.post("/api/v1/chat", json={"message": "x" * 2001})
    assert resp.status_code == 422


def test_chat_stats(client):
    resp = client.get("/api/v1/chat/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_conversations" in data


def test_docs_endpoint(client):
    """Swagger docs should be available."""
    resp = client.get("/docs")
    assert resp.status_code == 200
