"""
API Module for AMPL Chatbot.

FastAPI application with routes for:
- Chat interactions
- Lead management
- Admin operations
"""

from .main import create_app, app

__all__ = ["create_app", "app"]
