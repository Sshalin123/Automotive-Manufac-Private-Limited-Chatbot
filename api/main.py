"""
Main FastAPI application for AMPL Chatbot.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import chat, leads, admin
from .services import get_services, initialize_services

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("AMPL Chatbot starting up...")
    initialize_services()
    logger.info("AMPL Chatbot ready")
    yield
    logger.info("AMPL Chatbot shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AMPL Automotive Chatbot API",
        description="AI-powered chatbot for automotive sales with RAG, lead scoring, and CRM integration.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(leads.router, prefix="/api/v1", tags=["Leads"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "AMPL Automotive Chatbot",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs",
        }

    # Health check
    @app.get("/health")
    async def health():
        services = get_services()
        return {
            "status": "healthy" if services.is_ready else "degraded",
            "services": services.health(),
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
