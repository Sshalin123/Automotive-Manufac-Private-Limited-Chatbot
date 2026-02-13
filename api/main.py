"""
Main FastAPI application for AMPL Chatbot.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat, leads, admin, webhooks, scheduled, notifications
from .routes import auth, realtime, handoff, analytics, experiments, knowledge, tenants, compliance
from .services import get_services, initialize_services
from .middleware.metrics import MetricsMiddleware, metrics_endpoint
from .middleware.rate_limit import RateLimitMiddleware
from config.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("AMPL Chatbot starting up...")

    # Initialize database (if configured)
    settings = get_settings()
    if settings.database_url:
        try:
            from database.session import init_db
            await init_db(settings.database_url)
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database init failed (running without DB): {e}")

    initialize_services()
    logger.info("AMPL Chatbot ready")
    yield
    logger.info("AMPL Chatbot shutting down...")

    # Close database
    if settings.database_url:
        try:
            from database.session import close_db
            await close_db()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

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
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware
    app.add_middleware(MetricsMiddleware)

    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
    )

    # --- Core routers ---
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(leads.router, prefix="/api/v1", tags=["Leads"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])
    app.include_router(scheduled.router, prefix="/api/v1", tags=["Scheduled"])
    app.include_router(notifications.router, prefix="/api/v1", tags=["Notifications"])

    # --- Auth ---
    app.include_router(auth.router, prefix="/api/v1", tags=["Auth"])

    # --- Real-time ---
    app.include_router(realtime.router, prefix="/api/v1", tags=["Realtime"])

    # --- Handoff ---
    app.include_router(handoff.router, prefix="/api/v1", tags=["Handoff"])

    # --- Analytics ---
    app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])

    # --- Experiments ---
    app.include_router(experiments.router, prefix="/api/v1", tags=["Experiments"])

    # --- Knowledge base ---
    app.include_router(knowledge.router, prefix="/api/v1", tags=["Knowledge"])

    # --- Multi-tenancy ---
    app.include_router(tenants.router, prefix="/api/v1", tags=["Tenants"])

    # --- Compliance / GDPR ---
    app.include_router(compliance.router, prefix="/api/v1", tags=["Compliance"])

    # --- Prometheus metrics endpoint ---
    app.get("/metrics", tags=["Monitoring"])(metrics_endpoint)

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
