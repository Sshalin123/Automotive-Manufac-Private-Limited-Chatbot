"""
Async database session management for AMPL Chatbot.

Provides async engine, session factory, and FastAPI dependency.
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)

from .models import Base

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_db(database_url: str, pool_size: int = 5, max_overflow: int = 10):
    """
    Initialize the async database engine and create tables.

    Args:
        database_url: PostgreSQL connection string (must use asyncpg driver)
        pool_size: Connection pool size
        max_overflow: Max overflow connections
    """
    global _engine, _session_factory

    # Ensure async driver
    if database_url and "postgresql://" in database_url:
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url and "sqlite://" in database_url:
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    _engine = create_async_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=False,
    )

    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables (use Alembic in production)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized")


async def close_db():
    """Close the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    if not _session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
