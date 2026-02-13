"""
Repository classes for AMPL Chatbot data access layer.

Each repository encapsulates CRUD operations for a specific model.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Conversation, Message, Lead, LeadEvent,
    ScheduledMessage, Notification, Feedback, User, AuditLog,
)

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Data access for conversations and messages."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, conversation_id: str, **kwargs) -> Conversation:
        conv = Conversation(id=conversation_id, **kwargs)
        self.session.add(conv)
        await self.session.flush()
        return conv

    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities_json: Optional[Dict] = None,
        lead_score: Optional[float] = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            intent=intent,
            entities_json=entities_json,
            lead_score=lead_score,
        )
        self.session.add(msg)
        # Update conversation last_active_at
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(last_active_at=datetime.utcnow())
        )
        await self.session.flush()
        return msg

    async def get_messages(
        self, conversation_id: str, limit: int = 50
    ) -> List[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(
        self, tenant_id: Optional[str] = None, limit: int = 20
    ) -> List[Conversation]:
        q = select(Conversation).order_by(Conversation.last_active_at.desc()).limit(limit)
        if tenant_id:
            q = q.where(Conversation.tenant_id == tenant_id)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def count(self, tenant_id: Optional[str] = None) -> int:
        q = select(func.count(Conversation.id))
        if tenant_id:
            q = q.where(Conversation.tenant_id == tenant_id)
        result = await self.session.execute(q)
        return result.scalar() or 0


class LeadRepository:
    """Data access for leads and lead events."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Lead:
        lead = Lead(**kwargs)
        self.session.add(lead)
        await self.session.flush()
        # Record creation event
        self.session.add(LeadEvent(
            lead_id=lead.id,
            event_type="created",
            details_json={"score": kwargs.get("score"), "temperature": kwargs.get("temperature")},
        ))
        await self.session.flush()
        return lead

    async def update(self, lead_id: str, **kwargs) -> Optional[Lead]:
        lead = await self.get_by_id(lead_id)
        if not lead:
            return None
        for k, v in kwargs.items():
            if hasattr(lead, k):
                setattr(lead, k, v)
        # Record update event
        self.session.add(LeadEvent(
            lead_id=lead_id,
            event_type="updated",
            details_json=kwargs,
        ))
        await self.session.flush()
        return lead

    async def get_by_id(self, lead_id: str) -> Optional[Lead]:
        result = await self.session.execute(
            select(Lead).where(Lead.id == lead_id)
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> List[Lead]:
        result = await self.session.execute(
            select(Lead)
            .where(Lead.tenant_id == tenant_id)
            .order_by(Lead.created_at.desc())
            .offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def list_by_status(
        self, status: str, tenant_id: Optional[str] = None, limit: int = 50
    ) -> List[Lead]:
        q = select(Lead).where(Lead.status == status).order_by(Lead.created_at.desc()).limit(limit)
        if tenant_id:
            q = q.where(Lead.tenant_id == tenant_id)
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def list_by_temperature(
        self, temperature: str, tenant_id: Optional[str] = None, limit: int = 50
    ) -> List[Lead]:
        q = select(Lead).where(Lead.temperature == temperature).order_by(Lead.score.desc()).limit(limit)
        if tenant_id:
            q = q.where(Lead.tenant_id == tenant_id)
        result = await self.session.execute(q)
        return list(result.scalars().all())


class ScheduledMessageRepository:
    """Data access for scheduled messages."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> ScheduledMessage:
        msg = ScheduledMessage(**kwargs)
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def get_pending(self, before: Optional[datetime] = None) -> List[ScheduledMessage]:
        q = select(ScheduledMessage).where(ScheduledMessage.status == "pending")
        if before:
            q = q.where(ScheduledMessage.scheduled_at <= before)
        q = q.order_by(ScheduledMessage.scheduled_at.asc())
        result = await self.session.execute(q)
        return list(result.scalars().all())

    async def mark_sent(self, msg_id: str) -> None:
        await self.session.execute(
            update(ScheduledMessage)
            .where(ScheduledMessage.id == msg_id)
            .values(status="sent", sent_at=datetime.utcnow())
        )
        await self.session.flush()


class NotificationRepository:
    """Data access for notifications."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Notification:
        notif = Notification(**kwargs)
        self.session.add(notif)
        await self.session.flush()
        return notif

    async def get_by_conversation(self, conversation_id: str) -> List[Notification]:
        result = await self.session.execute(
            select(Notification)
            .where(Notification.conversation_id == conversation_id)
            .order_by(Notification.created_at.desc())
        )
        return list(result.scalars().all())


class FeedbackRepository:
    """Data access for feedback."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Feedback:
        fb = Feedback(**kwargs)
        self.session.add(fb)
        await self.session.flush()
        return fb

    async def get_by_conversation(self, conversation_id: str) -> List[Feedback]:
        result = await self.session.execute(
            select(Feedback)
            .where(Feedback.conversation_id == conversation_id)
            .order_by(Feedback.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_aggregate_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate feedback stats (avg rating, count)."""
        q = select(
            func.count(Feedback.id).label("count"),
            func.avg(Feedback.rating).label("avg_rating"),
        )
        if tenant_id:
            q = q.join(Conversation, Feedback.conversation_id == Conversation.id)
            q = q.where(Conversation.tenant_id == tenant_id)
        result = await self.session.execute(q)
        row = result.one()
        return {
            "count": row.count or 0,
            "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else None,
        }


class UserRepository:
    """Data access for users."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> User:
        user = User(**kwargs)
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_api_key(self, api_key: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.api_key == api_key, User.is_active == True)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_active_agents(self, tenant_id: Optional[str] = None) -> List[User]:
        q = select(User).where(User.is_active == True, User.role == "agent")
        if tenant_id:
            q = q.where(User.tenant_id == tenant_id)
        result = await self.session.execute(q)
        return list(result.scalars().all())


class AuditLogRepository:
    """Data access for audit logs (compliance)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(self, action: str, user_id: Optional[str] = None, **kwargs) -> AuditLog:
        entry = AuditLog(action=action, user_id=user_id, **kwargs)
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_recent(self, limit: int = 100) -> List[AuditLog]:
        result = await self.session.execute(
            select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
