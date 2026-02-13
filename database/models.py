"""
SQLAlchemy ORM models for AMPL Chatbot.

All persistent entities: conversations, messages, leads, users, etc.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey,
    JSON, Enum as SAEnum, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=_uuid)
    tenant_id = Column(String(36), nullable=True, index=True)
    channel = Column(String(20), default="web")  # web, whatsapp, email
    customer_phone = Column(String(15), nullable=True)
    customer_email = Column(String(255), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    leads = relationship("Lead", back_populates="conversation")

    __table_args__ = (
        Index("ix_conv_tenant_active", "tenant_id", "last_active_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(10), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    intent = Column(String(30), nullable=True)
    entities_json = Column(JSON, nullable=True)
    lead_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", uselist=False)


class Lead(Base):
    __tablename__ = "leads"

    id = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    tenant_id = Column(String(36), nullable=True, index=True)
    customer_name = Column(String(255), nullable=True)
    phone = Column(String(15), nullable=True)
    email = Column(String(255), nullable=True)
    score = Column(Float, default=0.0)
    temperature = Column(String(10), default="cold")  # hot, warm, cold
    status = Column(String(20), default="new")  # new, contacted, qualified, converted, lost
    assigned_to = Column(String(36), nullable=True)
    routed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)

    conversation = relationship("Conversation", back_populates="leads")
    events = relationship("LeadEvent", back_populates="lead", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_lead_tenant_status", "tenant_id", "status"),
        Index("ix_lead_temp", "temperature"),
    )


class LeadEvent(Base):
    __tablename__ = "lead_events"

    id = Column(String(36), primary_key=True, default=_uuid)
    lead_id = Column(String(36), ForeignKey("leads.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(30), nullable=False)  # created, score_updated, routed, status_changed
    details_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    lead = relationship("Lead", back_populates="events")


class ScheduledMessage(Base):
    __tablename__ = "scheduled_messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True)
    channel = Column(String(20), default="web")
    recipient = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    scheduled_at = Column(DateTime, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    status = Column(String(15), default="pending")  # pending, sent, failed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sched_status_time", "status", "scheduled_at"),
    )


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True)
    channel = Column(String(20), nullable=False)  # whatsapp, email, sms, widget
    notification_type = Column(String(30), nullable=False)
    content = Column(Text, nullable=False)
    recipient = Column(String(255), nullable=True)
    sent_at = Column(DateTime, nullable=True)
    status = Column(String(15), default="pending")  # pending, sent, failed
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    message_id = Column(String(36), ForeignKey("messages.id"), nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="feedback")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(20), default="agent")  # admin, manager, agent
    tenant_id = Column(String(36), nullable=True, index=True)
    api_key = Column(String(64), unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_user_apikey", "api_key"),
    )


class AuditLog(Base):
    """Audit trail for compliance (Gap 14.11)."""
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), nullable=True)
    action = Column(String(50), nullable=False)  # data_export, data_erase, login, etc.
    resource_type = Column(String(30), nullable=True)
    resource_id = Column(String(36), nullable=True)
    details_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
