"""
Abstract Channel Provider for AMPL Chatbot (Gap 14.4).

Base class for all messaging channel integrations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChannelMessage:
    """Message to send via a channel."""
    to: str  # Phone number or email
    content: str
    template_id: Optional[str] = None
    template_params: Optional[Dict[str, str]] = None
    media_url: Optional[str] = None


@dataclass
class ChannelResponse:
    """Response from channel send operation."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class ChannelProvider(ABC):
    """Abstract base class for messaging channels."""

    @abstractmethod
    async def send_message(self, message: ChannelMessage) -> ChannelResponse:
        """Send a text message."""
        ...

    @abstractmethod
    async def send_template(self, message: ChannelMessage) -> ChannelResponse:
        """Send a template message."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if channel is operational."""
        ...
