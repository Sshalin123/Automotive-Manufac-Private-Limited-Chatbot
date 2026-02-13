"""
Email Channel Providers for AMPL Chatbot (Gap 14.4).

Supports SendGrid (primary) and AWS SES (fallback).
"""

import logging
from typing import Optional

import httpx

from .base import ChannelProvider, ChannelMessage, ChannelResponse

logger = logging.getLogger(__name__)


class SendGridEmail(ChannelProvider):
    """Email via SendGrid API."""

    BASE_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(self, api_key: str, from_email: str, from_name: str = "AMPL"):
        self.api_key = api_key
        self.from_email = from_email
        self.from_name = from_name

    async def send_message(self, message: ChannelMessage) -> ChannelResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "personalizations": [{"to": [{"email": message.to}]}],
            "from": {"email": self.from_email, "name": self.from_name},
            "subject": message.template_params.get("subject", "AMPL Update") if message.template_params else "AMPL Update",
            "content": [{"type": "text/html", "value": message.content}],
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.BASE_URL, json=payload, headers=headers, timeout=10)
                success = resp.status_code in (200, 202)
                return ChannelResponse(
                    success=success,
                    message_id=resp.headers.get("X-Message-Id"),
                    error=resp.text if not success else None,
                )
        except Exception as e:
            logger.error(f"SendGrid send failed: {e}")
            return ChannelResponse(success=False, error=str(e))

    async def send_template(self, message: ChannelMessage) -> ChannelResponse:
        # For SendGrid, templates are handled via dynamic templates
        return await self.send_message(message)

    async def health_check(self) -> bool:
        return True  # SendGrid doesn't have a simple health endpoint


class SESEmail(ChannelProvider):
    """Email via AWS SES."""

    def __init__(self, region: str = "us-east-1", from_email: str = ""):
        self.region = region
        self.from_email = from_email
        self._client = None

    def _get_client(self):
        if not self._client:
            import boto3
            self._client = boto3.client("ses", region_name=self.region)
        return self._client

    async def send_message(self, message: ChannelMessage) -> ChannelResponse:
        import asyncio
        try:
            client = self._get_client()
            subject = message.template_params.get("subject", "AMPL Update") if message.template_params else "AMPL Update"
            resp = await asyncio.to_thread(
                client.send_email,
                Source=self.from_email,
                Destination={"ToAddresses": [message.to]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Html": {"Data": message.content}},
                },
            )
            return ChannelResponse(success=True, message_id=resp.get("MessageId"))
        except Exception as e:
            logger.error(f"SES send failed: {e}")
            return ChannelResponse(success=False, error=str(e))

    async def send_template(self, message: ChannelMessage) -> ChannelResponse:
        return await self.send_message(message)

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.get_send_quota()
            return True
        except Exception:
            return False


class EmailRouter:
    """Routes emails through primary or fallback provider."""

    def __init__(self, primary: ChannelProvider, fallback: Optional[ChannelProvider] = None):
        self.primary = primary
        self.fallback = fallback

    async def send(self, message: ChannelMessage) -> ChannelResponse:
        result = await self.primary.send_message(message)
        if not result.success and self.fallback:
            logger.warning("Primary email failed, trying fallback")
            result = await self.fallback.send_message(message)
        return result
