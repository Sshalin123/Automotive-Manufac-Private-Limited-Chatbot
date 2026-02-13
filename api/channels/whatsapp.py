"""
WhatsApp Channel Providers for AMPL Chatbot (Gap 14.4).

Supports Meta Cloud API (primary) and Gupshup (fallback).
"""

import logging
from typing import Optional

import httpx

from .base import ChannelProvider, ChannelMessage, ChannelResponse

logger = logging.getLogger(__name__)


class MetaCloudWhatsApp(ChannelProvider):
    """WhatsApp via Meta Cloud API."""

    BASE_URL = "https://graph.facebook.com/v18.0"

    def __init__(self, api_token: str, phone_number_id: str):
        self.api_token = api_token
        self.phone_number_id = phone_number_id

    async def send_message(self, message: ChannelMessage) -> ChannelResponse:
        url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        payload = {
            "messaging_product": "whatsapp",
            "to": message.to,
            "type": "text",
            "text": {"body": message.content},
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                msg_id = data.get("messages", [{}])[0].get("id")
                return ChannelResponse(success=True, message_id=msg_id)
        except Exception as e:
            logger.error(f"Meta WhatsApp send failed: {e}")
            return ChannelResponse(success=False, error=str(e))

    async def send_template(self, message: ChannelMessage) -> ChannelResponse:
        url = f"{self.BASE_URL}/{self.phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        components = []
        if message.template_params:
            components.append({
                "type": "body",
                "parameters": [
                    {"type": "text", "text": v}
                    for v in message.template_params.values()
                ],
            })
        payload = {
            "messaging_product": "whatsapp",
            "to": message.to,
            "type": "template",
            "template": {
                "name": message.template_id,
                "language": {"code": "en"},
                "components": components,
            },
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                msg_id = data.get("messages", [{}])[0].get("id")
                return ChannelResponse(success=True, message_id=msg_id)
        except Exception as e:
            logger.error(f"Meta WhatsApp template send failed: {e}")
            return ChannelResponse(success=False, error=str(e))

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.BASE_URL}/{self.phone_number_id}",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    timeout=5,
                )
                return resp.status_code == 200
        except Exception:
            return False


class GupshupWhatsApp(ChannelProvider):
    """WhatsApp via Gupshup API (fallback)."""

    BASE_URL = "https://api.gupshup.io/sm/api/v1"

    def __init__(self, api_key: str, app_name: str):
        self.api_key = api_key
        self.app_name = app_name

    async def send_message(self, message: ChannelMessage) -> ChannelResponse:
        url = f"{self.BASE_URL}/msg"
        payload = {
            "channel": "whatsapp",
            "source": self.app_name,
            "destination": message.to,
            "message": {"type": "text", "text": message.content},
            "src.name": self.app_name,
        }
        headers = {"apikey": self.api_key, "Content-Type": "application/x-www-form-urlencoded"}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, data=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return ChannelResponse(
                    success=data.get("status") == "submitted",
                    message_id=data.get("messageId"),
                )
        except Exception as e:
            logger.error(f"Gupshup send failed: {e}")
            return ChannelResponse(success=False, error=str(e))

    async def send_template(self, message: ChannelMessage) -> ChannelResponse:
        # Gupshup template sending uses same endpoint with different payload
        return await self.send_message(message)

    async def health_check(self) -> bool:
        return True  # No simple health endpoint for Gupshup


class WhatsAppRouter:
    """Routes WhatsApp messages through primary or fallback provider."""

    def __init__(self, primary: ChannelProvider, fallback: Optional[ChannelProvider] = None):
        self.primary = primary
        self.fallback = fallback

    async def send(self, message: ChannelMessage) -> ChannelResponse:
        result = await self.primary.send_message(message)
        if not result.success and self.fallback:
            logger.warning("Primary WhatsApp failed, trying fallback")
            result = await self.fallback.send_message(message)
        return result
