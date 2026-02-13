"""
AWS Bedrock LLM Provider.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockProvider:
    """
    AWS Bedrock LLM provider.

    Supports Claude models via Bedrock.
    """

    DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        region: str = "us-east-1",
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize Bedrock provider.

        Args:
            model_id: Bedrock model ID
            region: AWS region
            max_tokens: Maximum tokens for response
            temperature: Generation temperature
        """
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client = boto3.client("bedrock-runtime", region_name=region)
        logger.info(f"Bedrock provider initialized: {model_id} in {region}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Stop sequences

        Returns:
            Generated response
        """
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            }

            if system:
                body["system"] = system

            if stop_sequences:
                body["stop_sequences"] = stop_sequences

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            if "content" in response_body and response_body["content"]:
                return response_body["content"][0]["text"].strip()

            logger.warning("Empty response from Bedrock")
            return ""

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Bedrock generation failed: {e}")
            raise

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> str:
        """
        Generate response with conversation history.

        Args:
            messages: List of messages with role and content
            system: System prompt

        Returns:
            Generated response
        """
        try:
            # Format messages for Claude
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages
            }

            if system:
                body["system"] = system

            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            if "content" in response_body and response_body["content"]:
                return response_body["content"][0]["text"].strip()

            return ""

        except Exception as e:
            logger.error(f"Bedrock generation with history failed: {e}")
            raise

    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> str:
        """Async wrapper for generate."""
        # Bedrock doesn't have native async, so this is sync
        return self.generate(prompt, system)

    def health_check(self) -> bool:
        """Check if Bedrock is available."""
        try:
            self.generate("Hello", max_tokens=10)
            return True
        except:
            return False
