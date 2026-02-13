"""
OpenAI LLM Provider.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI LLM provider.

    Supports GPT-4 and GPT-3.5 models.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_id: Model ID
            max_tokens: Maximum tokens
            temperature: Generation temperature
        """
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        logger.info(f"OpenAI provider initialized: {model_id}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response from prompt.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated response
        """
        try:
            messages = []

            if system:
                messages.append({"role": "system", "content": system})

            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None
    ) -> str:
        """
        Generate response with conversation history.

        Args:
            messages: List of messages
            system: System prompt

        Returns:
            Generated response
        """
        try:
            formatted = []

            if system:
                formatted.append({"role": "system", "content": system})

            formatted.extend(messages)

            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=formatted,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation with history failed: {e}")
            raise

    async def agenerate(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> str:
        """Async generation using OpenAI async client."""
        try:
            from openai import AsyncOpenAI
            async_client = AsyncOpenAI()

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await async_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check if OpenAI is available."""
        try:
            self.generate("Hello", max_tokens=10)
            return True
        except:
            return False
