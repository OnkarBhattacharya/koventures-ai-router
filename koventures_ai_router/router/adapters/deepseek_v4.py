"""Thin OpenAI-compatible client for DeepSeek V4."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 30.0


class DeepSeekV4Client:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        user_message: str,
        system_prompt: str = (
            "You are a helpful educational AI assistant for school students aged 8-18. "
            "Provide clear, accurate, curriculum-appropriate answers."
        ),
        max_length: int = 300,
        temperature: float = 0.7,
        **_: Any,
    ) -> Optional[str]:
        if not self.api_key:
            logger.error("DeepSeekV4Client requires DEEPSEEK_API_KEY")
            return None

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_length,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("DeepSeek V4 generation failed: %s", exc)
            return None


def generate(
    user_message: str,
    system_prompt: str = (
        "You are a helpful educational AI assistant for school students aged 8-18."
    ),
    max_length: int = 300,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Optional[str]:
    client = DeepSeekV4Client(
        api_key=kwargs.get("api_key"),
        base_url=kwargs.get("base_url", DEFAULT_BASE_URL),
        model=kwargs.get("model", DEFAULT_MODEL),
    )
    return client.generate(user_message, system_prompt, max_length, temperature)
