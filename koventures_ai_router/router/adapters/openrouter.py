"""Thin OpenAI-compatible client for OpenRouter (fast-path fallback for Groq and other models)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Safety net only — the router always passes an explicit model id.
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        user_message: str,
        system_prompt: str = (
            "You are a helpful educational AI assistant for school students aged 8-18. "
            "Provide clear, accurate, curriculum-appropriate answers."
        ),
        max_length: int = 300,
        temperature: float = 0.7,
        image_data: Optional[str] = None,
        **_: Any,
    ) -> Optional[str]:
        if not self.api_key:
            logger.error("OpenRouterClient requires OPENROUTER_API_KEY")
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_length,
                "temperature": temperature,
            }

            if image_data:
                messages[-1] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }

            resp = client.chat.completions.create(**request_kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("OpenRouter generation failed: %s", exc)
            return None
