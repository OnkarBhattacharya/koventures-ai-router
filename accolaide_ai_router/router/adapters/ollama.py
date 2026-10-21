"""Thin client for self-hosted Ollama (offline backstop, e.g. Qwen 2.5 3B)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
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
        try:
            import requests as req

            endpoint = f"{self.base_url}/api/chat"
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "options": {
                    "num_predict": max_length,
                    "temperature": temperature,
                },
                "stream": False,
            }

            if image_data:
                payload["messages"][-1]["images"] = [image_data]

            resp = req.post(endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            text = message.get("content", "")
            return text.strip() if text else None
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            return None
