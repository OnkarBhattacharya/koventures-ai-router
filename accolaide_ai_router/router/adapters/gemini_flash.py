"""Thin client for Google Gemini Flash (primary provider)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiFlashClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
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
            logger.error("GeminiFlashClient requires GOOGLE_API_KEY or GEMINI_API_KEY")
            return None

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            contents: Any = None
            if image_data:
                try:
                    import base64

                    image_bytes = base64.b64decode(image_data)
                    from PIL import Image
                    import io

                    image = Image.open(io.BytesIO(image_bytes))
                    contents = [
                        system_prompt,
                        image,
                        user_message,
                    ]
                except Exception:
                    contents = f"{system_prompt}\n\nUser: {user_message}"
            else:
                contents = f"{system_prompt}\n\nUser: {user_message}"

            generation_config = {
                "max_output_tokens": max_length,
                "temperature": temperature,
            }
            response = model.generate_content(
                contents,
                generation_config=generation_config,
                request_options={"timeout": 30},
            )
            text = getattr(response, "text", "")
            return text.strip() if text else None
        except Exception as exc:
            logger.error("Gemini Flash generation failed: %s", exc)
            return None
