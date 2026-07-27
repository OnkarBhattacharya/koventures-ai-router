"""HuggingFace hosted adapter: lifts _generate_via_api and _generate_local from inference_service."""

from __future__ import annotations

import html
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_CASUAL_SEPARATOR = "\n\n### Assistant:\n"
_SEQ2SEQ_PREFIX = "You are an educational assistant. Answer clearly and helpfully.\n"


def _make_causal_prompt(user_message: str, system_prompt: str, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, KeyError, AttributeError) as exc:
            logger.debug("apply_chat_template failed, using fallback prompt: %s", exc)
    return (
        f"{html.escape(system_prompt)}\n\n"
        f"User: {html.escape(user_message)}{_CASUAL_SEPARATOR}"
    )


def _make_seq2seq_prompt(user_message: str, system_prompt: str) -> str:
    return f"{_SEQ2SEQ_PREFIX}Question: {html.escape(user_message)}\nAnswer:"


class HFHostedAdapter:
    """Thin wrapper around a HuggingFace InferenceClient + optional local torch model."""

    def __init__(
        self,
        api_client: Any = None,
        api_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_token: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
        torch: Any = None,
    ):
        self.api_client = api_client
        self.api_model_id = api_model_id.strip("/") if api_model_id else ""
        self.api_token = api_token
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch

    def generate(
        self,
        user_message: str,
        max_length: int = 300,
        max_retries: int = 3,
        system_prompt: str = (
            "You are a helpful educational AI assistant for school students aged 8-18. "
            "Provide clear, accurate, curriculum-appropriate answers. "
            "Explain concepts step by step in a way students can understand. "
            "Guide students to understanding rather than just giving answers. "
            "Keep responses concise and age-appropriate."
        ),
        image_data: Optional[str] = None,
    ) -> Optional[str]:
        if self.api_client and self.api_token:
            return self._generate_via_api(
                user_message, max_length, max_retries, system_prompt, image_data
            )
        if self.model is not None and self.tokenizer is not None and self.torch is not None:
            return self._generate_local(
                user_message, max_length, system_prompt, image_data
            )
        return None

    def _generate_via_api(
        self,
        user_message: str,
        max_length: int,
        max_retries: int,
        system_prompt: str,
        image_data: Optional[str],
    ) -> Optional[str]:
        model_id = self.api_model_id

        for attempt in range(max_retries):
            try:
                try:
                    content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
                    if image_data:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        })

                    result = self.api_client.chat_completion(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content},
                        ],
                        max_tokens=max_length,
                        temperature=0.7,
                    )
                    return result.choices[0].message.content.strip()
                except Exception as chat_err:
                    logger.debug(
                        "chat_completion not supported (%s), trying text_generation",
                        chat_err,
                    )
                    prompt = f"[INST] {system_prompt}\n\n{user_message} [/INST]"
                    result = self.api_client.text_generation(
                        prompt,
                        model=model_id,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        return_full_text=False,
                    )
                    return result.strip()

            except Exception as exc:
                err = str(exc)
                if "404" in err or "not found" in err.lower():
                    logger.error("Model not found via API: %s", model_id)
                    return None
                if "401" in err:
                    logger.error("Invalid HF_API_TOKEN.")
                    return None
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 10)
                    logger.warning(
                        "API error (%s). Retrying in %ss… (%s/%s)",
                        err, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                else:
                    logger.error("API failed after %s retries: %s", max_retries, err)
                    return None

        return None

    def _generate_local(
        self,
        user_message: str,
        max_length: int,
        system_prompt: str,
        image_data: Optional[str],
    ) -> Optional[str]:
        if image_data:
            logger.warning(
                "Local vision inference not yet optimized for CPU. "
                "Falling back to text description if available."
            )

        from transformers import AutoModelForCausalLM
        is_causal = isinstance(self.model, AutoModelForCausalLM)

        if is_causal:
            prompt = _make_causal_prompt(user_message, system_prompt, self.tokenizer)
            with self.torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                input_len = inputs["input_ids"].shape[1]
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                new_tokens = outputs[0][input_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            prompt = _make_seq2seq_prompt(user_message, system_prompt)
            with self.torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        logger.info("Local model response (%s chars): %s…", len(response), response[:80])
        return response if response else None
