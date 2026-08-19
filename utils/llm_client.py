"""
utils/llm_client.py

Thin wrapper around the OpenAI SDK used by every AI-facing module in the
codebase (AI_generator, AI_analysis, creator_profile, content_similarity_check,
email_assistant). Centralising this means the model names, retry behaviour,
and JSON-parsing logic only live in one place.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional

from openai import OpenAI

from config.settings import settings
from utils.logger import logger


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before calling the LLM client."
        )
    return OpenAI(api_key=settings.openai_api_key)


def chat_complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    """Single chat completion, returns raw text content."""
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def chat_complete_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Dict:
    """
    Chat completion that expects (and enforces) a JSON object response.
    Uses OpenAI's response_format=json_object mode, then defensively
    parses the result. Returns {} on parse failure rather than raising,
    so a single bad generation doesn't crash a batch pipeline.
    """
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=model or settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("chat_complete_json: model did not return valid JSON: %s", raw[:300])
        return {}
    except Exception as exc:  # noqa: BLE001 — surface any API error, don't crash caller
        logger.error("chat_complete_json failed: %s", exc)
        return {}


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Returns a single embedding vector for the given text."""
    if not text or not text.strip():
        return []
    client = get_client()
    response = client.embeddings.create(
        model=model or settings.openai_embedding_model,
        input=text[:8000],  # guard against oversized inputs
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Batched embedding call — cheaper than one call per item."""
    texts = [t[:8000] for t in texts if t and t.strip()]
    if not texts:
        return []
    client = get_client()
    response = client.embeddings.create(model=model or settings.openai_embedding_model, input=texts)
    return [d.embedding for d in response.data]
