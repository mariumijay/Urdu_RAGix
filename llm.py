"""
LLM Interface — OpenAI streaming wrapper
Compatible with openai>=1.0.0
"""

import os
import json
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from generation.prompt import build_rag_prompt, build_citations

logger = logging.getLogger(__name__)
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def stream_answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream answer tokens one by one."""
    model = model or os.getenv("LLM_MODEL", "gpt-4o")
    messages = build_rag_prompt(query, context_chunks)
    client = get_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        yield f"\n\nخرابی: {str(e)}"


async def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> dict:
    """Non-streaming: return full answer + citations."""
    model = model or os.getenv("LLM_MODEL", "gpt-4o")
    messages = build_rag_prompt(query, context_chunks)
    client = get_client()

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    answer    = response.choices[0].message.content or ""
    citations = build_citations(context_chunks)

    return {
        "answer":    answer,
        "citations": citations,
        "model":     model,
        "usage":     response.usage.model_dump() if response.usage else {},
    }
