"""
LLM Interface — Google Gemini wrapper
Uses the current google-genai SDK (google.genai)
"""

import os
import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

from generation.prompt import build_rag_prompt, build_citations

logger = logging.getLogger(__name__)
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate system message from user/assistant messages."""
    system = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            role = "model" if m["role"] == "assistant" else "user"
            user_messages.append({"role": role, "parts": [{"text": m["content"]}]})
    return system, user_messages


async def stream_answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream answer tokens one by one."""
    model_name = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")
    messages = build_rag_prompt(query, context_chunks)
    system, contents = _split_messages(messages)
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=1024,
        temperature=0.2,
    )

    try:
        async for chunk in client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error(f"Gemini streaming error: {e}")
        yield f"\n\nخرابی: {str(e)}"


async def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> dict:
    """Non-streaming: return full answer + citations."""
    model_name = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")
    messages = build_rag_prompt(query, context_chunks)
    system, contents = _split_messages(messages)
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=1024,
        temperature=0.2,
    )

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    answer = response.text or ""
    citations = build_citations(context_chunks)

    usage = {}
    if response.usage_metadata:
        usage = {
            "input_tokens":  response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }

    return {
        "answer":    answer,
        "citations": citations,
        "model":     model_name,
        "usage":     usage,
    }
