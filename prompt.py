"""
Urdu RAG Prompt Templates
Builds structured prompts for the LLM with retrieved Urdu context.
"""

SYSTEM_PROMPT = """آپ ایک ماہر اردو علمی مددگار ہیں۔ آپ کا کام صرف فراہم کردہ اقتباسات کی بنیاد پر سوالات کے جوابات دینا ہے۔

اہم ہدایات:
- صرف اقتباسات میں موجود معلومات استعمال کریں
- اگر جواب اقتباسات میں موجود نہیں تو صاف کہیں: "اس سوال کا جواب فراہم کردہ متن میں موجود نہیں"
- جواب واضح، مختصر اور اردو میں دیں
- اگر ممکن ہو تو ماخذ (صفحہ نمبر) کا حوالہ دیں"""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for the OpenAI Chat API.

    Args:
        query:          User's Urdu question.
        context_chunks: Top-k reranked chunk dicts (must have 'text', 'page_start').

    Returns:
        List of message dicts for openai.chat.completions.create(messages=...).
    """
    # Build context block
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk.get("page_start", "؟")
        text = chunk.get("text", "").strip()
        context_parts.append(f"اقتباس {i} (صفحہ {page}):\n{text}")

    context_block = "\n\n".join(context_parts)

    user_message = f"""اقتباسات:
{context_block}

سوال: {query}

جواب:"""

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]


def build_citations(context_chunks: list[dict]) -> list[dict]:
    """Build citation metadata list from context chunks."""
    citations = []
    for chunk in context_chunks:
        citations.append({
            "chunk_id":   chunk.get("chunk_id", ""),
            "page_start": chunk.get("page_start", 0),
            "book_title": chunk.get("book_title", ""),
            "chapter":    chunk.get("chapter", ""),
            "text_preview": chunk.get("text", "")[:120] + "…",
        })
    return citations
