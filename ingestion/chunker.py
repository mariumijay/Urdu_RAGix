"""
Urdu Sentence-Aware Chunker
Splits clean Urdu text into overlapping chunks that respect sentence
boundaries. Each chunk carries rich metadata for retrieval.
"""

import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


# Urdu sentence-terminal punctuation: ۔  ؟  !
SENTENCE_SPLITTER = re.compile(r"(?<=[۔؟!])\s+")

# Approximate token count: split on whitespace (Urdu words separated by spaces)
def _token_count(text: str) -> int:
    return len(text.split())


@dataclass
class Chunk:
    chunk_id:   str
    text:       str
    token_count: int
    book_title: str
    author:     str
    page_start: int
    page_end:   int
    chapter:    str
    position:   int          # sequential index among all chunks

    def to_dict(self) -> dict:
        return asdict(self)


def _split_into_sentences(text: str) -> list[str]:
    """Split Urdu text into individual sentences."""
    sentences = SENTENCE_SPLITTER.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    clean_text: str,
    book_title: str = "",
    author: str = "",
    chapter: str = "",
    page_number: int = 1,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Split clean Urdu text into overlapping, sentence-boundary-respecting chunks.

    Args:
        clean_text:  Normalized Urdu text (output of cleaner).
        book_title:  Source book title.
        author:      Book author.
        chapter:     Chapter name/number.
        page_number: Starting page number for this text block.
        chunk_size:  Target max tokens per chunk.
        overlap:     Overlap in tokens between consecutive chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    sentences = _split_into_sentences(clean_text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    position = 0
    i = 0  # sentence cursor

    while i < len(sentences):
        current_tokens = 0
        current_sentences: list[str] = []

        # Fill chunk up to chunk_size tokens
        while i < len(sentences):
            s_tokens = _token_count(sentences[i])
            if current_tokens + s_tokens > chunk_size and current_sentences:
                break
            current_sentences.append(sentences[i])
            current_tokens += s_tokens
            i += 1

        chunk_text_str = " ".join(current_sentences).strip()
        if not chunk_text_str:
            continue

        chunk = Chunk(
            chunk_id    = str(uuid.uuid4()),
            text        = chunk_text_str,
            token_count = current_tokens,
            book_title  = book_title,
            author      = author,
            page_start  = page_number,
            page_end    = page_number,
            chapter     = chapter,
            position    = position,
        )
        chunks.append(chunk)
        position += 1

        # Overlap: step back by sentences that cover ~overlap tokens
        overlap_tokens = 0
        step_back = 0
        for s in reversed(current_sentences):
            overlap_tokens += _token_count(s)
            step_back += 1
            if overlap_tokens >= overlap:
                break
        i -= step_back  # rewind cursor for overlap

        # Safety: if we rewound too far, advance at least one sentence
        if step_back >= len(current_sentences):
            i += 1

    return chunks


def chunks_from_pages(
    pages: list[dict],
    book_title: str = "",
    author: str = "",
    chapter: str = "",
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Convenience: build chunks from a list of page dicts.
    Each page dict: {"page_number": int, "text": str}
    Page metadata is attached to each chunk.
    """
    all_chunks: list[Chunk] = []
    position_counter = 0

    for page in pages:
        page_num = page.get("page_number", 1)
        text     = page.get("text", "")
        if not text.strip():
            continue

        page_chunks = chunk_text(
            clean_text  = text,
            book_title  = book_title,
            author      = author,
            chapter     = chapter,
            page_number = page_num,
            chunk_size  = chunk_size,
            overlap     = overlap,
        )
        for c in page_chunks:
            c.position  = position_counter
            c.page_start = page_num
            c.page_end   = page_num
            position_counter += 1

        all_chunks.extend(page_chunks)

    return all_chunks
