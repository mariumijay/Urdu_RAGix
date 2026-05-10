from __future__ import annotations

"""
file
main.py  — Unified Urdu RAG CLI
================================
Run after building both index sets:
    python preprocess.py --book data/ocr.txt          # Urdu A → embeddings/urdu_A/
    python -m ingestion.ingest_b data/                # Urdu B → embeddings/urdu_B/
    ollama serve                                       # Qwen model must be running
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
from config.config import GENRE_TO_MODE

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from generation.llm import _create_completion, DEFAULT_MODEL, generate_answer
from generation.prompt_b import get_prompt, detect_intent, build_paper_prompt
from ingestion.embedder import get_dataset_paths
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.query_normalizer import normalize_query
from retrieval.reranker import rerank
from retrieval.router import classify_query, classify_query_full, route_dataset

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",  "20"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "20"))
TOP_K_FINAL  = int(os.getenv("TOP_K_FINAL",  "5"))

RETRIEVAL_CONFIG: dict[str, dict] = {
    # writing genres — faiss-heavy (semantic similarity matters most)
    "application":      {"faiss_k": 1, "bm25_w": 0.4, "faiss_w": 0.6, "do_rerank": True},
    "letter":           {"faiss_k": 1, "bm25_w": 0.4, "faiss_w": 0.6, "do_rerank": True},
    "story":            {"faiss_k": 1, "bm25_w": 0.2, "faiss_w": 0.8, "do_rerank": True},
    "dialogue":         {"faiss_k": 1, "bm25_w": 0.3, "faiss_w": 0.7, "do_rerank": True},
    # tashreeh genres — balanced (need exact verse + semantic context)
    "tashreeh_ghazal":  {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    "tashreeh_nazam":   {"faiss_k": 4, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    "nasar_tashreeh":   {"faiss_k": 2, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    "poem_explanation": {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    # structured long — need more context chunks
    "khulasa":          {"faiss_k": 2, "bm25_w": 0.4, "faiss_w": 0.6, "do_rerank": True},
    "markazi_khyal":    {"faiss_k": 1, "bm25_w": 0.4, "faiss_w": 0.6, "do_rerank": True},
    # objective / one-line — bm25-heavy (keyword exact match matters)
    "mcq":              {"faiss_k": 3, "bm25_w": 0.6, "faiss_w": 0.4, "do_rerank": False},
    "word_meanings":    {"faiss_k": 2, "bm25_w": 0.7, "faiss_w": 0.3, "do_rerank": False},
    "sentence_correction": {"faiss_k": 2, "bm25_w": 0.7, "faiss_w": 0.3, "do_rerank": False},
    "zarbul_imsal":     {"faiss_k": 2, "bm25_w": 0.7, "faiss_w": 0.3, "do_rerank": False},
    # short responses — balanced
    "short_question":   {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    "comprehension":    {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
    "translation":      {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True},
}

# Default config for genres not explicitly listed above
_DEFAULT_RETRIEVAL = {"faiss_k": 3, "bm25_w": 0.5, "faiss_w": 0.5, "do_rerank": True}

def _genre_to_mode(genre: str) -> str:
    return GENRE_TO_MODE.get(genre, "short")

WORD_RANGES: dict[str, tuple[int | None, int | None]] = {
    # one-line / objective — no length check
    "mcq":                 (None, None),
    "word_meanings":       (None, None),
    "sentence_correction": (None, None),
    "zarbul_imsal":        (None, None),
    # short responses — no length check
    "short_question":      (None, None),
    "general_qa":          (None, None),
    "comprehension":       (None, None),
    "translation":         (None, None),
    # tashreeh — soft length targets
    "tashreeh_ghazal":     (120, 150),
    "tashreeh_nazam":      (120, 150),
    "nasar_tashreeh":      (130, 160),
    "poem_explanation":    (100, 130),
    # structured long
    "khulasa":             (250, 350),
    "markazi_khyal":       (80,  120),
    # writing genres — strict length enforced
    "application":         (150, 180),
    "letter":              (150, 200),
    "story":               (180, 220),
    "dialogue":            (150, 200),
}

_B_GENRES = frozenset({
    # one-line / objective
    "mcq", "word_meanings", "sentence_correction", "zarbul_imsal",
    # short
    "short_question", "general_qa", "comprehension", "translation",
    # tashreeh
    "tashreeh_ghazal", "tashreeh_nazam", "nasar_tashreeh", "poem_explanation",
    # structured long
    "khulasa", "markazi_khyal",
    # writing
    "application", "letter", "story", "dialogue",
})


# ── Global state ──────────────────────────────────────────────────────────────

class _State:
    faiss_a:  FAISSRetriever | None = None
    bm25_a:   BM25Retriever  | None = None
    manifest: dict = {}
    ready_a:  bool = False

    faiss_b:  FAISSRetriever | None = None
    bm25_b:   BM25Retriever  | None = None
    ready_b:  bool = False

state = _State()


# ── Urdu formatter ────────────────────────────────────────────────────────────

def format_urdu(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return get_display(arabic_reshaper.reshape(text))

def _dedupe_chunks(chunks: list[dict], similarity_threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate chunks based on text overlap.
    Uses a simple token-set Jaccard similarity to filter semantic duplicates.
    """
    if not chunks:
        return chunks
    
    seen_texts: list[set[str]] = []
    unique: list[dict] = []
    
    for c in chunks:
        text = c.get("text", "").strip()
        if not text:
            continue
        tokens = set(text.split())
        if not tokens:
            continue
        
        is_dup = False
        for prev_tokens in seen_texts:
            overlap = len(tokens & prev_tokens) / max(len(tokens | prev_tokens), 1)
            if overlap >= similarity_threshold:
                is_dup = True
                break
        
        if not is_dup:
            seen_texts.append(tokens)
            unique.append(c)
    
    return unique

# ── Index loading ─────────────────────────────────────────────────────────────

def _load_indexes_a() -> bool:
    paths = get_dataset_paths("urdu_A")
    missing = [k for k in ("faiss", "bm25", "metadata") if not paths[k].exists()]
    if missing:
        print(f"[WARN] Urdu A index files missing: {missing}  (غزل/نظم/نثر/سبق unavailable)")
        return False
    try:
        state.faiss_a = FAISSRetriever("urdu_A")
        state.faiss_a.load()
        state.bm25_a = BM25Retriever("urdu_A")
        state.bm25_a.load()

        manifest_path = paths["faiss"].parent / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                state.manifest = json.load(f)

        state.ready_a = True
        print(
            f"[OK] Urdu A loaded — "
            f"{state.manifest.get('total_chunks', '?')} chunks | "
            f"Book: {state.manifest.get('book_title', 'Unknown')}"
        )
        return True
    except Exception as exc:
        logger.error("Failed to load Urdu A indexes: %s", exc, exc_info=True)
        return False


def _load_indexes_b() -> bool:
    paths = get_dataset_paths("urdu_B")
    if not any(paths[k].exists() for k in ("faiss", "bm25", "metadata")):
        print("[WARN] Urdu B index files not found.  (درخواست/خط/مضمون/کہانی unavailable)")
        return False
    try:
        state.faiss_b = FAISSRetriever("urdu_B")
        ok_faiss = state.faiss_b.load()
        state.bm25_b = BM25Retriever("urdu_B")
        ok_bm25 = state.bm25_b.load()
        state.ready_b = ok_faiss and ok_bm25
        if state.ready_b:
            print(f"[OK] Urdu B loaded — {state.faiss_b.index.ntotal} vectors")
        return state.ready_b
    except Exception as exc:
        logger.error("Failed to load Urdu B indexes: %s", exc, exc_info=True)
        return False


def load_all_indexes() -> bool:
    ok_a = _load_indexes_a()
    ok_b = _load_indexes_b()
    if not ok_a and not ok_b:
        print(
            "\n[ERROR] No indexes loaded.\n"
            "        Run  python preprocess.py --book data/ocr.txt    for Urdu A\n"
            "        Run  python -m ingestion.ingest_b data/          for Urdu B\n"
        )
        return False
    return True


# ── Urdu A pipeline ───────────────────────────────────────────────────────────

def _retrieve_a(urdu_query: str, top_k: int, mode: str = "short") -> tuple[list[dict], list[dict]]:
    dense  = state.faiss_a.search(urdu_query, top_k=TOP_K_DENSE)
    sparse = state.bm25_a.search(urdu_query,  top_k=TOP_K_SPARSE)
    fused  = reciprocal_rank_fusion([dense, sparse], mode=mode)
    ranked = rerank(urdu_query, fused, top_k=top_k)
    ranked = _dedupe_chunks(ranked)
    citations = [
        {
            "chunk_id":   c["chunk_id"],
            "text":       c["text"],
            "chapter":    c.get("chapter"),
            "page_start": c.get("page_start"),
            "score":      round(c.get("rerank_score", c.get("score", 0.0)), 4),
        }
        for c in ranked
    ]
    return ranked, citations


def _print_result_a(result: dict) -> None:
    if "error" in result:
        print(f"\n[خرابی] {result['error']}\n")
        return
    answer = re.sub(r"<think>.*?</think>", "", result["answer"], flags=re.DOTALL)
    answer = re.sub(r"جواب:\s*", "", answer).strip()
    print(format_urdu(answer))


# ── Urdu B pipeline ───────────────────────────────────────────────────────────

def _retrieve_b(urdu_query: str, genre: str, mode: str = "short") -> list[dict]:
    if genre == "short_question":
        cfg = {**RETRIEVAL_CONFIG.get(genre, _DEFAULT_RETRIEVAL), "faiss_k": 8}
    else:
        cfg = RETRIEVAL_CONFIG.get(genre, _DEFAULT_RETRIEVAL)
    faiss_n = max(10, int(cfg["faiss_w"] * 20))
    bm25_n  = max(10, int(cfg["bm25_w"]  * 20))
    dense  = state.faiss_b.search(urdu_query, top_k=faiss_n)
    sparse = state.bm25_b.search(urdu_query,  top_k=bm25_n)
    fused  = reciprocal_rank_fusion([dense, sparse], mode=mode, genre=genre)
    if cfg["do_rerank"]:
        return _dedupe_chunks(rerank(urdu_query, fused, top_k=cfg["faiss_k"]))
    return _dedupe_chunks(fused[:cfg["faiss_k"]])


async def _generate_b(genre: str, chunks: list[dict], query: str) -> str:
    messages = get_prompt(genre, chunks, query)
    response = await _create_completion(
        DEFAULT_MODEL, messages, False, temperature=0.3, max_tokens=2048,
    )
    raw = response.choices[0].message.content
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


async def _validate(genre: str, output: str) -> list[str]:
    min_w, max_w = WORD_RANGES.get(genre, (None, None))
    skip_length  = min_w is None
    length_line  = "3. (length check skipped — always Y for this genre)" if skip_length \
                   else f"3. Is word count between {min_w} and {max_w}? YES/NO"
    length_fmt   = "length:Y" if skip_length else "length:{Y/N}"

    prompt = (
        f"Given this {genre} output, check ONLY:\n"
        f"1. Does it have correct opening structure? YES/NO\n"
        f"2. Does it have correct closing structure? YES/NO\n"
        f"{length_line}\n"
        f"Output format: opening:{{Y/N}} closing:{{Y/N}} {length_fmt}\n"
        f"Text:\n{output}"
    )
    try:
        resp = await _create_completion(
            DEFAULT_MODEL, [{"role": "user", "content": prompt}],
            False, temperature=0.0, max_tokens=30,
        )
        result = resp.choices[0].message.content.strip()
        failing = []
        if "opening:N" in result: failing.append("opening")
        if "closing:N" in result: failing.append("closing")
        if not skip_length and "length:N" in result: failing.append("length")
        sentences = [s.strip() for s in re.split(r'[۔!؟]', output) if len(s.strip()) > 15]
        if len(sentences) != len(set(sentences)):
            failing.append("repetition")
        return failing
    except Exception:
        return []


async def _fix(genre: str, output: str, failing_parts: list[str]) -> str:
    parts_str = " and ".join(failing_parts)
    fix_prompt = (
        f"Fix ONLY the {parts_str} of this {genre}. "
        f"Keep everything else unchanged.\nOriginal:\n{output}"
    )
    try:
        resp = await _create_completion(
            DEFAULT_MODEL, [{"role": "user", "content": fix_prompt}],
            False, temperature=0.2, max_tokens=2048,
        )
        fixed = resp.choices[0].message.content
        return re.sub(r"<think>.*?</think>", "", fixed, flags=re.DOTALL).strip()
    except Exception:
        return output


def _print_result_b(result: dict) -> None:
    if "error" in result:
        print(f"\n[خرابی] {result['error']}\n")
        return
    genre     = result.get("genre", "")
    validated = result.get("validated", True)
    print(f"\n[صنف: {genre}]  {'✓' if validated else '~fixed'}")
    print("-" * 60)
    print(format_urdu(result["answer"]))
    print()


# ── Both-dataset pipeline (merged retrieval) ──────────────────────────────────

def _retrieve_both(urdu_query: str) -> list[dict]:
    result_lists = []
    
    if state.ready_a:
        result_lists += [
            state.faiss_a.search(urdu_query, top_k=3),
            state.bm25_a.search(urdu_query,  top_k=3),
        ]

    if state.ready_b:
        result_lists += [
            state.faiss_b.search(urdu_query, top_k=3),
            state.bm25_b.search(urdu_query,  top_k=3),
        ]
    fused = reciprocal_rank_fusion(result_lists)
    fused = fused[:20]
    fused = _dedupe_chunks(fused)
    return rerank(urdu_query, fused, top_k=4)


# ── Paper helper (shared between fast-path and LLM-path) ──────────────────────

async def _run_paper(urdu_query: str) -> None:
    print("\n[پرچہ ساز] جاری ہے… (6 حصے — تقریباً 2 منٹ)")

    import random

    meta_a = list(state.faiss_a._metadata) if state.ready_a and state.faiss_a._metadata else []
    meta_b = list(state.faiss_b._metadata) if state.ready_b and state.faiss_b._metadata else []

    # ── Genre-targeted filters ─────────────────────────────────────────────
    def by_genre(pool, genres):
        """Return chunks whose 'genre' field matches any of the given genres.
        Falls back to the full pool if no match found."""
        hits = [c for c in pool if c.get("genre", "") in genres]
        return hits if hits else pool

    nazam_pool   = by_genre(meta_a, {"نظم"})
    ghazal_pool  = by_genre(meta_a, {"غزل"})
    sabaq_pool   = by_genre(meta_a, {"نثر", "سبق", "تشریح"})
    khat_pool    = by_genre(meta_b, {"خط", "درخواست"})
    kahani_pool  = by_genre(meta_b, {"کہانی", "مکالمہ", "آپ بیتی", "مضمون"})
    qawaid_pool  = by_genre(meta_b, {"قواعد", "محاورات", "اوقاف"})

    used_ids: set = set()
    # ── Helpers ────────────────────────────────────────────────────────────
    def pick(pool, n):
        """Sample n unique chunks NEVER previously used in this paper."""
        available = [c for c in pool if c.get("chunk_id") not in used_ids]
        n = min(n, len(available))
        chosen = random.sample(available, n) if n > 0 else []
        used_ids.update(c.get("chunk_id") for c in chosen if c.get("chunk_id"))
        return chosen

    def cap(chunks, max_chars=300):
        """Hard-cap each chunk's text to limit input tokens."""
        return [{**c, "text": c.get("text", "")[:max_chars]} for c in chunks]

    # ── Build per-part context ─────────────────────────────────────────────
    # Part 1 — MCQs: mixed Urdu A (sabaq + nazam + ghazal)
    p1 = cap(pick(sabaq_pool, 5) + pick(nazam_pool, 5) + pick(ghazal_pool, 5))

    # Part 2 — Q2 (4 nazam ashaar + 3 ghazal ashaar) + Q3 (nasr passage)
    p2 = cap(pick(nazam_pool, 4) + pick(ghazal_pool, 3) + pick(sabaq_pool, 2))

    # Part 3 — Q4 short questions from sabaq
    p3 = cap(pick(sabaq_pool, 8))

    # Part 4 — Q5 khulasa topics (2 sabaq) + Q6 markazi khyal (1 nazam)
    p4 = cap(pick(sabaq_pool, 2) + pick(nazam_pool, 1))

    # Part 5 — Q7 khat/darkhwast + Q8 kahani/mukalma (Urdu B)
    p5 = cap(pick(khat_pool, 1) + pick(kahani_pool, 1))

    # Part 6 — Q9 qawaid / zarb-ul-amsal
    p6 = cap(pick(qawaid_pool, 2))

    # Fallback: if any part is empty use a mixed sample
    fallback = cap(pick(meta_a, 3) + pick(meta_b, 1))

    part_configs = [
        (1, p1 or fallback, 2000),
        (2, p2 or fallback,  1500),
        (3, p3 or fallback,  1500),
        (4, p4 or fallback,  1200),
        (5, p5 or fallback,  1200),
        (6, p6 or fallback,  1000),
    ]


    for i, (part, p_chunks, max_tok) in enumerate(part_configs):
        print(f"  ⏳ حصہ {part}/6 تیار ہو رہا ہے…")
        messages = build_paper_prompt(urdu_query, p_chunks, part=part)

        response = await _create_completion(
            DEFAULT_MODEL, messages, False,
            temperature=0.6,
            max_tokens=max_tok,
        )

        raw = response.choices[0].message.content
        paper_text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        print(format_urdu(paper_text))
        print("\n" + "-"*40 + "\n")

        if i < len(part_configs) - 1:
            await asyncio.sleep(20)

    print()

