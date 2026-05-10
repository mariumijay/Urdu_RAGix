"""
api.py — FastAPI REST API for the Urdu RAG System
Exposes: POST /query, POST /ingest, GET /chunks, GET /health
"""

from __future__ import annotations
from generation.prompt_b import detect_intent  # add this import at top
from config.config import GENRE_TO_MODE 
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.embedder import get_dataset_paths, get_embedding_model, ingest_chunks
from generation.llm import generate_answer, stream_answer, DEFAULT_MODEL
from models.schemas import (
    QueryRequest,
    QueryResponse,
    CitationSchema,
    ChunkSchema,
    ChunksResponse,
    HealthResponse,
    IngestResponse,
    SUPPORTED_DATASETS,
)
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.query_normalizer import normalize_query
from retrieval.reranker import rerank
from retrieval.router import classify_query_full

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",  "20"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "20"))
TOP_K_FINAL  = int(os.getenv("TOP_K_FINAL",  "5"))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Urdu RAG API",
    description="REST API for the Unified Urdu RAG System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

class _State:
    faiss_a:  FAISSRetriever | None = None
    bm25_a:   BM25Retriever  | None = None
    ready_a:  bool = False

    faiss_b:  FAISSRetriever | None = None
    bm25_b:   BM25Retriever  | None = None
    ready_b:  bool = False

state = _State()


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    _load_indexes_a()
    _load_indexes_b()


def _load_indexes_a() -> bool:
    paths = get_dataset_paths("urdu_A")
    missing = [k for k in ("faiss", "bm25", "metadata") if not paths[k].exists()]
    if missing:
        logger.warning("Urdu A index files missing: %s", missing)
        return False
    try:
        state.faiss_a = FAISSRetriever("urdu_A")
        state.faiss_a.load()
        state.bm25_a = BM25Retriever("urdu_A")
        state.bm25_a.load()
        state.ready_a = True
        logger.info("Urdu A indexes loaded.")
        return True
    except Exception as exc:
        logger.error("Failed to load Urdu A indexes: %s", exc)
        return False


def _load_indexes_b() -> bool:
    paths = get_dataset_paths("urdu_B")
    if not any(paths[k].exists() for k in ("faiss", "bm25", "metadata")):
        logger.warning("Urdu B index files not found.")
        return False
    try:
        state.faiss_b = FAISSRetriever("urdu_B")
        ok_faiss = state.faiss_b.load()
        state.bm25_b = BM25Retriever("urdu_B")
        ok_bm25 = state.bm25_b.load()
        state.ready_b = ok_faiss and ok_bm25
        if state.ready_b:
            logger.info("Urdu B indexes loaded.")
        return state.ready_b
    except Exception as exc:
        logger.error("Failed to load Urdu B indexes: %s", exc)
        return False


# ── Retrieval helper ──────────────────────────────────────────────────────────

def _retrieve(urdu_query: str, top_k: int) -> list[dict]:
    result_lists = []
    if state.ready_a:
        result_lists += [
            state.faiss_a.search(urdu_query, top_k=TOP_K_DENSE),
            state.bm25_a.search(urdu_query,  top_k=TOP_K_SPARSE),
        ]
    if state.ready_b:
        result_lists += [
            state.faiss_b.search(urdu_query, top_k=TOP_K_DENSE),
            state.bm25_b.search(urdu_query,  top_k=TOP_K_SPARSE),
        ]
    if not result_lists:
        return []
    fused = reciprocal_rank_fusion(result_lists)
    return rerank(urdu_query, fused, top_k=top_k)


# ── POST /query ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not state.ready_a and not state.ready_b:
        raise HTTPException(status_code=503, detail="No indexes loaded. Run ingestion first.")

    urdu_query, _ = normalize_query(req.query)
    await classify_query_full(urdu_query)
    
    # ── Detect genre + mode ──────────────────────────────
    genre = detect_intent(urdu_query)
    mode  = GENRE_TO_MODE.get(genre, "short")
    
    chunks = _retrieve(urdu_query, top_k=req.top_k)

    if req.stream:
        async def event_stream():
            async for char in stream_answer(urdu_query, chunks, mode=mode, genre=genre):
                yield char
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await generate_answer(urdu_query, chunks, mode=mode, genre=genre)

    citations = [
        CitationSchema(
            chunk_id     = c.get("chunk_id", ""),
            page_start   = c.get("page_start", 1),
            book_title   = c.get("book_title", ""),
            chapter      = c.get("chapter", ""),
            text_preview = c.get("text", "")[:120],
        )
        for c in chunks
    ]

    return QueryResponse(
        answer    = result["answer"],
        citations = citations,
        model     = result["model"],
        usage     = result.get("usage", {}),
    )


# ── POST /ingest ──────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file:        UploadFile = File(...),
    book_title:  str        = Form(default=""),
    author:      str        = Form(default=""),
    chapter:     str        = Form(default=""),
    page_start:  int        = Form(default=1),
    chunk_size:  int        = Form(default=400),
    overlap:     int        = Form(default=50),
    dataset:     str        = Form(default="urdu_A"),
):
    # Validate dataset before touching any files
    if dataset not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid dataset '{dataset}'. "
                f"Supported datasets are: {sorted(SUPPORTED_DATASETS)}"
            ),
        )

    try:
        raw_bytes = await file.read()
        raw_text  = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}")

    cleaned = clean_text(raw_text)
    chunks  = chunk_text(
        clean_text = cleaned,
        book_title = book_title,
        author     = author,
        chapter    = chapter,
        page_number= page_start,
        chunk_size = chunk_size,
        overlap    = overlap,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from the uploaded file.")

    # Ingest into the validated target dataset
    stats = ingest_chunks(chunks, dataset=dataset)

    # Reload the indexes for the affected dataset immediately
    if dataset == "urdu_A":
        _load_indexes_a()
    elif dataset == "urdu_B":
        _load_indexes_b()

    return IngestResponse(
        status         = "ok",
        chunks_indexed = stats.get("chunks_indexed", len(chunks)),
        embedding_dim  = stats.get("embedding_dim", 0),
        faiss_total    = stats.get("faiss_total", 0),
        message        = f"Ingested {len(chunks)} chunks into dataset '{dataset}'.",
    )


# ── GET /chunks ───────────────────────────────────────────────────────────────

@app.get("/chunks", response_model=ChunksResponse)
async def chunks_endpoint():
    all_chunks: list[ChunkSchema] = []

    for dataset in ("urdu_A", "urdu_B"):
        meta_path = Path("embeddings") / dataset / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                all_chunks.append(
                    ChunkSchema(
                        chunk_id    = str(r.get("chunk_id", "")),
                        text        = r.get("text", ""),
                        token_count = int(r.get("token_count", 0)),
                        book_title  = r.get("book_title", r.get("source", "")),
                        author      = r.get("author", ""),
                        page_start  = int(r.get("page_start", 1)),
                        page_end    = int(r.get("page_end", r.get("page_start", 1))),
                        chapter     = r.get("chapter", r.get("genre", "")),
                        position    = int(r.get("position", r.get("faiss_index", 0))),
                    )
                )
        except Exception as exc:
            logger.error("Failed to load metadata for %s: %s", dataset, exc)

    return ChunksResponse(total=len(all_chunks), chunks=all_chunks)


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    faiss_vectors = 0
    if state.ready_a and state.faiss_a and state.faiss_a.index:
        faiss_vectors += state.faiss_a.index.ntotal
    if state.ready_b and state.faiss_b and state.faiss_b.index:
        faiss_vectors += state.faiss_b.index.ntotal

    chunks_count = 0
    for dataset in ("urdu_A", "urdu_B"):
        meta_path = Path("embeddings") / dataset / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                chunks_count += len(data)
            except Exception:
                pass

    bm25_ready = (
        (state.ready_a and state.bm25_a is not None and state.bm25_a.is_ready)
        or (state.ready_b and state.bm25_b is not None and state.bm25_b.is_ready)
    )

    embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    llm_model       = DEFAULT_MODEL

    return HealthResponse(
        status          = "ok",
        faiss_vectors   = faiss_vectors,
        chunks_count    = chunks_count,
        bm25_ready      = bm25_ready,
        embedding_model = embedding_model,
        llm_model       = llm_model,
    )