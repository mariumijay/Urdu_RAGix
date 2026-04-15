"""
Urdu Advanced RAG Pipeline — FastAPI Backend
Endpoints:
  POST /ingest        — Upload + process Urdu OCR text
  POST /query         — Ask a question (streaming or non-streaming)
  GET  /chunks        — Browse indexed chunks
  GET  /health        — System health
  DELETE /index       — Reset all indexes
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ingestion.cleaner  import clean_urdu_text
from ingestion.chunker  import chunk_text
from ingestion.embedder import (
    ingest_chunks, load_metadata, load_faiss_index, load_bm25_index,
    FAISS_PATH, METADATA_PATH, BM25_PATH,
)
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever  import BM25Retriever
from retrieval.hybrid           import reciprocal_rank_fusion
from retrieval.reranker         import rerank
from generation.llm             import stream_answer, generate_answer
from generation.prompt          import build_citations
from models.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    ChunksResponse, ChunkSchema,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global retriever singletons
# ---------------------------------------------------------------------------
faiss_retriever = FAISSRetriever()
bm25_retriever  = BM25Retriever()

# Ingestion status tracker
_ingest_status: dict = {"state": "idle", "message": ""}


# ---------------------------------------------------------------------------
# Lifespan: load existing indexes on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Urdu RAG server...")
    faiss_retriever.load()
    bm25_retriever.load()
    logger.info("Retrievers initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Urdu Advanced RAG API",
    version="1.0.0",
    description="Advanced RAG pipeline for Urdu language books",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: run ingestion in background
# ---------------------------------------------------------------------------
def _run_ingestion(
    raw_text: str,
    book_title: str,
    author: str,
    chapter: str,
    page_start: int,
    chunk_size: int,
    overlap: int,
) -> None:
    global _ingest_status
    try:
        _ingest_status = {"state": "cleaning", "message": "Cleaning OCR text..."}
        clean_text = clean_urdu_text(raw_text)

        _ingest_status = {"state": "chunking", "message": "Chunking text..."}
        chunks = chunk_text(
            clean_text  = clean_text,
            book_title  = book_title,
            author      = author,
            chapter     = chapter,
            page_number = page_start,
            chunk_size  = chunk_size,
            overlap     = overlap,
        )

        if not chunks:
            _ingest_status = {"state": "error", "message": "No chunks produced. Check input text."}
            return

        _ingest_status = {"state": "indexing", "message": f"Indexing {len(chunks)} chunks..."}
        stats = ingest_chunks(chunks)

        # Reload retrievers
        faiss_retriever.load()
        bm25_retriever.load()

        _ingest_status = {
            "state":   "done",
            "message": f"Indexed {stats['chunks_indexed']} chunks successfully.",
            **stats,
        }
        logger.info(f"Background ingestion complete: {stats}")

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        _ingest_status = {"state": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    background_tasks: BackgroundTasks,
    file:        UploadFile = File(..., description="Plain-text Urdu OCR file (.txt)"),
    book_title:  str  = Form(default=""),
    author:      str  = Form(default=""),
    chapter:     str  = Form(default=""),
    page_start:  int  = Form(default=1),
    chunk_size:  int  = Form(default=400),
    overlap:     int  = Form(default=50),
):
    """
    Upload a raw Urdu OCR text file.
    Runs cleaning → chunking → embedding → FAISS+BM25 indexing in background.
    Poll GET /health to monitor progress.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    if len(raw_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="File is too short or empty.")

    global _ingest_status
    _ingest_status = {"state": "starting", "message": "Ingestion queued..."}

    background_tasks.add_task(
        _run_ingestion,
        raw_text, book_title, author, chapter, page_start, chunk_size, overlap,
    )

    return IngestResponse(
        status         = "processing",
        chunks_indexed = 0,
        embedding_dim  = 0,
        faiss_total    = 0,
        message        = "Ingestion started. Poll GET /health for status.",
    )


# ---------------------------------------------------------------------------
# GET /ingest/status
# ---------------------------------------------------------------------------
@app.get("/ingest/status")
async def ingest_status():
    """Check current ingestion progress."""
    return _ingest_status


# ---------------------------------------------------------------------------
# POST /query  (streaming + non-streaming)
# ---------------------------------------------------------------------------
@app.post("/query")
async def query(req: QueryRequest):
    """
    Submit an Urdu question.
    Returns streamed answer (SSE) or full JSON depending on req.stream.
    """
    if not faiss_retriever.is_ready or not bm25_retriever.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. Please ingest a document first via POST /ingest.",
        )

    top_k_dense  = int(os.getenv("TOP_K_DENSE",  "20"))
    top_k_sparse = int(os.getenv("TOP_K_SPARSE", "20"))
    top_k_fuse   = 30
    top_k_final  = req.top_k

    # Stage 1: Hybrid retrieval
    dense_results  = faiss_retriever.search(req.query, top_k=top_k_dense)
    sparse_results = bm25_retriever.search(req.query,  top_k=top_k_sparse)

    # Stage 2: RRF fusion
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        top_n=top_k_fuse,
    )

    # Stage 3: Rerank
    context_chunks = rerank(req.query, fused, top_k=top_k_final)

    if not context_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # Streaming response
    if req.stream:
        async def event_generator():
            # First send citations as a special SSE event
            citations = build_citations(context_chunks)
            yield f"event: citations\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"
            # Then stream answer tokens
            async for token in stream_answer(req.query, context_chunks):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    result = await generate_answer(req.query, context_chunks)
    return QueryResponse(**result)


# ---------------------------------------------------------------------------
# GET /chunks
# ---------------------------------------------------------------------------
@app.get("/chunks", response_model=ChunksResponse)
async def get_chunks(
    page:    int = 0,
    limit:   int = 50,
    search:  str = "",
):
    """Browse all indexed chunks with optional keyword filter."""
    metadata = load_metadata()
    if not metadata:
        return ChunksResponse(total=0, chunks=[])

    # Optional keyword filter
    if search.strip():
        kw = search.strip().lower()
        metadata = [
            m for m in metadata
            if kw in m.get("text", "").lower()
            or kw in m.get("book_title", "").lower()
            or kw in m.get("chapter", "").lower()
        ]

    total = len(metadata)
    paged = metadata[page * limit : (page + 1) * limit]

    chunks = [ChunkSchema(**m) for m in paged]
    return ChunksResponse(total=total, chunks=chunks)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Return system health + ingestion status."""
    faiss_index = load_faiss_index()
    metadata    = load_metadata()
    bm25_index  = load_bm25_index()

    return HealthResponse(
        status          = _ingest_status.get("state", "idle"),
        faiss_vectors   = faiss_index.ntotal if faiss_index else None,
        chunks_count    = len(metadata) if metadata else None,
        bm25_ready      = bm25_index is not None,
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"),
        llm_model       = os.getenv("LLM_MODEL", "gpt-4o"),
    )


# ---------------------------------------------------------------------------
# DELETE /index
# ---------------------------------------------------------------------------
@app.delete("/index")
async def delete_index():
    """Reset all indexes (FAISS, BM25, metadata)."""
    deleted = []
    for path in [FAISS_PATH, METADATA_PATH, BM25_PATH]:
        if Path(path).exists():
            Path(path).unlink()
            deleted.append(str(path))

    # Reload (will be empty)
    faiss_retriever.load()
    bm25_retriever.load()

    global _ingest_status
    _ingest_status = {"state": "idle", "message": "Index reset."}

    return {"status": "deleted", "files": deleted}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
