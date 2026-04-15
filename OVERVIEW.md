# Urdu Advanced RAG — Implementation Overview

## What It Does

A Retrieval-Augmented Generation (RAG) pipeline built specifically for Urdu books/OCR text.
You upload a `.txt` Urdu file, it gets indexed, then you ask questions in Urdu and get cited answers via Google Gemini.

---

## Architecture at a Glance

```
Upload .txt
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────────────┐
│   Cleaner   │────▶│   Chunker    │────▶│        Embedder          │
│ (normalize  │     │ (sentence-   │     │ multilingual-e5-large    │
│  Urdu OCR)  │     │  boundary    │     │ → FAISS index (dense)    │
│             │     │  overlap)    │     │ → BM25 index  (sparse)   │
└─────────────┘     └──────────────┘     └──────────────────────────┘

Query (Urdu)
     │
     ▼
┌────────────┐   ┌────────────┐
│   FAISS    │   │    BM25    │   ← Stage 1: Hybrid retrieval (top-20 each)
│  (dense)   │   │  (sparse)  │
└─────┬──────┘   └─────┬──────┘
      └────────┬────────┘
               ▼
        ┌─────────────┐
        │  RRF Fusion │   ← Stage 2: Reciprocal Rank Fusion → top-30
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │  Reranker   │   ← Stage 3: Cross-encoder ms-marco-MiniLM → top-k
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │   Gemini    │   ← Stage 4: Answer generation with citations
        └─────────────┘
```

---

## Module Breakdown

### Ingestion (`ingestion/`)

| File | Role |
|------|------|
| `cleaner.py` | NFC normalization, fixes Urdu char variants (ي→ی, ك→ک), strips diacritics, OCR noise, page numbers |
| `chunker.py` | Splits on Urdu sentence endings (`۔ ؟ !`), produces overlapping chunks with token-count control |
| `embedder.py` | Embeds chunks with `intfloat/multilingual-e5-large`, saves FAISS + BM25 + metadata JSON to `storage/` |

**Chunk metadata stored per chunk:**
`chunk_id`, `text`, `token_count`, `book_title`, `author`, `chapter`, `page_start`, `page_end`, `position`

---

### Retrieval (`retrieval/`)

| File | Role |
|------|------|
| `faiss_retriever.py` | Dense cosine similarity search using FAISS `IndexFlatIP`. Query gets `"query: "` prefix for e5 models. |
| `bm25_retriever.py` | Sparse keyword search using `BM25Okapi` (whitespace tokenized for Urdu). |
| `hybrid.py` | RRF fusion: `score = Σ 1/(60 + rank)` — rewards chunks appearing high in both lists. |
| `reranker.py` | Cross-encoder `ms-marco-MiniLM-L-6-v2` re-scores query-chunk pairs. Falls back to RRF scores if unavailable. |

---

### Generation (`generation/`)

| File | Role |
|------|------|
| `prompt.py` | Builds the Urdu system prompt + numbered excerpts block. System instruction is fully in Urdu — LLM is told to only answer from provided excerpts. |
| `llm.py` | Google Gemini wrapper (`google-genai` SDK). Supports both streaming (SSE tokens) and non-streaming responses. |

---

### API (`main.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ingest` | POST | Upload `.txt` file + metadata → background indexing |
| `/ingest/status` | GET | Poll ingestion state (`starting/cleaning/chunking/indexing/done/error`) |
| `/query` | POST | Ask a question, returns answer + citations (streaming or JSON) |
| `/chunks` | GET | Browse/search all indexed chunks with pagination |
| `/health` | GET | Index stats + current ingestion state |
| `/index` | DELETE | Wipe all indexes and start fresh |

---

## Key Design Decisions

- **Hybrid retrieval** — dense alone misses exact keyword matches in Urdu; sparse alone misses semantic similarity. RRF combines both without needing score normalization.
- **Sentence-boundary chunking** — chunks never cut mid-sentence, preserving meaning. Overlap ensures context at chunk boundaries is not lost.
- **e5 prefix convention** — `intfloat/multilingual-e5-large` requires `"passage: "` prefix at index time and `"query: "` at query time for correct cosine alignment.
- **Cross-encoder reranking** — bi-encoder retrieval is fast but approximate; the cross-encoder does full query-passage attention for precise final ranking.
- **Urdu-only LLM instruction** — the system prompt is written in Urdu and constrains the model to only use retrieved excerpts, reducing hallucination.

---

## Storage Layout

```
storage/
  faiss.index      ← FAISS binary index (vectors)
  bm25.pkl         ← BM25Okapi pickled object
  metadata.json    ← All chunk metadata (text + source info)
```

All three are rebuilt atomically on each `/ingest` call and wiped by `DELETE /index`.

---

## Environment Variables (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Required. Google AI Studio key. |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model for answer generation |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder |
| `TOP_K_DENSE` | `20` | FAISS candidates per query |
| `TOP_K_SPARSE` | `20` | BM25 candidates per query |
| `TRANSFORMERS_OFFLINE` | `1` | Set to `0` on first run to download models |
| `HF_HUB_OFFLINE` | `1` | Set to `0` on first run to download models |
