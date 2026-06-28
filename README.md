# Customer Support AI Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline for intelligent customer support, built as a master's thesis project. The system combines hybrid dense/sparse retrieval, a multi-intent classifier, LangGraph-orchestrated agents, and an LLM-as-a-Judge evaluation framework — all served through a FastAPI backend and a React/TypeScript frontend.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Known Limitations](#known-limitations)

---

## Overview

This system ingests enterprise knowledge bases (PDFs, Word documents, PowerPoint files, spreadsheets, scanned images, and more), indexes them into a hybrid vector + sparse store, and answers customer queries through a multi-stage RAG pipeline. A 7-intent classifier routes each query to the most appropriate handler — whether that is a vector retrieval chain, a CSV/structured-data query engine, or a fallback LLM response.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  7-Intent Classifier │  (LLM-based intent routing)
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────────────┐
    │                                     │
    ▼                                     ▼
RAG Chain                        CSV Query Engine
    │                            (Groq-generated pandas
    │                             in RestrictedPython sandbox)
    ▼
Hybrid Retrieval
  ├── Dense: BGE-M3 embeddings → Qdrant
  └── Sparse: BM25 (bm25s / rank-bm25)
       └── RRF Fusion
           │
           ▼
    Jina Cross-Encoder Reranker
           │
           ▼
    LangGraph Orchestrator
    (multi-tier conversational memory)
           │
           ▼
    LLM Response Generation
           │
           ▼
  LLM-as-a-Judge Evaluation
  (Qwen3-32B via Groq)
```

---

## Features

- **Hybrid Retrieval** — BGE-M3 dense embeddings fused with BM25 sparse retrieval via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking** — Jina reranker refines candidate passages before generation.
- **7-Intent Classifier** — Routes queries to the correct handler (FAQ, structured data, escalation, etc.).
- **Multi-Format Document Ingestion** — Supports PDF (including scanned PDFs via OCR), DOCX, PPTX, XLSX, CSV, and more.
- **OCR Pipeline** — Local OCR using Ollama/Gemma 4; image captioning via Groq/LLaMA 4 Scout.
- **CSV / Structured Data Query Engine** — Generates pandas code via Groq and executes it in a RestrictedPython sandbox.
- **LangGraph Orchestration** — Stateful, multi-turn conversation management with multi-tier memory.
- **LLM-as-a-Judge Evaluation** — Automated quality evaluation using Qwen3-32B via Groq.
- **Google Workspace Crawlers** — Gmail and Google Drive crawlers for ingesting enterprise data.
- **React/TypeScript Frontend** — Clean chat interface (`rag-ui/`).
- **Redis Support** — Optional Redis-backed conversation buffer for multi-worker deployments.
- **JWT Authentication** — Secured API endpoints via PyJWT.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Server | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| Embeddings | BGE-M3 (FlagEmbedding) |
| Vector Store | Qdrant |
| Sparse Retrieval | BM25S / rank-bm25 |
| Reranking | Jina cross-encoder (sentence-transformers) |
| LLM Inference | Groq (Qwen3-32B, LLaMA 4 Scout) |
| OCR | Ollama / Gemma 4 (local) |
| Document Parsing | MarkItDown, pypdf, pdfplumber, python-docx, openpyxl |
| CSV Query Engine | Groq + RestrictedPython sandbox |
| Database | PostgreSQL (SQLAlchemy + psycopg2) |
| Caching | Redis (optional) |
| Frontend | React + TypeScript |
| Evaluation | Groq (Qwen3-32B), rouge-score, nltk, LangSmith |
| Auth | PyJWT |
| Testing | pytest + pytest-asyncio |

---

## Repository Structure

```
.
├── crawlers/               # Gmail and Google Drive data crawlers
│   └── requirements.txt    # Crawler-specific dependencies
├── csv_query_engine/       # Structured data query engine (pandas + RestrictedPython)
├── file_preparation/       # Document parsing and chunking utilities
├── file_processor/         # Ingestion pipeline (OCR, embedding, Qdrant upload)
├── rag-ui/                 # React/TypeScript frontend
├── scripts/                # Utility scripts (indexing, evaluation runs, etc.)
├── tests/                  # pytest test suite
├── connection.py           # PostgreSQL connection helper
├── pyproject.toml          # Project metadata and editable install config
└── requirements.txt        # Python dependencies
```

---

## Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18 (for the frontend)
- A running **Qdrant** instance (local Docker or cloud)
- A running **PostgreSQL** instance
- **Ollama** installed locally (for local OCR via Gemma 4)
- A **Groq** API key
- (Optional) **Redis** for multi-worker conversation memory
- (Optional) Google OAuth credentials for the Gmail/Drive crawlers

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Majd1029/Customer-Support-AI-Assistant.git
cd Customer-Support-AI-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

The editable install (`pip install -e .`) ensures `from file_preparation.X import Y` style imports resolve correctly from any working directory.

### 4. Install crawler dependencies (optional)

```bash
pip install -r crawlers/requirements.txt
```

### 5. Install frontend dependencies

```bash
cd rag-ui
npm install
```

---

## Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `PG_USER` | PostgreSQL username |
| `PG_PASSWORD` | PostgreSQL password |
| `PG_HOST` | PostgreSQL host (e.g. `localhost`) |
| `PG_PORT` | PostgreSQL port (default `5432`) |
| `PG_DB` | PostgreSQL database name |
| `QDRANT_URL` | Qdrant instance URL |
| `QDRANT_API_KEY` | Qdrant API key (if using Qdrant Cloud) |
| `GROQ_API_KEY` | Groq API key |
| `REDIS_URL` | Redis URL, e.g. `redis://localhost:6379/0` (optional) |
| `LANGSMITH_API_KEY` | LangSmith key for tracing (optional) |
| `JWT_SECRET` | Secret key for JWT token signing |

Verify the PostgreSQL connection:

```bash
python connection.py
```

---

## Running the System

### Start the API server

```bash
uvicorn file_processor.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend

```bash
cd rag-ui
npm run dev
```

The UI will be available at `http://localhost:5173` by default.

### Ingest documents

Place your documents in the appropriate input directory and run the ingestion script:

```bash
python scripts/ingest.py --source ./data/
```

---

## Evaluation

The system uses an LLM-as-a-Judge framework powered by **Qwen3-32B** via the Groq API. Evaluation metrics include faithfulness, answer relevance, and context recall, supplemented by ROUGE scores and NLTK-based text metrics.

```bash
python scripts/evaluate.py
```

> **Note on Groq rate limits:** The evaluation pipeline may hit HTTP 429 errors under high load. The system uses `tenacity` for automatic retries with exponential backoff. Consider batching evaluation runs or upgrading your Groq tier for large test sets.

---

## Testing

```bash
pytest tests/ -v
```

Async tests are supported via `pytest-asyncio`.

---

## Known Limitations

- `transformers` must be pinned below `5.0.0` to avoid conflicts with `sentence-transformers` (used by the Jina reranker). See `requirements.txt` for the pinned range.
- `protobuf` must be `≤ 3.20.2` for compatibility with `qdrant-client ≥ 1.9`.
- OCR quality depends on the local Ollama/Gemma 4 model. For production-grade OCR on complex scanned documents, a higher-capacity model is recommended.
- The CSV query engine executes Groq-generated pandas code inside a RestrictedPython sandbox. Complex or deeply nested queries may occasionally produce incorrect pandas expressions; always validate results on critical data.

---
