# Temporal Video RAG & Agentic QA System
> **Course Project:** CS431 - Deep Learning Techniques and Applications  
> **Topic:** Intelligent Temporal-Grounded Video Lecture Q&A System via Agentic Self-RAG Architecture

[![CI Pipeline](https://github.com/duylw/temporal-rag-qa-system/actions/workflows/ci.yaml/badge.svg)](https://github.com/duylw/temporal-rag-qa-system/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20RAG-FF8800?style=flat&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FC521F?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## 1. Team Members

| No. | Full Name            | Student ID | GitHub                                     |
| :---:| :---------------------| :----------:| :-------------------------------------------|
| 1   | **Lương Quang Duy**  | 23520368   | [@duylw](https://github.com/duylw)         |
| 2   | **Nguyễn Bá Long**   | 23520880   | [@NBasLongz](https://github.com/NBasLongz) |
| 3   | **Dương Thái Ý Nhi** | 23521106   | [@dtynhi](https://github.com/dtynhi)       |

---

## 2. Executive Summary

This project delivers a production-grade multimodal Question & Answering (Q&A) system engineered to solve the challenge of information retrieval and precise timestamp localization across long-form academic video lectures.

The system is built upon an **Agentic Self-RAG** workflow powered by **LangGraph**, combining **Hybrid Retrieval (Dense Semantic Embeddings + Sparse BM25 via Reciprocal Rank Fusion)**, **Temporal Grounding & Chunk Stitching** (merging adjacent video intervals into coherent lecture segments), and an extensible **Multi-Provider LLM Engine** supporting Google Gemini 3.5, Google Gemma 4, Groq LPU, OpenRouter, and OpenAI.

### Key Engineering Highlights:
- **Agentic LangGraph Workflow**: State machine with automated query guardrailing, retrieval, self-evaluation loop (Answer Grading), and iterative query rewriting with guided feedback.
- **Temporal Grounding & Video Synchronization**: Automatically consolidates contiguous video chunks into clean time intervals (`MM:SS - MM:SS`) and dynamically seeks the embedded video player to exact lecture moments.
- **Hybrid Retrieval with RRF**: Fuses high-density vector representations (`gemini-embedding-2-preview` on ChromaDB) and lexical search (`Rank-BM25`) using Reciprocal Rank Fusion without requiring dedicated GPU infrastructure.
- **Unified Multi-Provider LLM Engine**: Native support for Google Gemini (3.5 Flash Lite, 3.5 Flash, Gemma 4 26B/31B), Groq Cloud (Llama 3.3 70B, DeepSeek R1 Distill), OpenRouter, and OpenAI GPT-4o with automated graceful fallback handling.
- **Microservice Architecture**: Fully containerized backend (FastAPI, asyncpg, SQLAlchemy 2.0) and frontend (Gradio 6) communicating over asynchronous REST interfaces.
- **Automated CI/CD & Testing**: Comprehensive unit and integration test suite (Pytest + Asyncio) integrated into a GitHub Actions CI pipeline with 100% deterministic test execution via mocking.

---

## 3. System Architecture

```
[ User / Interactive Gradio Web UI ]
                   │
                   ▼ (HTTP / JWT Bearer Authentication)
┌─────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Application Gateway                       │
│   (OAuth2 Security, SlowAPI Rate Limiting 10 req/min, Asynchronous I/O) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LangGraph Agentic State Graph                        │
│                                                                         │
│   [ guardrail_node ] ──(Out of Scope)───> [ out_of_scope_node ]         │
│          │ (Valid Query)                                                │
│          ▼                                                              │
│   [ retrieve_node ] ──> [ Hybrid Search: BM25 + ChromaDB (RRF) ]        │
│          │                                                              │
│          ▼                                                              │
│   [ generate_answer ] ──> [ Temporal Chunk Merging & Synthesis ]        │
│          │                                                              │
│          ▼                                                              │
│   [ grade_answer ] ──(Unsatisfactory & Iter < 3)──> [ rewrite_query ] ─┐│
│          │                                                │            ││
│          │ (Satisfactory or Max Iterations Reached)       └────────────┘│
│          ▼                                                              │
│   [ invoke_response ] ──> Formats Structured Agentic Response           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
┌────────────────────────────────┐              ┌────────────────────────────────┐
│      ChromaDB (Vector DB)      │              │      PostgreSQL 16 (Relational)│
│  - Dense Semantic Embeddings   │              │  - User authentication & state │
│  - Temporal chunk metadata     │              │  - Video & Lecture Chunk Index │
└────────────────────────────────┘              └────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer / Component | Technologies | Purpose |
| :--- | :--- | :--- |
| **API & Backend Framework** | FastAPI, Uvicorn, Pydantic v2 | High-throughput asynchronous REST API services |
| **Database & ORM** | PostgreSQL 16, SQLAlchemy 2.0 (Async), Alembic | Relational data persistence, user management, chunk catalog |
| **Agent & Graph Orchestration** | LangGraph, LangChain, Rank-BM25 | State-machine based Self-RAG and hybrid search orchestration |
| **Vector Storage** | ChromaDB | High-performance embedding index for dense similarity search |
| **LLM & Embeddings** | Google Gemini 3.5, Gemma 4, Groq LPU, OpenRouter | Text generation, query rewriting, guardrails, self-grading |
| **Frontend Client** | Gradio 6, HTML5 Video Player | Interactive web dashboard with dynamic video timestamp seeking |
| **Observability & Tracing** | Langfuse OpenTelemetry | Latency tracking, token consumption monitoring, LLM tracing |
| **Security & Protection** | PyJWT, Passlib (bcrypt), SlowAPI | Authentication, password hashing, and API DDoS rate limiting |
| **Testing & CI/CD** | Pytest, Pytest-Asyncio, Pytest-Cov, GitHub Actions | Automated unit/integration test execution and coverage reports |
| **Containerization** | Docker, Docker Compose, Astral UV | Production-ready multi-service container orchestration |

---

## 5. Project Directory Structure

```
temporal-rag-qa-system/
├── .github/
│   └── workflows/
│       └── ci.yaml             # Automated GitHub Actions CI pipeline
├── data/                       # Lecture transcripts, metadata, and knowledge base
├── public/                     # Static assets and monitoring dashboard
├── src/
│   ├── api/                    # REST route controllers (auth, users, videos, chunks, agentic_ask)
│   ├── core/                   # Security, JWT tokens, config schemas, rate limiter, logging
│   ├── database/               # Async SQLAlchemy session management, migrations, seed script
│   ├── gradio_ui/              # Standalone Gradio frontend (components, handlers, video utils)
│   ├── models/                 # SQLAlchemy ORM models (User, Video, Chunk)
│   ├── repositories/           # Data Access Layer (CRUD abstractions)
│   ├── schemas/                # Pydantic schemas for request/response validation
│   └── services/
│       ├── rag/                # LangGraph state machine, nodes, tools, prompts, llm_factory
│       └── user.py, video.py   # Domain services
├── tests/
│   ├── conftest.py             # Global test fixtures and deterministic mock environments
│   ├── unit/                   # Unit tests (temporal utils, llm factory, rag nodes, auth)
│   └── integration/            # Integration tests (FastAPI endpoints, Gradio client)
├── compose.yaml                # Multi-container Docker Compose production definition
├── Dockerfile                  # Backend container build specification
├── Dockerfile.gradio           # Lightweight Gradio frontend container build specification
├── pyproject.toml              # Project dependencies and tool configurations (Astral UV)
└── README.md
```

---

## 6. Data Confidentiality Statement

> **Notice Regarding Dataset Assets:**
> The original high-resolution lecture video files (`.mp4`) are proprietary course materials and are **kept confidential (omitted from this public repository)**.
>
> **Autonomous Functionality:**
> All associated lecture transcripts, structural metadata, temporal segment boundaries, ChromaDB vector embeddings, and PostgreSQL seed schemas are pre-computed and packaged within the automated initialization pipeline. Consequently, **evaluators and recruiters can build, launch, and test 100% of the Agentic RAG workflow, API endpoints, and UI features without needing local raw video files**.

---

## 7. Installation & Quickstart Guide

### 7.1 Prerequisites
- **Docker** and **Docker Compose** (Recommended for seamless containerized execution).
- Alternatively, **Python 3.12+** and **Astral UV** package manager for local development.

### 7.2 Environment Configuration
Copy the environment template and provide the required API credentials:
```bash
cp .env.example .env
```
Key configuration parameters in `.env`:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key          # (Optional) For ultra-fast Llama 3.3 / DeepSeek R1
OPENROUTER_API_KEY=your_openrouter_key  # (Optional)
SECRET_KEY=your_secure_jwt_secret_key
```

### 7.3 Launch via Docker Compose
Deploy all services in detached mode with a single command:
```bash
docker compose up -d --build
```

Once started, access the respective services:
- **Gradio Web Interface**: `http://localhost:7860`
- **FastAPI Interactive Docs (Swagger)**: `http://localhost:8000/docs`
- **ChromaDB Vector Store API**: `http://localhost:8008`
- **PostgreSQL Adminer Dashboard**: `http://localhost:8081`

---

## 8. Automated Testing & CI Pipeline

The project enforces strict software engineering standards with isolated, deterministic unit and integration test coverage. All external LLM calls are mocked during testing to avoid API quota consumption.

Continuous integration is managed automatically via **GitHub Actions**. Real-time execution logs and test reports can be monitored at the [GitHub Actions CI Pipeline](https://github.com/duylw/temporal-rag-qa-system/actions/workflows/ci.yaml).

Execute the Pytest test suite with code coverage analysis locally:
```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Suite Execution Output:
```text
======================= 32 passed in 15.02s ========================
- Unit Tests: 100% PASSED (Temporal algorithms, LLM multi-provider factory, LangGraph nodes, Auth)
- Integration Tests: 100% PASSED (FastAPI REST endpoints, Gradio client)
```

---

## 9. Academic Affiliation & License
This project was developed for academic and research purposes as part of the course **CS431 - Deep Learning Techniques and Applications** at the University of Information Technology (UIT - VNU-HCM).