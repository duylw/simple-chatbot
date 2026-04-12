# FastAPI Database Project & Agentic RAG Engine

This project serves as a robust backend foundation focusing on modern AI infrastructure, establishing best practices for API design, clean code architecture, and complex application lifecycles by building the foundation for an LLM-assisted Agentic Chatbot.

## Architecture Highlights

The codebase implements a strict layered **Repository Pattern** ensuring maximal separation of concerns and testability:

- **`src/api/`**: FastAPI route handlers directly utilizing robust strict Dependency Injection.
- **`src/services/`**: Core business logic and advanced orchestration (e.g., LangGraph configurations and LLM routing).
- **`src/repositories/`**: Complete encapsulation of SQLAlchemy 2.0 asynchronous calls.
- **`src/models/` & `src/schemas/`**: Separation between database schemas (SQLAlchemy) and interface definitions (Pydantic V2).
- **`src/database/`**: Configuration, lifecycle auto-seeding routines, and core database interactions.
- **`src/dependencies.py`**: A unified container resolving application state and database sessions.

## Tech Stack

- **Python 3.12+**
- **FastAPI** for high-performance API routing (using the `lifespan` context API)
- **SQLAlchemy 2.0+** (async) + **asyncpg** mapping
- **PostgreSQL** as the relational database + **Chroma/VectorStores** for embeddings
- **LangChain / LangGraph** for Agentic Rag workflow definitions
- **Pydantic V2** for reliable object validation
- **Docker & Docker Compose** for streamlined containerized development
- **uv** for ultra-fast pip dependency management

## Getting Started

### Option 1: Full Docker Environment (Recommended)

This runs the entire stack inside Docker network bindings (API, Vector stores, and PostgreSQL database). Code synchronization leverages Docker's watch mechanism for immediate reloading upon save without constant image building.

1. **Start the environment in watch mode:**
   ```bash
   docker compose watch```

2. **Access the API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### Option 2: Local Backend + Docker Database

This runs the Python backend directly on your machine while keeping the database containerized.

1. **Start only the database:**
   ```bash
   docker compose up db -d
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Run the local development server:**
   ```bash
   uv run uvicorn main:app --reload
   ```

4. **Access the API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
