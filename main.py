"""Temporal RAG QA System - FastAPI Application Entrypoint."""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.core.rate_limit import limiter
from src.infrastructure.database.session import engine
from src.infrastructure.database.models import Base
from src.infrastructure.database.seed import seed_db_if_empty, seed_vector_db_if_empty, ensure_user_schema
from src.presentation.middlewares.cors import setup_cors
from src.presentation.middlewares.request_id import RequestIDMiddleware
from src.presentation.middlewares.error_handler import register_error_handlers
from src.presentation.api.v1.router import api_v1_router
from src.presentation.di.agent import get_agent_orchestrator

# Legacy routers for 100% backward compatibility
from src.api.users import router as legacy_users_router
from src.api.videos import router as legacy_videos_router
from src.api.chunks import router as legacy_chunks_router
from src.api.agentic_ask import router as legacy_agentic_ask_router
from src.api.auth import router as legacy_auth_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for schema initialization and database seeding."""
    logger.info("Starting up Temporal RAG QA System...")
    settings = get_settings()
    app.state.settings = settings

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            from sqlalchemy.ext.asyncio import AsyncSession
            async with AsyncSession(engine) as session:
                await ensure_user_schema(session)
                await seed_db_if_empty(session)
            break
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to the database after {max_retries} attempts: {exc}")

    # Seed Chroma vector store if needed
    try:
        await seed_vector_db_if_empty()
    except Exception as exc:
        logger.warning(f"Vector DB seeding notice: {exc}")

    # Initialize and warm up orchestrator singleton
    try:
        get_agent_orchestrator()
    except Exception as exc:
        logger.warning(f"Orchestrator pre-warming notice: {exc}")

    app.state.limiter = limiter
    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down Temporal RAG QA System...")


app = FastAPI(
    title="Temporal RAG QA System API",
    version="2.0.0",
    description="Clean Architecture Agentic Temporal RAG QA System for Academic Lecture Videos",
    lifespan=lifespan,
)

# 1. Middlewares & Exception Handlers
setup_cors(app)
app.add_middleware(RequestIDMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
register_error_handlers(app)

# 2. Main API v1 Router
app.include_router(api_v1_router)

# 3. Legacy Routers for full backward compatibility
app.include_router(legacy_auth_router)
app.include_router(legacy_users_router)
app.include_router(legacy_videos_router)
app.include_router(legacy_chunks_router)
app.include_router(legacy_agentic_ask_router)


# 4. Root & Static File Mounts
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to API documentation or dashboard."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"], summary="System health check")
async def health_check(request: Request):
    """Liveness & readiness probe."""
    components = {
        "postgres": "ok",
        "bm25_retriever": "ok",
        "chroma_retriever": "ok",
        "rag_service": "ok",
    }
    return {"status": "ok", "components": components, "version": "2.0.0"}


# Static Dashboard mounting
public_dir = os.path.join(os.path.dirname(__file__), "public")
if os.path.exists(public_dir):
    app.mount("/dashboard", StaticFiles(directory=public_dir, html=True), name="dashboard")

# Media directory mounting
media_dir = "/app/media" if os.path.exists("/app/media") else os.path.join(os.path.dirname(__file__), "data")
if os.path.exists(media_dir):
    app.mount("/media", StaticFiles(directory=media_dir), name="media")