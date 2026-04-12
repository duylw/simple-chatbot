# Project Guidelines

## Code Style
- **Language**: Python 3.12+
- **Frameworks**: FastAPI, SQLAlchemy 2.0+ (async via `asyncpg`), Pydantic V2 for data validation, LangGraph for orchestration.
- **Async**: Use asynchronous operations exclusively (FastAPI and SQLAlchemy). Any synchronous I/O operations must be deferred to `asyncio.to_thread`.
- **Primary Keys**: Use `uuid.UUID` for Primary Keys across all database models by default.

## Architecture
Clean layered architecture following the repository pattern:
- **`src/api/`**: FastAPI route handlers and endpoint definitions. Keep routes thin. Choose appropriate REST patterns. Use `Annotated` typing for `Depends` injections (e.g. from `src/dependencies.py`).
- **`src/schemas/`**: Pydantic V2 models for request/response validation. Always enable `model_config = {"from_attributes": True}` for proper ORM serialization.
- **`src/services/`**: Business logic layer (and LangGraph nodes/agents). Does not contain routing or raw database connections.
- **`src/repositories/`**: Data access layer. Isolate all SQLAlchemy `select`, `insert`, `update`, `delete` queries here. Never use direct `session.execute` within `api/`.
- **`src/models/`**: SQLAlchemy 2.0 declarative models (strict use of `Mapped` and `mapped_column`).
- **`src/dependencies.py`**: Centralize all FastAPI dependency injection (e.g., retrieving `App.state` properties). Avoid shadowing imports.
- **`src/database/`**: Database configuration (`config.py`), session management (`session.py`), and auto-seeding logic (`seed.py`).

## Conventions
- **Routing Modules**: Delegate business logic strictly to `services/` and database access to `repositories/`. Avoid processing states manually in routers.
- **Transactions**: Evaluate when to use `session.commit()` (e.g., in services for multi-step transactions rather than deep inside repositories). 
- **Error Handling**: Throw `HTTPException` early at the `api/` or `dependencies.py` edges instead of bubbling raw python exceptions from the internal services. 
- **Graph State Tracking**: For `langgraph` state nodes, always preserve state integrity enforcing strict `TypedDict` keys (e.g., watch out for pluralization issues like `source` vs `sources`).
- **Networking**: Default database connections refer to internal Docker networking (`db:5432`), not `localhost:5432`.

## Build and Developer Environment
- **Package Manager**: `uv`
- **Docker Compose**: The application runs entirely via Docker Compose (`backend`, `db` as Postgres 16, and `adminer`). Hot-reload mapping via `docker compose watch` runs sync without container rebuilding.
- **Database Initialization**: `Base.metadata.create_all` and CSV seeding are automatically handled during the FastAPI lifespan. Use `app.state` to hold global resources like vector stores.
