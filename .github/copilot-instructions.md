# Project Guidelines

## Code Style
- **Language**: Python 3.12+
- **Frameworks**: FastAPI, SQLAlchemy 2.0+, Pydantic for data validation, Langchain for language model integration.
- **Async**: Use asynchronous operations where applicable (e.g., async database sessions).

## Architecture
Clean layered architecture following the repository pattern:
- **`src/api/`**: FastAPI route handlers and endpoint definitions.
- **`src/schemas/`**: Pydantic models for request/response validation.
- **`src/services/`**: Business logic layer.
- **`src/repositories/`**: Data access layer (repository pattern abstraction).
- **`src/models/`**: SQLAlchemy ORM models.
- **`src/database/`**: Database configuration and session management.
- **`src/utils/`**: Shared utility functions and helpers.

## Build and Test
- **Package Manager**: `uv`
- **Install dependencies**: `uv sync`
- **Run development server**: `uv run uvicorn main:app --reload` (or activate the virtual environment and run `uvicorn`)

## Conventions
- **Routing**: Keep API routes thin. Delegate business logic to `services/` and database access to `repositories/`.
- **Data Validation**: Always use Pydantic models in `schemas/` for API requests and responses.
- **Database**: Use SQLAlchemy 2.0 syntax. Isolate queries within the `repositories/` layer.
- **Language Model Integration**: Use Langchain for integrating with language models. Ensure proper handling of prompts and responses.

*(Note: See `README.md` and `README.Docker.md` for broader project setup and Docker configurations as they are updated.)*
