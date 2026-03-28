# FastAPI Database Project

This project focuses on establishing best practices for API design, backend development, and clean code architecture by building the foundation for a simple chatbot application.

## Architecture Overview

The project implements a clean layered architecture using the **Repository Pattern** to ensure separation of concerns:

- **`src/api/`**: FastAPI route handlers (Controller layer)
- **`src/services/`**: Business logic layer (Service layer)
- **`src/repositories/`**: Database queries and data access (Repository layer)
- **`src/models/`**: SQLAlchemy ORM definitions
- **`src/schemas/`**: Pydantic models for request/response validation
- **`src/database/`**: Database configuration and session management

## Tech Stack

- **Python 3.12+**
- **FastAPI** for high-performance API routing
- **SQLAlchemy 2.0+** (async) for database interactions
- **PostgreSQL** as the relational database
- **Pydantic V2** for data validation
- **Docker & Docker Compose** for containerized local development
- **uv** for fast dependency management

## Getting Started

### Option 1: Full Docker Environment (Recommended)

This runs both the API and the Database inside interconnected Docker containers. It supports live-reloading as you edit code (`docker compose watch`).

1. **Start the environment in watch mode:**
   ```bash
   docker compose watch
   ```

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
