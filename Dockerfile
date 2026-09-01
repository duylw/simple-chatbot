# ==============================================================================
# Stage 1: Builder - Compile dependencies into virtual environment using UV
# ==============================================================================
FROM python:3.12-slim AS builder

# Install UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Compile bytecode and install production dependencies (excluding dev and gradio groups)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --no-dev --no-group gradio --locked

# ==============================================================================
# Stage 2: Production Runtime - Hardened, Minimal, Non-root
# ==============================================================================
FROM python:3.12-slim AS runtime

# Security hardening: Create non-root system user and group
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy virtual environment and application code from builder / repository
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup . /app

# Ensure directories for sqlite/data/logs exist and have proper permissions
RUN mkdir -p /app/data /app/media && chown -R appuser:appgroup /app

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Switch to non-root user
USER appuser

EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]