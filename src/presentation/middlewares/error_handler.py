"""Global Exception Handlers for Domain & Infrastructure errors."""
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.domain.exceptions import (
    DomainException,
    EntityNotFoundError,
    EntityAlreadyExistsError,
    GuardrailViolationError,
    InvalidCredentialsError,
    LLMProviderError,
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app."""

    @app.exception_handler(EntityNotFoundError)
    async def handle_not_found(request: Request, exc: EntityNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Not Found", "message": str(exc)},
        )

    @app.exception_handler(EntityAlreadyExistsError)
    async def handle_already_exists(request: Request, exc: EntityAlreadyExistsError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "Conflict", "message": str(exc)},
        )

    @app.exception_handler(InvalidCredentialsError)
    async def handle_invalid_credentials(request: Request, exc: InvalidCredentialsError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Unauthorized", "message": str(exc)},
        )

    @app.exception_handler(GuardrailViolationError)
    async def handle_guardrail_violation(request: Request, exc: GuardrailViolationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "message": str(exc)},
        )

    @app.exception_handler(LLMProviderError)
    async def handle_llm_error(request: Request, exc: LLMProviderError):
        logger.error(f"LLM Provider Error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"error": "Bad Gateway", "message": "Upstream LLM provider failure. Please try again."},
        )

    @app.exception_handler(DomainException)
    async def handle_domain_generic(request: Request, exc: DomainException):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Domain Error", "message": str(exc)},
        )
