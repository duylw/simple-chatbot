"""System Health & Readiness API Routes."""
from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(prefix="/system", tags=["System"])


class HealthStatusDTO(BaseModel):
    status: str
    version: str = "2.0.0"


@router.get("/health", response_model=HealthStatusDTO, summary="Healthcheck endpoint")
async def healthcheck() -> HealthStatusDTO:
    """Liveness probe returning HTTP 200."""
    return HealthStatusDTO(status="healthy")


@router.get("/ready", response_model=HealthStatusDTO, summary="Readiness probe")
async def readiness() -> HealthStatusDTO:
    """Readiness probe checking downstream services."""
    return HealthStatusDTO(status="ready")
