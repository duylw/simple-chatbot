"""Central API v1 Router."""
from fastapi import APIRouter

from src.presentation.api.v1.routes.agent import router as agent_router
from src.presentation.api.v1.routes.videos import router as videos_router
from src.presentation.api.v1.routes.chunks import router as chunks_router
from src.presentation.api.v1.routes.auth import router as auth_router
from src.presentation.api.v1.routes.system import router as system_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(agent_router)
api_v1_router.include_router(videos_router)
api_v1_router.include_router(chunks_router)
api_v1_router.include_router(auth_router)
api_v1_router.include_router(system_router)
