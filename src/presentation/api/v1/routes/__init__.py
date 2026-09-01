"""API v1 routes package."""
from .agent import router as agent_router
from .videos import router as videos_router
from .chunks import router as chunks_router
from .auth import router as auth_router
from .system import router as system_router

__all__ = [
    "agent_router",
    "videos_router",
    "chunks_router",
    "auth_router",
    "system_router",
]
