"""Database and Repository Dependency Providers."""
from collections.abc import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.session import get_db_session
from src.infrastructure.repositories.user_repo import PostgresUserRepository
from src.infrastructure.repositories.video_repo import PostgresVideoRepository
from src.infrastructure.repositories.chunk_repo import PostgresChunkRepository
from src.domain.interfaces.repositories import IUserRepository, IVideoRepository, IChunkRepository


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency yielding async SQLAlchemy database session."""
    async for session in get_db_session():
        yield session


def get_user_repository(session: AsyncSession = Depends(get_db)) -> IUserRepository:
    """Provide IUserRepository implementation."""
    return PostgresUserRepository(session)


def get_video_repository(session: AsyncSession = Depends(get_db)) -> IVideoRepository:
    """Provide IVideoRepository implementation."""
    return PostgresVideoRepository(session)


def get_chunk_repository(session: AsyncSession = Depends(get_db)) -> IChunkRepository:
    """Provide IChunkRepository implementation."""
    return PostgresChunkRepository(session)
