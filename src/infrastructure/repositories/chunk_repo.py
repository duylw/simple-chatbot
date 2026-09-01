"""PostgreSQL Chunk Repository Adapter."""
from typing import Optional, List
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.chunk import Chunk
from src.domain.interfaces.repositories import IChunkRepository
from src.infrastructure.database.models import ChunkModel


class PostgresChunkRepository(IChunkRepository):
    """PostgreSQL implementation of IChunkRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, chunk_id: uuid.UUID) -> Optional[Chunk]:
        """Find a chunk by its UUID."""
        chunk_orm = await self.session.get(ChunkModel, chunk_id)
        return chunk_orm.to_domain() if chunk_orm else None

    async def get_by_video_id(
        self,
        video_id: uuid.UUID,
        from_timestamp: int = 0,
        to_timestamp: int = 0
    ) -> List[Chunk]:
        """Find chunks belonging to a video within optional timestamp boundaries."""
        stmt = select(ChunkModel).where(ChunkModel.video_id == video_id)
        if from_timestamp > 0:
            stmt = stmt.where(ChunkModel.timestamp >= from_timestamp)
        if to_timestamp > 0:
            stmt = stmt.where(ChunkModel.timestamp <= to_timestamp)

        stmt = stmt.order_by(ChunkModel.timestamp.asc())
        result = await self.session.execute(stmt)
        return [item.to_domain() for item in result.scalars().all()]

    async def list(self, limit: int = 10, offset: int = 0) -> List[Chunk]:
        """List chunks with pagination."""
        stmt = select(ChunkModel).offset(offset).limit(limit).order_by(ChunkModel.timestamp.asc())
        result = await self.session.execute(stmt)
        return [item.to_domain() for item in result.scalars().all()]
