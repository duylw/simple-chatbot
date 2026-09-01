"""PostgreSQL Video Repository Adapter."""
from typing import Optional, List
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.video import Video
from src.domain.interfaces.repositories import IVideoRepository
from src.infrastructure.database.models import VideoModel


class PostgresVideoRepository(IVideoRepository):
    """PostgreSQL implementation of IVideoRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, video_id: uuid.UUID) -> Optional[Video]:
        """Find a video by its UUID."""
        vid_orm = await self.session.get(VideoModel, video_id)
        return vid_orm.to_domain() if vid_orm else None

    async def get_by_name(self, name: str) -> List[Video]:
        """Find videos matching name substring."""
        stmt = select(VideoModel).where(VideoModel.name.ilike(f"%{name}%"))
        result = await self.session.execute(stmt)
        return [item.to_domain() for item in result.scalars().all()]

    async def list(self, limit: int = 10, offset: int = 0) -> List[Video]:
        """List videos with pagination."""
        stmt = select(VideoModel).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return [item.to_domain() for item in result.scalars().all()]
