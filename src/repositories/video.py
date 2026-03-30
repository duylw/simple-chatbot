import uuid
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.video import Video

class VideoRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, video_id: uuid.UUID) -> Video:
        result = await self.session.get(Video, video_id)
        return result
    
    async def get_by_name(self, video_name: str) -> List[Video]:
        result = await self.session.execute(
            select(Video).where(Video.name == video_name)
        )
        return result.scalars().all()
    
    async def list(self, limit: int = 10, offset: int = 0) -> List[Video]:
        result = await self.session.execute(
            select(Video).limit(limit).offset(offset)
        )
        return result.scalars().all()