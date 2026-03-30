import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from src.repositories.video import VideoRepository
from src.models.video import Video

class VideoService:
    def __init__(self, session: AsyncSession):
        self.repository = VideoRepository(session)

    async def get_video_by_id(self, video_id: str) -> Video | None:
        # Convert the string to a UUID object to match the parameter
        target_uuid = uuid.UUID(video_id)
        
        return await self.repository.get_by_id(target_uuid)
    
    async def get_video_by_name(self, name: str) -> list[Video]:
        return await self.repository.get_by_name(name)
    
    async def list_videos(self, limit: int = 10, offset: int = 0) -> list[Video]:
        return await self.repository.list(limit=limit, offset=offset)