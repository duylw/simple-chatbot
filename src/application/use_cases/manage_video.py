"""Manage Video Use Case."""
import uuid
from typing import Optional, List

from src.domain.interfaces.repositories import IVideoRepository
from src.domain.exceptions import EntityNotFoundError
from src.application.dtos.video import VideoDTO, VideoListResponseDTO


class ManageVideoUseCase:
    """Handles video querying and retrieval business logic."""

    def __init__(self, video_repo: IVideoRepository):
        self.video_repo = video_repo

    async def get_by_id(self, video_id: uuid.UUID) -> VideoDTO:
        """Fetch a specific video by ID."""
        video = await self.video_repo.get_by_id(video_id)
        if not video:
            raise EntityNotFoundError(f"Video with id '{video_id}' not found.")
        return VideoDTO(id=video.id, name=video.name, url=video.url)

    async def list_videos(self, limit: int = 10, offset: int = 0) -> VideoListResponseDTO:
        """List videos with pagination."""
        videos = await self.video_repo.list(limit=limit, offset=offset)
        items = [VideoDTO(id=v.id, name=v.name, url=v.url) for v in videos]
        return VideoListResponseDTO(total=len(items), items=items)
