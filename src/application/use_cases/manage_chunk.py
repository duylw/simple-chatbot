"""Manage Chunk Use Case."""
import uuid
from typing import Optional, List

from src.domain.interfaces.repositories import IChunkRepository
from src.domain.exceptions import EntityNotFoundError
from src.application.dtos.chunk import ChunkDTO, ChunkListResponseDTO


class ManageChunkUseCase:
    """Handles transcript chunk querying and retrieval business logic."""

    def __init__(self, chunk_repo: IChunkRepository):
        self.chunk_repo = chunk_repo

    async def get_by_id(self, chunk_id: uuid.UUID) -> ChunkDTO:
        """Fetch a specific chunk by ID."""
        chunk = await self.chunk_repo.get_by_id(chunk_id)
        if not chunk:
            raise EntityNotFoundError(f"Chunk with id '{chunk_id}' not found.")
        return ChunkDTO(
            id=chunk.id,
            content=chunk.content,
            timestamp=chunk.timestamp,
            duration=chunk.duration,
            time_range=chunk.time_range,
            video_id=chunk.video_id,
        )

    async def get_by_video_id(
        self,
        video_id: uuid.UUID,
        from_timestamp: int = 0,
        to_timestamp: int = 0
    ) -> ChunkListResponseDTO:
        """Fetch chunks belonging to a video within optional timestamp range."""
        chunks = await self.chunk_repo.get_by_video_id(
            video_id=video_id,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        items = [
            ChunkDTO(
                id=c.id,
                content=c.content,
                timestamp=c.timestamp,
                duration=c.duration,
                time_range=c.time_range,
                video_id=c.video_id,
            )
            for c in chunks
        ]
        return ChunkListResponseDTO(total=len(items), items=items)

    async def list_chunks(self, limit: int = 10, offset: int = 0) -> ChunkListResponseDTO:
        """List chunks with pagination."""
        chunks = await self.chunk_repo.list(limit=limit, offset=offset)
        items = [
            ChunkDTO(
                id=c.id,
                content=c.content,
                timestamp=c.timestamp,
                duration=c.duration,
                time_range=c.time_range,
                video_id=c.video_id,
            )
            for c in chunks
        ]
        return ChunkListResponseDTO(total=len(items), items=items)
