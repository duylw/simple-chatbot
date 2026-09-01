"""Chunk DTO schemas."""
import uuid
from typing import List, Optional
from pydantic import BaseModel


class ChunkDTO(BaseModel):
    """Transcript Chunk representation DTO."""
    id: uuid.UUID
    content: str
    timestamp: int
    duration: int
    time_range: str
    video_id: uuid.UUID


class ChunkListResponseDTO(BaseModel):
    """Paginated list of transcript chunks."""
    total: int
    items: List[ChunkDTO]
