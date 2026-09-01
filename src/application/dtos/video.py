"""Video DTO schemas."""
import uuid
from typing import List, Optional
from pydantic import BaseModel


class VideoDTO(BaseModel):
    """Video representation DTO."""
    id: uuid.UUID
    name: str
    url: Optional[str] = None


class VideoListResponseDTO(BaseModel):
    """Paginated list of videos."""
    total: int
    items: List[VideoDTO]
