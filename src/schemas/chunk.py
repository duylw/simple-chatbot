from pydantic import BaseModel

class ChunkBase(BaseModel):
    content: str
    timestamp: int
    duration: int

class ChunkResponse(ChunkBase):
    id: str  # UUID as string
    video_id: str  # UUID as string

    class Config:
        orm_mode = True