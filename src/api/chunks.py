import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from src.schemas.chunk import ChunkResponse
from src.services.chunk import ChunkService
from src.dependencies import get_chunk_service
from typing import List

router = APIRouter(prefix="/chunks", tags=["Chunks"])

@router.get("/", response_model=List[ChunkResponse])
async def list_chunks(
    chunk_service: ChunkService = Depends(get_chunk_service),
    limit: int = 10,
    offset: int = 0
):
    return await chunk_service.list_chunks(limit, offset)

@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: str, 
    chunk_service: ChunkService = Depends(get_chunk_service)
):
    try:
        uuid.UUID(chunk_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid chunk UUID format"
        )

    chunk = await chunk_service.get_chunk_by_id(chunk_id)
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Chunk not found"
        )
    return chunk

@router.get("/video/{video_id}", response_model=List[ChunkResponse])
async def get_chunks_by_video_id(
    video_id: str,
    from_timestamp: int = 0,
    to_timestamp: int = 0,
    chunk_service: ChunkService = Depends(get_chunk_service)
):
    try:
        uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid video UUID format"
        )

    chunks = await chunk_service.get_chunk_by_video_id(video_id, from_timestamp, to_timestamp)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No chunks found for the given video ID and timestamp range"
        )
    return chunks