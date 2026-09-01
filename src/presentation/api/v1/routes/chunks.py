"""Chunk Management API Routes."""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query

from src.application.dtos.chunk import ChunkDTO, ChunkListResponseDTO
from src.application.use_cases.manage_chunk import ManageChunkUseCase
from src.domain.interfaces.repositories import IChunkRepository
from src.presentation.di.database import get_chunk_repository

router = APIRouter(prefix="/chunks", tags=["Chunks"])


def get_manage_chunk_use_case(chunk_repo: IChunkRepository = Depends(get_chunk_repository)) -> ManageChunkUseCase:
    return ManageChunkUseCase(chunk_repo)


@router.get("", response_model=ChunkListResponseDTO, summary="List all indexed transcript chunks")
async def list_chunks(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    use_case: ManageChunkUseCase = Depends(get_manage_chunk_use_case),
) -> ChunkListResponseDTO:
    """Retrieve paginated list of all transcript chunks."""
    return await use_case.list_chunks(limit=limit, offset=offset)


@router.get("/{chunk_id}", response_model=ChunkDTO, summary="Get chunk details by ID")
async def get_chunk(
    chunk_id: uuid.UUID,
    use_case: ManageChunkUseCase = Depends(get_manage_chunk_use_case),
) -> ChunkDTO:
    """Retrieve detailed information about a single transcript chunk."""
    return await use_case.get_by_id(chunk_id)


@router.get("/video/{video_id}", response_model=ChunkListResponseDTO, summary="Get chunks for a specific video")
async def get_chunks_by_video(
    video_id: uuid.UUID,
    from_timestamp: int = Query(0, ge=0, description="Start timestamp in seconds"),
    to_timestamp: int = Query(0, ge=0, description="End timestamp in seconds"),
    use_case: ManageChunkUseCase = Depends(get_manage_chunk_use_case),
) -> ChunkListResponseDTO:
    """Retrieve all chunks belonging to a video, optionally filtered by timestamp window."""
    return await use_case.get_by_video_id(
        video_id=video_id,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )
