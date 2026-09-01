"""Video Management API Routes."""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query

from src.application.dtos.video import VideoDTO, VideoListResponseDTO
from src.application.use_cases.manage_video import ManageVideoUseCase
from src.domain.interfaces.repositories import IVideoRepository
from src.presentation.di.database import get_video_repository

router = APIRouter(prefix="/videos", tags=["Videos"])


def get_manage_video_use_case(video_repo: IVideoRepository = Depends(get_video_repository)) -> ManageVideoUseCase:
    return ManageVideoUseCase(video_repo)


@router.get("", response_model=VideoListResponseDTO, summary="List all indexed videos")
async def list_videos(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    use_case: ManageVideoUseCase = Depends(get_manage_video_use_case),
) -> VideoListResponseDTO:
    """Retrieve paginated list of all lecture videos."""
    return await use_case.list_videos(limit=limit, offset=offset)


@router.get("/{video_id}", response_model=VideoDTO, summary="Get video details by ID")
async def get_video(
    video_id: uuid.UUID,
    use_case: ManageVideoUseCase = Depends(get_manage_video_use_case),
) -> VideoDTO:
    """Retrieve detailed information about a single video."""
    return await use_case.get_by_id(video_id)
