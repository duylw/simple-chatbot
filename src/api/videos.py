from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_db_session
from src.schemas.video import VideoResponse

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.get("/", response_model=list[VideoResponse])
async def list_videos(
    db: AsyncSession = Depends(get_db_session),
    limit: int = 10,
    offset: int = 0
):
    # Placeholder implementation
    pass

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str, 
    db: AsyncSession = Depends(get_db_session)
):
    # Placeholder implementation
    pass
