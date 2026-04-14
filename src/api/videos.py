from fastapi import APIRouter, HTTPException, status

from src.schemas.video import VideoResponse
from src.dependencies import VideoServiceDep

router = APIRouter(prefix="/videos", tags=["Videos"])
    
@router.get("/", response_model=list[VideoResponse])
async def list_videos(
    video_service: VideoServiceDep,
    limit: int = 10,
    offset: int = 0
):
    result = await video_service.list_videos(limit=limit, offset=offset)

    return result

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    video_service: VideoServiceDep
):
    video = await video_service.get_video_by_id(video_id)

    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Video not found"
        )
        
    return video