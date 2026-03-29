from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.session import get_db_session
from src.schemas.chunk import ChunkResponse

router = APIRouter(prefix="/chunks", tags=["Chunks"])

@router.get("/", response_model=list[ChunkResponse])
async def list_chunks(
    video_id: str,
    db: AsyncSession = Depends(get_db_session),
    limit: int = 10,
    offset: int = 0
):
    # Placeholder implementation
    pass

@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    chunk_id: str, 
    db: AsyncSession = Depends(get_db_session)
):
    # Placeholder implementation
    pass