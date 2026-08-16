from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.chunk import Chunk
from typing import List, Optional

import uuid

class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
        
    async def get_by_id(self, chunk_id: uuid.UUID) -> Optional[Chunk]:
        return await self.session.get(Chunk, chunk_id)
    
    async def list(self,
                   limit: int = 10,
                   offset: int = 0
                   ) -> List[Chunk]:
        result = await self.session.execute(
            select(Chunk).limit(limit).offset(offset)
        )
        return result.scalars().all()
    
    async def get_by_video_id(self,
                              video_id: uuid.UUID,
                              from_timestamp: int = 0,
                              to_timestamp: int = 0
                              ) -> List[Chunk]:
        
        stmt = select(Chunk).where(Chunk.video_id == video_id)
        if from_timestamp > 0:
            stmt = stmt.where(Chunk.timestamp >= from_timestamp)
        if to_timestamp > 0:
            stmt = stmt.where(Chunk.timestamp <= to_timestamp)
            
        result = await self.session.execute(stmt.order_by(Chunk.timestamp.asc()))
        return result.scalars().all()