from src.services.user import UserService
from src.services.video import VideoService

from src.database.session import async_session_maker, get_db_session
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import Depends

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_user_service(dependency_session: AsyncSession = Depends(get_db_session)):
    return UserService(dependency_session)


def get_video_service(dependency_session: AsyncSession = Depends(get_db_session)):
    return VideoService(dependency_session)


