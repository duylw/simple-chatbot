"""SQLAlchemy 2.0 Async Session Factory & Engine Configuration."""
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from src.core.config import get_settings

# Create the async engine
engine = create_async_engine(
    get_settings().database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

# Create the async session factory
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide transactional async database session generator."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
