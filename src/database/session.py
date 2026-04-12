from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.core.config import Settings, get_settings

# Create the async engine
engine = create_async_engine(
    get_settings().database_url,
    echo=False, # Set to False in production
    future=True,
)

# Create the async session factory
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)