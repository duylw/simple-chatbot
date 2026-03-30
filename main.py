from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.users import router as users_router
from src.api.videos import router as videos_router
from src.api.chunks import router as chunks_router
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import engine
from src.models.base import Base
from src.database.seed import seed_data_if_empty

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts
    async with engine.begin() as conn:
        # Create all tables defined in your models
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        # Optionally seed the database with initial data if it's empty
        await seed_data_if_empty(session)
    
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(users_router)
app.include_router(videos_router)
app.include_router(chunks_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)