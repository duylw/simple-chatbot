"""Unit tests for Infrastructure Repositories (User, Video, Chunk)."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.entities.user import User
from src.domain.entities.video import Video
from src.domain.entities.chunk import Chunk
from src.infrastructure.repositories.user_repo import PostgresUserRepository
from src.infrastructure.repositories.video_repo import PostgresVideoRepository
from src.infrastructure.repositories.chunk_repo import PostgresChunkRepository
from src.infrastructure.database.models import UserModel, VideoModel, ChunkModel


@pytest.mark.asyncio
async def test_user_repository_create_and_get():
    """Verify PostgresUserRepository creates and returns domain User entity."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    repo = PostgresUserRepository(mock_session)

    user_domain = User(email="test@example.com", name="Test User")
    created = await repo.create(user_domain, hashed_password="hashed_pwd_123")

    assert created.email == "test@example.com"
    assert created.name == "Test User"
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_video_repository_get_by_id():
    """Verify PostgresVideoRepository queries and maps VideoModel to Video domain entity."""
    mock_session = AsyncMock()
    repo = PostgresVideoRepository(mock_session)

    vid_id = uuid.uuid4()
    mock_model = VideoModel(id=vid_id, name="[CS431] Part 1.mp4", url="http://video.url")
    mock_session.get.return_value = mock_model

    result = await repo.get_by_id(vid_id)
    assert result is not None
    assert result.id == vid_id
    assert result.name == "[CS431] Part 1.mp4"
    assert isinstance(result, Video)


@pytest.mark.asyncio
async def test_chunk_repository_get_by_video_id():
    """Verify PostgresChunkRepository retrieves chunks and maps them to Chunk domain entities."""
    mock_session = AsyncMock()
    repo = PostgresChunkRepository(mock_session)

    vid_id = uuid.uuid4()
    c1_id = uuid.uuid4()
    mock_chunks = [
        ChunkModel(id=c1_id, content="Chunk 1 text", timestamp=0, duration=20, video_id=vid_id),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_chunks
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result

    chunks = await repo.get_by_video_id(vid_id, from_timestamp=0, to_timestamp=100)
    assert len(chunks) == 1
    assert chunks[0].id == c1_id
    assert chunks[0].content == "Chunk 1 text"
    assert isinstance(chunks[0], Chunk)
