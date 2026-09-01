"""Repository Interfaces (Ports) for Domain Layer."""
from abc import ABC, abstractmethod
from typing import Optional, List
import uuid
from src.domain.entities.user import User
from src.domain.entities.video import Video
from src.domain.entities.chunk import Chunk


class IUserRepository(ABC):
    """Port for User persistence operations."""

    @abstractmethod
    async def create(self, user: User, hashed_password: str) -> User:
        """Create and persist a new user."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Find a user by email address."""
        pass

    @abstractmethod
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by ID."""
        pass

    @abstractmethod
    async def list(self, limit: int = 10, offset: int = 0) -> List[User]:
        """List users with pagination."""
        pass


class IVideoRepository(ABC):
    """Port for Video persistence operations."""

    @abstractmethod
    async def get_by_id(self, video_id: uuid.UUID) -> Optional[Video]:
        """Find a video by its UUID."""
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> List[Video]:
        """Find videos matching or containing name."""
        pass

    @abstractmethod
    async def list(self, limit: int = 10, offset: int = 0) -> List[Video]:
        """List videos with pagination."""
        pass


class IChunkRepository(ABC):
    """Port for Chunk persistence operations."""

    @abstractmethod
    async def get_by_id(self, chunk_id: uuid.UUID) -> Optional[Chunk]:
        """Find a transcript chunk by its UUID."""
        pass

    @abstractmethod
    async def get_by_video_id(
        self,
        video_id: uuid.UUID,
        from_timestamp: int = 0,
        to_timestamp: int = 0
    ) -> List[Chunk]:
        """Find chunks belonging to a video within optional timestamp boundaries."""
        pass

    @abstractmethod
    async def list(self, limit: int = 10, offset: int = 0) -> List[Chunk]:
        """List chunks with pagination."""
        pass
