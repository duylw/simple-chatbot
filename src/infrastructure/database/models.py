"""SQLAlchemy 2.0 ORM Models and Domain Mappers."""
import uuid
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

from src.domain.entities.user import User
from src.domain.entities.video import Video
from src.domain.entities.chunk import Chunk

Base = declarative_base()


class UserModel(Base):
    """PostgreSQL ORM Table mapping for Users."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String(255), nullable=False, default="")
    is_active = Column(Boolean, default=True)

    def to_domain(self) -> User:
        """Convert ORM model to pure Domain entity."""
        return User(
            id=self.id,
            email=self.email,
            name=self.name,
            hashed_password=self.hashed_password,
            is_active=self.is_active,
        )

    @classmethod
    def from_domain(cls, user: User, hashed_password: str = "") -> "UserModel":
        """Create ORM model instance from pure Domain entity."""
        return cls(
            id=user.id,
            email=user.email,
            name=user.name,
            hashed_password=hashed_password or user.hashed_password,
            is_active=user.is_active,
        )


class VideoModel(Base):
    """PostgreSQL ORM Table mapping for Videos."""
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)

    chunks = relationship("ChunkModel", back_populates="video", cascade="all, delete-orphan")

    def to_domain(self) -> Video:
        """Convert ORM model to pure Domain entity."""
        return Video(
            id=self.id,
            name=self.name,
            url=self.url,
        )

    @classmethod
    def from_domain(cls, video: Video) -> "VideoModel":
        """Create ORM model instance from pure Domain entity."""
        return cls(
            id=video.id,
            name=video.name,
            url=video.url,
        )


class ChunkModel(Base):
    """PostgreSQL ORM Table mapping for Video Chunks."""
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    timestamp = Column(Integer, nullable=False)
    duration = Column(Integer, nullable=False)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=True)

    video = relationship("VideoModel", back_populates="chunks")

    def to_domain(self) -> Chunk:
        """Convert ORM model to pure Domain entity."""
        return Chunk(
            id=self.id,
            content=self.content,
            timestamp=self.timestamp,
            duration=self.duration,
            video_id=self.video_id,
        )

    @classmethod
    def from_domain(cls, chunk: Chunk) -> "ChunkModel":
        """Create ORM model instance from pure Domain entity."""
        return cls(
            id=chunk.id,
            content=chunk.content,
            timestamp=chunk.timestamp,
            duration=chunk.duration,
            video_id=chunk.video_id,
        )


# Backward compatibility aliases
UserORM = UserModel
VideoORM = VideoModel
ChunkORM = ChunkModel
