"""Chunk Domain Entity (Video Transcript Chunk)."""
from dataclasses import dataclass, field
import uuid
from typing import Optional


@dataclass
class Chunk:
    """Pure domain entity representing a temporal transcript slice of a lecture video."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    content: str = ""
    timestamp: int = 0  # Start time in seconds
    duration: int = 15  # Duration in seconds
    video_id: Optional[uuid.UUID] = None

    def __post_init__(self):
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)
        if isinstance(self.video_id, str):
            self.video_id = uuid.UUID(self.video_id)

    @property
    def end_timestamp(self) -> int:
        """Calculate the end timestamp in seconds."""
        return self.timestamp + self.duration

    @property
    def time_range_formatted(self) -> str:
        """Format timestamp range as MM:SS - MM:SS or HH:MM:SS - HH:MM:SS."""
        start_str = self._format_seconds(self.timestamp)
        end_str = self._format_seconds(self.end_timestamp)
        return f"{start_str} - {end_str}"

    @staticmethod
    def _format_seconds(seconds: int) -> str:
        if seconds < 0:
            return "00:00"
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
