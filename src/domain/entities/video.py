"""Video Domain Entity."""
from dataclasses import dataclass, field
import uuid
from typing import Optional


@dataclass
class Video:
    """Pure domain entity representing a lecture video."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = ""
    url: str = ""

    def __post_init__(self):
        if isinstance(self.id, str):
            self.id = uuid.UUID(self.id)
