"""User Domain Entity."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """Pure domain entity representing a system user."""
    id: Optional[int] = None
    email: str = ""
    name: str = ""
    hashed_password: str = ""
    is_active: bool = True

    def __post_init__(self):
        if self.email:
            self.email = self.email.strip().lower()
