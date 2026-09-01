"""Authentication DTO schemas."""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegisterDTO(BaseModel):
    """User registration payload."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    name: Optional[str] = None


class UserLoginDTO(BaseModel):
    """User login payload."""
    email: EmailStr
    password: str


class TokenResponseDTO(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class UserResponseDTO(BaseModel):
    """Public user profile response."""
    id: Optional[int] = None
    email: EmailStr
    name: Optional[str] = None
