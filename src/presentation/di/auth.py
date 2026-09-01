"""Authentication Dependency Providers."""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from src.core.security import SECRET_KEY, ALGORITHM
from src.domain.entities.user import User
from src.domain.interfaces.repositories import IUserRepository
from src.presentation.di.database import get_user_repository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    user_repo: IUserRepository = Depends(get_user_repository),
) -> Optional[User]:
    """Retrieve currently authenticated user from JWT token, or None if anonymous."""
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
    except Exception:
        return None

    return await user_repo.get_by_email(email)


async def require_current_user(
    current_user: Optional[User] = Depends(get_current_user),
) -> User:
    """Enforce authentication requirement on protected routes."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
