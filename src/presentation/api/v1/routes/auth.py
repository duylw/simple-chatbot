"""Authentication API Routes."""
from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from src.application.dtos.auth import (
    UserRegisterDTO,
    UserLoginDTO,
    TokenResponseDTO,
    UserResponseDTO,
)
from src.application.use_cases.auth import AuthUseCase
from src.domain.entities.user import User
from src.domain.interfaces.repositories import IUserRepository
from src.presentation.di.database import get_user_repository
from src.presentation.di.auth import require_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_auth_use_case(user_repo: IUserRepository = Depends(get_user_repository)) -> AuthUseCase:
    return AuthUseCase(user_repo)


@router.post("/register", response_model=UserResponseDTO, status_code=status.HTTP_201_CREATED, summary="Register user")
async def register(
    dto: UserRegisterDTO,
    use_case: AuthUseCase = Depends(get_auth_use_case),
) -> UserResponseDTO:
    """Register a new user account."""
    return await use_case.register(dto)


@router.post("/token", response_model=TokenResponseDTO, summary="Login for JWT access token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    use_case: AuthUseCase = Depends(get_auth_use_case),
) -> TokenResponseDTO:
    """OAuth2 compatible token login endpoint."""
    login_dto = UserLoginDTO(email=form_data.username, password=form_data.password)
    return await use_case.login(login_dto)


@router.get("/me", response_model=UserResponseDTO, summary="Get current logged in user")
async def get_current_user_profile(
    current_user: User = Depends(require_current_user),
) -> UserResponseDTO:
    """Retrieve profile of the currently authenticated user."""
    return UserResponseDTO(id=current_user.id, email=current_user.email, name=current_user.name)
