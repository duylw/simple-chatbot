"""PostgreSQL User Repository Adapter."""
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.user import User
from src.domain.interfaces.repositories import IUserRepository
from src.infrastructure.database.models import UserModel


class PostgresUserRepository(IUserRepository):
    """PostgreSQL implementation of IUserRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user: User, hashed_password: str) -> User:
        """Persist a new user to Postgres and return domain entity."""
        db_user = UserModel.from_domain(user, hashed_password=hashed_password)
        self.session.add(db_user)
        await self.session.commit()
        await self.session.refresh(db_user)
        return db_user.to_domain()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Find a user by normalized email."""
        clean_email = email.strip().lower()
        stmt = select(UserModel).where(UserModel.email == clean_email)
        result = await self.session.execute(stmt)
        user_orm = result.scalars().first()
        return user_orm.to_domain() if user_orm else None

    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by numeric ID."""
        user_orm = await self.session.get(UserModel, user_id)
        return user_orm.to_domain() if user_orm else None

    async def list(self, limit: int = 10, offset: int = 0) -> List[User]:
        """List users with pagination."""
        stmt = select(UserModel).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return [item.to_domain() for item in result.scalars().all()]
