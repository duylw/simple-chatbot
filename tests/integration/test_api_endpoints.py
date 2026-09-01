import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock
from main import app
from src.presentation.di.auth import get_current_user
from src.presentation.di.agent import get_ask_question_use_case
from src.domain.entities.user import User
from src.application.dtos.agent import AgentResponseDTO, SourceCitationDTO
from src.core.security import create_access_token


@pytest.fixture
def mock_user():
    return User(
        id=1,
        email="teststudent@example.com",
        name="Test Student",
        hashed_password="fake_hashed_password",
        is_active=True,
    )


@pytest.fixture
def mock_use_case():
    use_case = MagicMock()
    use_case.execute = AsyncMock(return_value=AgentResponseDTO(
        query="Cơ chế Self-Attention",
        rewritten_query="Cơ chế Self-Attention trong Transformer",
        answer="Self-Attention tính toán ma trận Attention Score bằng công thức Softmax(QK^T / sqrt(d_k))V.",
        sources=[
            SourceCitationDTO(
                video_name="Attention.mp4",
                timestamp=120.0,
                time_range="02:00 - 02:30",
                content="Cơ chế Attention",
            )
        ],
        n_iterations=1,
        n_llm_calls=1,
        execution_time=0.85,
        guardrail_result="Câu hỏi hợp lệ",
    ))
    return use_case


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify health endpoint responds with JSON status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "components" in data


@pytest.mark.asyncio
async def test_agent_ask_authenticated_flow(mock_user, mock_use_case):
    """Verify /api/v1/agent/ask accepts valid authenticated request and returns AgentResponseDTO."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_ask_question_use_case] = lambda: mock_use_case

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            token = create_access_token(subject=mock_user.email)
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "question": "Cơ chế Self-Attention hoạt động như thế nào?",
                "model": "gemini-3.5-flash-lite",
            }
            response = await client.post("/api/v1/agent/ask", json=payload, headers=headers)
            assert response.status_code == 200
            data = response.json()

            assert data["query"] == "Cơ chế Self-Attention"
            assert "Self-Attention" in data["answer"]
            assert len(data["sources"]) == 1
            assert data["sources"][0]["video_name"] == "Attention.mp4"
            assert data["execution_time"] == 0.85
            assert data["n_llm_calls"] == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_agent_ask_validation_error(mock_user, mock_use_case):
    """Verify /api/v1/agent/ask returns 422 for invalid/empty question."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_ask_question_use_case] = lambda: mock_use_case

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            token = create_access_token(subject=mock_user.email)
            headers = {"Authorization": f"Bearer {token}"}
            # Empty question triggers 422 Unprocessable Entity
            response = await client.post("/api/v1/agent/ask", json={"question": ""}, headers=headers)
            assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
