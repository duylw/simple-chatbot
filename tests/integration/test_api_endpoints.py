import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from main import app
from src.dependencies import get_current_user, get_agentic_rag_service
from src.models.user import User
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
def mock_rag_service():
    service = MagicMock()
    service.ask = AsyncMock(return_value={
        "query": "Cơ chế Self-Attention",
        "rewritten_query": "Cơ chế Self-Attention trong Transformer",
        "answer": "Self-Attention tính toán ma trận Attention Score bằng công thức Softmax(QK^T / sqrt(d_k))V.",
        "sources": [
            Document(
                page_content="Cơ chế Attention",
                metadata={
                    "video_name": "Attention.mp4",
                    "timestamp": 120,
                    "duration": 30,
                    "time_range": "02:00 - 02:30",
                },
            )
        ],
        "n_iterations": 0,
        "n_llm_calls": 1,
        "execution_time": 0.85,
        "guardrail_result": "Câu hỏi hợp lệ",
    })
    return service


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
async def test_agentic_ask_unauthorized(mock_rag_service):
    """Verify /agentic_ask/ rejects unauthenticated requests with 401."""
    app.dependency_overrides[get_agentic_rag_service] = lambda: mock_rag_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/agentic_ask/", json={"question": "What is Attention?"})
            assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_agentic_ask_authenticated_flow(mock_user, mock_rag_service):
    """Verify /agentic_ask/ accepts valid authenticated request and returns AgenticAskResponse."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_agentic_rag_service] = lambda: mock_rag_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            token = create_access_token(subject=mock_user.email)
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "question": "Cơ chế Self-Attention hoạt động như thế nào?",
                "model": "gemini-3.5-flash-lite",
            }
            response = await client.post("/agentic_ask/", json=payload, headers=headers)
            assert response.status_code == 200
            data = response.json()

            assert data["query"] == payload["question"]
            assert "Self-Attention" in data["answer"]
            assert len(data["sources"]) == 1
            assert data["sources"][0]["metadata"]["video_name"] == "Attention.mp4"
            assert data["execution_time"] == 0.85
            assert data["n_llm_calls"] == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_agentic_ask_validation_error(mock_user, mock_rag_service):
    """Verify /agentic_ask/ returns 422 for invalid/empty question."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_agentic_rag_service] = lambda: mock_rag_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            token = create_access_token(subject=mock_user.email)
            headers = {"Authorization": f"Bearer {token}"}
            # Empty question triggers 422 Unprocessable Entity
            response = await client.post("/agentic_ask/", json={"question": ""}, headers=headers)
            assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
