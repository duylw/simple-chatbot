"""Integration tests for Presentation Layer API v1 endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from main import app
from src.presentation.di.agent import get_ask_question_use_case
from src.application.dtos.agent import AgentResponseDTO, SourceCitationDTO


@pytest.mark.asyncio
async def test_api_v1_system_health():
    """Verify GET /api/v1/system/health returns HTTP 200 with healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
async def test_api_v1_agent_models():
    """Verify GET /api/v1/agent/models returns list of supported LLMs."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/agent/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0
        assert any(m["id"] == "gemini-3.5-flash-lite" for m in models)


@pytest.mark.asyncio
async def test_api_v1_agent_ask_endpoint():
    """Verify POST /api/v1/agent/ask returns structured AgentResponseDTO."""
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=AgentResponseDTO(
        query="Self-Attention là gì?",
        rewritten_query="Cơ chế Self-Attention trong Transformer",
        answer="Self-Attention là cơ chế cho phép mô hình tính toán mối quan hệ giữa các từ.",
        sources=[
            SourceCitationDTO(
                video_name="CS431 - Lecture 5.mp4",
                timestamp=45.0,
                time_range="00:45 - 01:05",
                content="Giải thích Attention Score.",
            )
        ],
        n_iterations=1,
        n_llm_calls=2,
        execution_time=1.05,
        guardrail_result="Hợp lệ",
    ))

    app.dependency_overrides[get_ask_question_use_case] = lambda: mock_use_case

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "question": "Self-Attention là gì?",
            "model": "gemini-3.5-flash-lite",
        }
        response = await client.post("/api/v1/agent/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "Self-Attention là gì?"
        assert "Self-Attention là cơ chế" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["video_name"] == "CS431 - Lecture 5.mp4"
        assert data["n_iterations"] == 1

    app.dependency_overrides.clear()
