import pytest
import jwt
from pydantic import ValidationError
from src.core.security import get_password_hash, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from src.schemas.agentic_ask import AskRequest, AgenticAskResponse
from langchain_core.documents import Document


def test_password_hashing():
    """Verify bcrypt password hashing and verification."""
    password = "MySecurePassword123!"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_flow():
    """Verify JWT creation and decoding."""
    data = "user@example.com"
    token = create_access_token(subject=data, additional_claims={"role": "student"})
    assert isinstance(token, str)

    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == "user@example.com"
    assert decoded["role"] == "student"
    assert "exp" in decoded


def test_ask_request_validation():
    """Verify AskRequest validates question length and optional model."""
    # Valid
    req = AskRequest(question="Giải thích Self-Attention", model="groq/llama-3.3-70b-versatile")
    assert req.question == "Giải thích Self-Attention"
    assert req.model == "groq/llama-3.3-70b-versatile"

    # Empty question raises validation error (min_length=1)
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_agentic_ask_response_validation():
    """Verify AgenticAskResponse enforces string type for rewritten_query and answer."""
    doc = Document(page_content="Content", metadata={"video_name": "v1.mp4", "timestamp": 0})
    resp = AgenticAskResponse(
        query="Self-Attention",
        rewritten_query="Cơ chế Self-Attention",
        answer="Giải thích chi tiết...",
        sources=[doc],
        n_iterations=0,
        n_llm_calls=1,
        execution_time=1.25,
        guardrail_result="Hợp lệ",
    )
    assert resp.rewritten_query == "Cơ chế Self-Attention"
    assert len(resp.sources) == 1
    assert resp.execution_time == 1.25
