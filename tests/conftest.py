import os
import sys
from pathlib import Path
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables before imports
os.environ["GOOGLE_API_KEY"] = "AIzaSyTestMockKeyForTestingPurposes12345"
os.environ["GROQ_API_KEY"] = "gsk_test_mock_groq_key_12345"
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-test_mock_openrouter_key"
os.environ["OPENAI_API_KEY"] = "sk-test_mock_openai_key"
os.environ["SECRET_KEY"] = "test-secret-key-for-jwt-signing-12345"
os.environ["ENVIRONMENT"] = "development"

from langchain_core.documents import Document
from src.domain.entities.agent import AnswerGrade, GuardrailEvaluation


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Ensure all test runs use safe mock API keys."""
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyTestMockKeyForTestingPurposes12345")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_groq_key_12345")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test_mock_openrouter_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test_mock_openai_key")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-jwt-signing-12345")
    monkeypatch.setenv("ENVIRONMENT", "development")


@pytest.fixture
def sample_temporal_documents() -> list[Document]:
    """Sample lecture documents with contiguous and disjoint timestamps."""
    return [
        # Video 1: 3 contiguous chunks (0s-20s, 20s-40s, 40s-60s)
        Document(
            page_content="Cơ chế Self-Attention tính toán ma trận Attention Score.",
            metadata={
                "video_name": "[CS431] Part 3_ Co che Self-Attention.mp4",
                "timestamp": 0,
                "duration": 20,
                "slide_number": 12,
            },
        ),
        Document(
            page_content="Ma trận Q, K, V được nhân với nhau và chia cho căn bậc hai của d_k.",
            metadata={
                "video_name": "[CS431] Part 3_ Co che Self-Attention.mp4",
                "timestamp": 20,
                "duration": 20,
                "slide_number": 13,
            },
        ),
        Document(
            page_content="Hàm Softmax được áp dụng để chuẩn hóa trọng số attention thành phân phối xác suất.",
            metadata={
                "video_name": "[CS431] Part 3_ Co che Self-Attention.mp4",
                "timestamp": 40,
                "duration": 20,
                "slide_number": 14,
            },
        ),
        # Video 1: 1 disjoint chunk (180s-200s)
        Document(
            page_content="Multi-Head Attention chiếu các vector Q, K, V vào nhiều không gian con khác nhau.",
            metadata={
                "video_name": "[CS431] Part 3_ Co che Self-Attention.mp4",
                "timestamp": 180,
                "duration": 20,
                "slide_number": 25,
            },
        ),
        # Video 2: 1 chunk
        Document(
            page_content="Mạng tích chập Convolutional Neural Network (CNN) sử dụng các kernel trượt trên ảnh.",
            metadata={
                "video_name": "[CS431] Part 1_ Mang CNN.mp4",
                "timestamp": 10,
                "duration": 30,
                "slide_number": 5,
            },
        ),
    ]


@pytest.fixture
def mock_guardrail_eval_relevant() -> GuardrailEvaluation:
    return GuardrailEvaluation(
        is_lecture_related=True,
        reasoning="Câu hỏi liên quan trực tiếp đến cơ chế Self-Attention trong môn học.",
        feedback="Đang tìm kiếm thông tin về cơ chế Self-Attention...",
    )


@pytest.fixture
def mock_guardrail_eval_irrelevant() -> GuardrailEvaluation:
    return GuardrailEvaluation(
        is_lecture_related=False,
        reasoning="Câu hỏi về thời tiết ngoài phạm vi bài giảng.",
        feedback="Xin lỗi, tôi chỉ hỗ trợ giải đáp kiến thức học thuật của môn học.",
    )


@pytest.fixture
def mock_answer_grade_relevant() -> AnswerGrade:
    return AnswerGrade(
        is_relevant=True,
        reasoning="Câu trả lời đầy đủ, chi tiết và chính xác theo bài giảng.",
        suggestion="",
    )


@pytest.fixture
def mock_answer_grade_irrelevant() -> AnswerGrade:
    return AnswerGrade(
        is_relevant=False,
        reasoning="Câu trả lời chưa đủ chi tiết về công thức toán học.",
        suggestion="Thử tìm kiếm với từ khóa 'Scaled Dot-Product Attention'.",
    )
