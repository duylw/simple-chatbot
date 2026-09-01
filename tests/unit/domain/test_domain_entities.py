"""Unit tests for Domain Entities, Value Objects and Exceptions."""
import uuid
import pytest
from src.domain.entities.user import User
from src.domain.entities.video import Video
from src.domain.entities.chunk import Chunk
from src.domain.entities.agent import (
    GuardrailEvaluation,
    AnswerGrade,
    TemporalCitation,
    AgentResponseDomain,
)
from src.domain.exceptions import (
    DomainError,
    EntityNotFoundError,
    EntityAlreadyExistsError,
    GuardrailViolationError,
    LLMProviderError,
    InvalidCredentialsError,
)


def test_user_entity_creation():
    """Verify User domain entity normalizes email and sets default active state."""
    user = User(
        id=1,
        email="  TEST@Example.com ",
        name="John Doe",
        hashed_password="hashed_secret_123",
        is_active=True,
    )
    assert user.id == 1
    assert user.email == "test@example.com"
    assert user.name == "John Doe"
    assert user.is_active is True


def test_video_entity_creation():
    """Verify Video domain entity generates valid UUID and formats strings."""
    custom_id = uuid.uuid4()
    video = Video(
        id=custom_id,
        name="[CS431] Lecture 1.mp4",
        url="http://localhost:8000/media/videos/lecture1.mp4",
    )
    assert video.id == custom_id
    assert video.name == "[CS431] Lecture 1.mp4"

    # String UUID conversion
    video_str = Video(
        id=str(custom_id),
        name="Lecture 2",
        url="",
    )
    assert isinstance(video_str.id, uuid.UUID)
    assert video_str.id == custom_id


def test_chunk_entity_temporal_formatting():
    """Verify Chunk domain entity computes end_timestamp and formats readable time ranges."""
    vid_id = uuid.uuid4()
    chunk = Chunk(
        content="Lecture snippet text",
        timestamp=75,  # 01:15
        duration=30,   # 30s -> end 01:45 (105s)
        video_id=vid_id,
    )
    assert chunk.end_timestamp == 105
    assert chunk.time_range_formatted == "01:15 - 01:45"

    # Test hours formatting (> 3600s)
    long_chunk = Chunk(
        content="Long lecture snippet",
        timestamp=3665,  # 01:01:05
        duration=45,     # end 3710 (01:01:50)
    )
    assert long_chunk.time_range_formatted == "01:01:05 - 01:01:50"


def test_guardrail_evaluation_schema():
    """Verify GuardrailEvaluation pydantic model parses valid payload."""
    eval_res = GuardrailEvaluation(
        is_lecture_related=True,
        reasoning="Câu hỏi về Self-Attention thuộc phạm vi môn học.",
        feedback="Đang tìm kiếm thông tin về Attention.",
    )
    assert eval_res.is_lecture_related is True
    assert "Self-Attention" in eval_res.reasoning


def test_answer_grade_schema():
    """Verify AnswerGrade model handles suggestions for reflection retry."""
    grade = AnswerGrade(
        is_relevant=False,
        suggestion="Thử tìm kiếm với từ khóa Transformer Decoder",
        reasoning="Câu trả lời chưa đủ chi tiết.",
    )
    assert grade.is_relevant is False
    assert "Transformer" in grade.suggestion


def test_temporal_citation():
    """Verify TemporalCitation holds structured video metadata."""
    citation = TemporalCitation(
        video_name="Attention.mp4",
        time_range="01:20 - 02:40",
        start_seconds=80,
        end_seconds=160,
        slide_number=14,
        content_snippet="Q, K, V matrices calculation.",
    )
    assert citation.video_name == "Attention.mp4"
    assert citation.slide_number == 14


def test_domain_exceptions():
    """Verify domain exceptions format error messages correctly without HTTP status dependencies."""
    err1 = EntityNotFoundError("Video", "abc-123")
    assert str(err1) == "Video with identifier 'abc-123' not found."
    assert isinstance(err1, DomainError)

    err2 = EntityAlreadyExistsError("User", "email", "test@example.com")
    assert "User with email='test@example.com' already exists." in str(err2)

    err3 = GuardrailViolationError("Off-topic prompt", "Chỉ hỗ trợ nội dung bài giảng")
    assert err3.reasoning == "Off-topic prompt"
    assert err3.feedback == "Chỉ hỗ trợ nội dung bài giảng"

    err4 = LLMProviderError("groq", "Rate limit exceeded")
    assert "LLM Provider 'groq' error: Rate limit exceeded" in str(err4)

    err5 = InvalidCredentialsError()
    assert str(err5) == "Invalid email or password."
