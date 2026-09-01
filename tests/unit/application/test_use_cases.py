"""Unit tests for Application Use Cases (AskQuestion, Video, Chunk, Auth)."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document

from src.application.dtos.agent import QueryRequestDTO
from src.application.use_cases.ask_question import AskQuestionUseCase
from src.application.use_cases.manage_video import ManageVideoUseCase
from src.application.use_cases.manage_chunk import ManageChunkUseCase
from src.domain.entities.video import Video
from src.domain.entities.chunk import Chunk
from src.domain.exceptions import EntityNotFoundError


@pytest.mark.asyncio
async def test_ask_question_use_case():
    """Verify AskQuestionUseCase coordinates with orchestrator and maps response."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.ask = AsyncMock(return_value={
        "query": "Cơ chế Self-Attention",
        "rewritten_query": "Self-Attention Mechanism",
        "answer": "Giải thích chi tiết",
        "sources": [
            Document(page_content="Chunk 1", metadata={"video_name": "CS431", "timestamp": 10, "duration": 15})
        ],
        "n_iterations": 1,
        "n_llm_calls": 2,
        "execution_time": 1.25,
        "guardrail_result": "Hợp lệ",
    })

    use_case = AskQuestionUseCase(mock_orchestrator)
    req = QueryRequestDTO(question="Cơ chế Self-Attention", model="gemini-3.5-flash-lite")
    response = await use_case.execute(req)

    assert response.query == "Cơ chế Self-Attention"
    assert response.answer == "Giải thích chi tiết"
    assert len(response.sources) == 1
    assert response.sources[0].video_name == "CS431"
    assert response.sources[0].time_range == "00:10 - 00:25"
    assert response.n_iterations == 1


@pytest.mark.asyncio
async def test_manage_video_use_case():
    """Verify ManageVideoUseCase retrieves and formats video DTOs."""
    mock_repo = AsyncMock()
    vid_id = uuid.uuid4()
    mock_repo.get_by_id.return_value = Video(id=vid_id, name="Video 1", url="http://video1")

    use_case = ManageVideoUseCase(mock_repo)
    res = await use_case.get_by_id(vid_id)

    assert res.id == vid_id
    assert res.name == "Video 1"
    assert res.url == "http://video1"


@pytest.mark.asyncio
async def test_manage_video_not_found():
    """Verify ManageVideoUseCase raises EntityNotFoundError when video does not exist."""
    mock_repo = AsyncMock()
    mock_repo.get_by_id.return_value = None

    use_case = ManageVideoUseCase(mock_repo)
    with pytest.raises(EntityNotFoundError):
        await use_case.get_by_id(uuid.uuid4())
