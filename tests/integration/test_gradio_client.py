import pytest
import httpx
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.documents import Document
from src.gradio_ui.utils import (
    format_sources_dataframe,
    sort_sources,
    parse_timestamp_to_seconds,
    build_video_player,
    empty_sources_dataframe,
)
from src.gradio_ui.api_client import BackendClient


def test_format_and_sort_sources_dataframe():
    """Verify Gradio DataFrame formats video names, time ranges, and sort ordering."""
    docs = [
        Document(
            page_content="Content B",
            metadata={"video_name": "Video B", "timestamp": 120, "duration": 30, "time_range": "02:00 - 02:30"},
        ),
        Document(
            page_content="Content A",
            metadata={"video_name": "Video A", "timestamp": 60, "duration": 20, "time_range": "01:00 - 01:20"},
        ),
    ]

    df = format_sources_dataframe(docs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Video" in df.columns
    assert "Timestamp" in df.columns
    assert "URL" in df.columns
    assert df.iloc[0]["Timestamp"] == "02:00 - 02:30"

    # Test sorting
    sorted_df = sort_sources(df)
    assert sorted_df.iloc[0]["Video"] == "Video A"
    assert sorted_df.iloc[1]["Video"] == "Video B"


def test_parse_timestamp_to_seconds():
    """Verify timestamp parser handles range strings, integer seconds, and standard MM:SS."""
    assert parse_timestamp_to_seconds("02:15 - 03:00") == 135
    assert parse_timestamp_to_seconds("00:45") == 45
    assert parse_timestamp_to_seconds("01:10:05") == 4205
    assert parse_timestamp_to_seconds(120) == 120
    assert parse_timestamp_to_seconds(None) == 0
    assert parse_timestamp_to_seconds("invalid") == 0


def test_build_video_player():
    """Verify video player generates HTML with media fragment timestamp."""
    html_out = build_video_player("http://localhost:8000/media/videos/lecture.mp4", start_seconds=135)
    assert "<video" in html_out
    assert "#t=135" in html_out

    placeholder = build_video_player(None)
    assert "Select a video" in placeholder or "video" in placeholder.lower()


@pytest.mark.asyncio
async def test_backend_client_ask_question():
    """Verify BackendClient constructs payload and parses successful JSON response."""
    client = BackendClient(base_url="http://mock-api:8000")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": "Cơ chế Attention",
        "answer": "Giải thích chi tiết Attention...",
        "sources": [
            {"page_content": "Doc 1", "metadata": {"video_name": "Attention.mp4", "timestamp": 60}}
        ],
        "n_iterations": 0,
        "n_llm_calls": 1,
        "execution_time": 1.1,
        "guardrail_result": "Hợp lệ",
    }
    mock_response.headers = httpx.Headers({
        "X-RateLimit-Remaining": "9",
        "X-RateLimit-Reset": "1700000000",
    })

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await client.ask_question(
            question="Cơ chế Attention",
            token="test-jwt-token",
            model="gemini-3.5-flash-lite",
        )

        assert result.ok is True
        assert result.answer == "Giải thích chi tiết Attention..."
        assert len(result.sources) == 1
        assert result.rate_limit_remaining == 9
