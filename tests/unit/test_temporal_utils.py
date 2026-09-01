import pytest
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from src.application.agent.context_manager import ContextManager

format_timestamp = ContextManager.format_timestamp
merge_temporal_chunks = ContextManager.merge_temporal_chunks
format_context = ContextManager.format_context
extract_text_content = ContextManager.extract_text_content
extract_sources_from_tool_messages = ContextManager.extract_sources_from_tool_messages


def test_format_timestamp():
    """Verify timestamp formatting for seconds, minutes, hours, negative, and None."""
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(25) == "00:25"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3665) == "01:01:05"
    assert format_timestamp(-10) == "00:00"
    assert format_timestamp(None) == "00:00"


def test_merge_temporal_chunks(sample_temporal_documents):
    """Verify contiguous chunks in the same video are merged into a single segment."""
    merged = merge_temporal_chunks(sample_temporal_documents)

    # Initial docs: 3 contiguous (0-60s) + 1 disjoint (180-200s) from Video 1 + 1 from Video 2 = 3 merged docs
    assert len(merged) == 3

    # Check merged contiguous chunk
    first_doc = merged[0]
    assert first_doc.metadata["video_name"] == "[CS431] Part 3_ Co che Self-Attention.mp4"
    assert first_doc.metadata["start_time"] == 0
    assert first_doc.metadata["end_time"] == 60
    assert first_doc.metadata["duration"] == 60
    assert first_doc.metadata["time_range"] == "00:00 - 01:00"
    assert "Self-Attention" in first_doc.page_content
    assert "Softmax" in first_doc.page_content

    # Check disjoint chunk
    second_doc = merged[1]
    assert second_doc.metadata["start_time"] == 180
    assert second_doc.metadata["end_time"] == 200
    assert second_doc.metadata["time_range"] == "03:00 - 03:20"

    # Check video 2 chunk
    third_doc = merged[2]
    assert third_doc.metadata["video_name"] == "[CS431] Part 1_ Mang CNN.mp4"
    assert third_doc.metadata["time_range"] == "00:10 - 00:40"


def test_merge_temporal_empty_and_single():
    """Verify edge cases for empty list and single doc in merge_temporal_chunks."""
    assert merge_temporal_chunks([]) == []

    single_doc = Document(page_content="Single chunk", metadata={"timestamp": 50, "duration": 10})
    merged = merge_temporal_chunks([single_doc])
    assert len(merged) == 1
    assert merged[0].metadata["time_range"] == "00:50 - 01:00"


def test_format_context(sample_temporal_documents):
    """Verify format_context creates structured context blocks with video name, timestamps, and slides."""
    context_str = format_context(sample_temporal_documents)
    assert "Không tìm thấy" not in context_str
    assert "[Nguồn: [CS431] Part 3_ Co che Self-Attention.mp4 | Mốc thời gian: 00:00 - 01:00" in context_str
    assert "Slide: 12" in context_str
    assert "[Nguồn: [CS431] Part 1_ Mang CNN.mp4 | Mốc thời gian: 00:10 - 00:40" in context_str
    assert "---" in context_str

    empty_context = format_context([])
    assert "Không tìm thấy nội dung bài giảng liên quan." in empty_context


def test_extract_text_content():
    """Verify extract_text_content handles plain strings, lists of dicts, and objects."""
    # 1. Plain string
    assert extract_text_content("  Hello World!  ") == "Hello World!"
    assert extract_text_content(None) == ""

    # 2. List of text blocks (Gemini 3.5 / multimodal format)
    gemini_blocks = [
        {"type": "text", "text": "Kiến trúc Transformer"},
        {"type": "text", "text": "bao gồm Encoder và Decoder."},
    ]
    assert extract_text_content(gemini_blocks) == "Kiến trúc Transformer bao gồm Encoder và Decoder."

    # 3. List of strings
    assert extract_text_content(["Line 1", "Line 2"]) == "Line 1 Line 2"


def test_extract_sources_from_tool_messages_artifact():
    """Verify extraction when ToolMessage has Document objects in artifact."""
    doc = Document(page_content="Nội dung", metadata={"video_name": "vid1.mp4", "timestamp": 30})
    msg = ToolMessage(content="Nội dung text", artifact=[doc], tool_call_id="call_1")
    sources = extract_sources_from_tool_messages([msg])
    assert len(sources) == 1
    assert sources[0].page_content == "Nội dung"
    assert sources[0].metadata["video_name"] == "vid1.mp4"


def test_extract_sources_from_tool_messages_ast():
    """Verify safe AST extraction from python stringified [Document(...)]."""
    str_content = "[Document(metadata={'video_name': 'vid2.mp4', 'timestamp': 60, 'duration': 20}, page_content='Chunk text content')]"
    msg = ToolMessage(content=str_content, tool_call_id="call_2")
    sources = extract_sources_from_tool_messages([msg])
    assert len(sources) == 1
    assert sources[0].page_content == "Chunk text content"
    assert sources[0].metadata["video_name"] == "vid2.mp4"
    assert sources[0].metadata["timestamp"] == 60
