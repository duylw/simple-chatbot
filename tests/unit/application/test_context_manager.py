"""Unit tests for ContextManager (Temporal Chunk Stitching & Formatting)."""
import pytest
from langchain_core.documents import Document
from src.application.agent.context_manager import ContextManager


def test_format_timestamp():
    """Verify timestamp formatter handles seconds, minutes, and hours."""
    assert ContextManager.format_timestamp(0) == "00:00"
    assert ContextManager.format_timestamp(45) == "00:45"
    assert ContextManager.format_timestamp(125) == "02:05"
    assert ContextManager.format_timestamp(3665) == "01:01:05"
    assert ContextManager.format_timestamp(-10) == "00:00"
    assert ContextManager.format_timestamp(None) == "00:00"


def test_merge_contiguous_temporal_chunks(sample_temporal_documents):
    """Verify ContextManager merges adjacent chunks (<= 15s gap) of the same video."""
    merged = ContextManager.merge_temporal_chunks(sample_temporal_documents, max_gap_seconds=15.0)

    # Video 1 has 3 contiguous chunks (0-20, 20-40, 40-60) and 1 disjoint chunk (180-200)
    # Video 2 has 1 chunk (10-40)
    # Total merged documents should be 3 (1 merged chunk for Video 1, 1 disjoint chunk for Video 1, 1 chunk for Video 2)
    assert len(merged) == 3

    # Check first merged chunk
    v1_merged = merged[0]
    assert v1_merged.metadata["time_range"] == "00:00 - 01:00"
    assert v1_merged.metadata["start_time"] == 0
    assert v1_merged.metadata["end_time"] == 60
    assert "Cơ chế Self-Attention" in v1_merged.page_content
    assert "Hàm Softmax" in v1_merged.page_content


def test_format_context_with_citations(sample_temporal_documents):
    """Verify ContextManager formats rich context string with source and timestamp blocks."""
    formatted = ContextManager.format_context(sample_temporal_documents)
    assert "[Nguồn:" in formatted
    assert "Mốc thời gian: 00:00 - 01:00" in formatted
    assert "Mạng tích chập Convolutional Neural Network" in formatted


def test_extract_text_content():
    """Verify extract_text_content handles strings, lists, dicts, and objects."""
    assert ContextManager.extract_text_content("  plain string  ") == "plain string"
    assert ContextManager.extract_text_content(["hello", "world"]) == "hello world"
    assert ContextManager.extract_text_content([{"text": "dict content"}]) == "dict content"
    assert ContextManager.extract_text_content(None) == ""
