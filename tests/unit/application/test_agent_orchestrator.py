"""Unit tests for AgentOrchestrator LangGraph nodes and execution flow."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from src.application.agent.state import Context, AnswerGrade, GuardrailEvaluation
from src.application.agent.nodes.guardrail import invoke_query_guardrail, continue_after_guardrail
from src.application.agent.nodes.rewrite import invoke_query_rewrite
from src.application.agent.nodes.generate import invoke_generate_answer
from src.application.agent.nodes.grade import invoke_grade_answer


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.context = Context(
        llm_model="gemini-3.5-flash-lite",
        temperature=0.0,
        n_iterations=2,
    )
    return runtime


@pytest.mark.asyncio
async def test_guardrail_node_relevant(mock_runtime, mock_guardrail_eval_relevant):
    """Verify guardrail node accepts lecture-related query."""
    with patch("src.application.agent.nodes.guardrail.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_guardrail_eval_relevant)
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Giải thích cơ chế Self-Attention")]}
        result = await invoke_query_guardrail(state, mock_runtime)

        assert result["guardrail_result"].is_lecture_related is True
        assert result["original_query"] == "Giải thích cơ chế Self-Attention"
        assert result["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_guardrail_node_irrelevant(mock_runtime, mock_guardrail_eval_irrelevant):
    """Verify guardrail node detects out-of-scope query."""
    with patch("src.application.agent.nodes.guardrail.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_guardrail_eval_irrelevant)
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Thời tiết Hà Nội hôm nay?")]}
        result = await invoke_query_guardrail(state, mock_runtime)

        assert result["guardrail_result"].is_lecture_related is False


@pytest.mark.asyncio
async def test_rewrite_query_node(mock_runtime):
    """Verify query rewrite produces clean hypothetical document string."""
    with patch("src.application.agent.nodes.rewrite.get_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Cơ chế Self-Attention trong Transformer"))
        mock_get_llm.return_value = mock_llm

        state = {
            "original_query": "Tìm hiểu Attention",
            "messages": [HumanMessage(content="Tìm hiểu Attention")],
            "answer_grade": [AnswerGrade(is_relevant=False, suggestion="Bổ sung Transformer", reasoning="")],
            "n_llm_calls": 1,
        }
        result = await invoke_query_rewrite(state, mock_runtime)

        assert isinstance(result["rewritten_query"][0], str)
        assert result["rewritten_query"][0] == "Cơ chế Self-Attention trong Transformer"
        assert result["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_generate_answer_node(mock_runtime, sample_temporal_documents):
    """Verify generate_answer merges temporal chunks and formats clean pedagogical text."""
    with patch("src.application.agent.nodes.generate.get_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Self-Attention tính toán Attention Score."))
        mock_get_llm.return_value = mock_llm

        state = {
            "original_query": "Cơ chế Self-Attention",
            "sources": sample_temporal_documents,
            "n_llm_calls": 0,
        }
        result = await invoke_generate_answer(state, mock_runtime)

        assert result["answer"] == "Self-Attention tính toán Attention Score."
        assert len(result["sources"]) == 3
        assert result["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_grade_answer_node_reflection_and_retry(mock_runtime, mock_answer_grade_irrelevant):
    """Verify grade_answer triggers query_rewrite on first failure and polite fallback on max iterations."""
    with patch("src.application.agent.nodes.grade.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_answer_grade_irrelevant)
        mock_get_llm.return_value = mock_llm

        # Lần 1 (Iteration 1 of 2) -> Chưa đạt -> Routes to rewrite_query, tăng lên Lần 2
        state_iter1 = {
            "original_query": "Transformer là gì?",
            "answer": "Không tìm thấy thông tin",
            "n_iterations": 1,
            "n_llm_calls": 1,
        }
        res_iter1 = await invoke_grade_answer(state_iter1, mock_runtime)
        assert res_iter1["routing_decision"] == "rewrite_query"
        assert res_iter1["n_iterations"] == 2

        # Lần 2 (Iteration 2 of 2, đạt max_iterations=2) -> Dừng lại và trả về fallback message
        state_iter2 = {
            "original_query": "Transformer là gì?",
            "answer": "Không tìm thấy thông tin",
            "n_iterations": 2,
            "n_llm_calls": 2,
        }
        res_iter2 = await invoke_grade_answer(state_iter2, mock_runtime)
        assert res_iter2["routing_decision"] == "response"
        assert "Xin lỗi, tôi không thể tìm thấy nội dung bài giảng" in res_iter2["answer"]
