import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from src.services.rag.context import Context
from src.services.rag.state import AnswerGrade, GuardrailEvaluation
from src.services.rag.nodes.guardrail_node import invoke_query_guardrail
from src.services.rag.nodes.rewrite_query_node import invoke_query_rewrite
from src.services.rag.nodes.grade_answer_node import invoke_grade_answer
from src.services.rag.nodes.generate_answer_node import invoke_generate_answer


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
    with patch("src.services.rag.nodes.guardrail_node.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_guardrail_eval_relevant)
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Giải thích Self-Attention")]}
        result = await invoke_query_guardrail(state, mock_runtime)

        assert result["guardrail_result"].is_lecture_related is True
        assert result["original_query"] == "Giải thích Self-Attention"
        assert result["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_guardrail_node_irrelevant(mock_runtime, mock_guardrail_eval_irrelevant):
    """Verify guardrail node detects out-of-scope query."""
    with patch("src.services.rag.nodes.guardrail_node.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_guardrail_eval_irrelevant)
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="Hôm nay trời mưa hay nắng?")]}
        result = await invoke_query_guardrail(state, mock_runtime)

        assert result["guardrail_result"].is_lecture_related is False


@pytest.mark.asyncio
async def test_rewrite_query_node(mock_runtime):
    """Verify query rewrite node produces clean string and incorporates suggestion."""
    with patch("src.services.rag.nodes.rewrite_query_node.get_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Cơ chế Attention và Transformer"))
        mock_get_llm.return_value = mock_llm

        state = {
            "original_query": "Tìm hiểu Attention",
            "messages": [HumanMessage(content="Tìm hiểu Attention")],
            "answer_grade": [AnswerGrade(is_relevant=False, suggestion="Tìm thêm Transformer", reasoning="")],
            "n_llm_calls": 1,
        }
        result = await invoke_query_rewrite(state, mock_runtime)

        assert isinstance(result["rewritten_query"][0], str)
        assert result["rewritten_query"][0] == "Cơ chế Attention và Transformer"
        assert result["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_grade_answer_node_relevant(mock_runtime, mock_answer_grade_relevant):
    """Verify grade_answer routes to response when answer is satisfactory."""
    with patch("src.services.rag.nodes.grade_answer_node.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_answer_grade_relevant)
        mock_get_llm.return_value = mock_llm

        state = {
            "original_query": "Transformer là gì?",
            "answer": "Transformer là mô hình Deep Learning...",
            "n_iterations": 0,
            "n_llm_calls": 1,
        }
        result = await invoke_grade_answer(state, mock_runtime)

        assert result["routing_decision"] == "response"
        assert result["answer_grade"][0].is_relevant is True


@pytest.mark.asyncio
async def test_grade_answer_node_retry_and_max_fallback(mock_runtime, mock_answer_grade_irrelevant):
    """Verify grade_answer triggers query_rewrite on first failure and polite fallback on max iterations."""
    with patch("src.services.rag.nodes.grade_answer_node.get_structured_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_answer_grade_irrelevant)
        mock_get_llm.return_value = mock_llm

        # 1. Iteration 0 of 2 -> Routes to rewrite_query
        state_iter0 = {
            "original_query": "Transformer là gì?",
            "answer": "Không tìm thấy thông tin",
            "n_iterations": 0,
            "n_llm_calls": 1,
        }
        res_iter0 = await invoke_grade_answer(state_iter0, mock_runtime)
        assert res_iter0["routing_decision"] == "rewrite_query"
        assert res_iter0["n_iterations"] == 1

        # 2. Iteration 1 of 2 (max reached) -> Routes to response with fallback message
        state_iter1 = {
            "original_query": "Transformer là gì?",
            "answer": "Không tìm thấy thông tin",
            "n_iterations": 1,
            "n_llm_calls": 2,
        }
        res_iter1 = await invoke_grade_answer(state_iter1, mock_runtime)
        assert res_iter1["routing_decision"] == "response"
        assert "Xin lỗi, tôi không thể tìm thấy nội dung bài giảng" in res_iter1["answer"]


@pytest.mark.asyncio
async def test_generate_answer_node(mock_runtime, sample_temporal_documents):
    """Verify generate_answer node merges temporal chunks and formats clean text."""
    with patch("src.services.rag.nodes.generate_answer_node.get_chat_model") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Self-Attention sử dụng ma trận Q, K, V."))
        mock_get_llm.return_value = mock_llm

        state = {
            "original_query": "Cơ chế Self-Attention",
            "sources": sample_temporal_documents,
            "n_llm_calls": 0,
        }
        result = await invoke_generate_answer(state, mock_runtime)

        assert isinstance(result["answer"], str)
        assert result["answer"] == "Self-Attention sử dụng ma trận Q, K, V."
        # Verify contiguous chunks were merged in output sources
        assert len(result["sources"]) == 3
        assert result["n_llm_calls"] == 1
