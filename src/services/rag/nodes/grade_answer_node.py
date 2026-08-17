from src.services.rag.state import (
    AnswerGrade,
    ThreadState
)
from src.services.rag.context import Context
from src.services.rag.prompts import (
    answer_grade_prompt
)
from src.services.rag.nodes.utils import (
    get_latest_query,
)

from langgraph.runtime import Runtime
from src.services.rag.llm_factory import get_structured_chat_model
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


async def invoke_grade_answer(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Grade the generated answer for relevance, accuracy, and completeness."""
    logger.info("NODE: grade_answer")
    updates = {}

    answer = state.get("answer", "")
    query = state.get("original_query") or get_latest_query(state.get("messages", []))

    prompt = answer_grade_prompt.format(query=query, generated_answer=answer)

    try:
        llm = get_structured_chat_model(
            schema=AnswerGrade,
            model_name=runtime.context.llm_model,
            temperature=runtime.context.temperature
        )
        res = await llm.ainvoke(prompt)
    except Exception as e:
        logger.warning(f"Grade answer structured call failed with '{runtime.context.llm_model}': {e}. Falling back to default Gemini.")
        try:
            fallback_llm = get_structured_chat_model(
                schema=AnswerGrade,
                model_name="gemini-3.5-flash-lite",
                temperature=0.0
            )
            res = await fallback_llm.ainvoke(prompt)
        except Exception as inner_e:
            logger.error(f"Fallback grade answer also failed: {inner_e}. Defaulting to relevant.")
            res = AnswerGrade(
                is_relevant=True,
                reasoning="Tự động chấp nhận câu trả lời do lỗi chấm điểm",
                suggestion=""
            )

    updates["answer_grade"] = [res]
    updates["n_llm_calls"] = state.get("n_llm_calls", 0) + 1

    is_relevant = res.is_relevant
    current_iteration = state.get("n_iterations", 0)
    max_iterations = runtime.context.n_iterations

    if not is_relevant:
        logger.info("Answer is not relevant / insufficient.")

        # Loop if no relevant answer and we haven't hit max iterations
        if current_iteration < max_iterations - 1:
            logger.info(f"Answer is not relevant, Iteration {current_iteration + 1}: Rewriting query with suggestion: {res.suggestion}")
            updates["n_iterations"] = current_iteration + 1
            updates["routing_decision"] = "rewrite_query"
        else:
            logger.info(f"Max iterations ({max_iterations}) reached. Providing polite fallback response.")
            fallback_msg = (
                f"Xin lỗi, tôi không thể tìm thấy nội dung bài giảng phù hợp để giải đáp câu hỏi trên sau {max_iterations} nỗ lực tìm kiếm.\n\n"
                "Nguyên nhân có thể là do:\n"
                "1. Nội dung này không nằm trong phạm vi các slide hoặc bài giảng hiện có của môn học.\n"
                "2. Các từ khóa bạn sử dụng chưa khớp với thuật ngữ chuyên ngành được dùng trong bài giảng.\n\n"
                "Bạn vui lòng kiểm tra lại câu hỏi hoặc bổ sung thêm thuật ngữ tiếng Anh chuyên ngành (nếu có) để tôi có thể hỗ trợ tốt hơn."
            )
            updates["answer"] = fallback_msg
            updates["routing_decision"] = "response"
    else:
        logger.info(f"Answer is relevant and satisfactory (Reasoning: {res.reasoning}). Routing to response.")
        updates["routing_decision"] = "response"

    return updates