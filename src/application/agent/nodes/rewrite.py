"""Query Rewrite Node: HyDE (Hypothetical Document Embeddings) expansion."""
import logging
from typing import Dict
from langgraph.runtime import Runtime

from src.application.agent.state import ThreadState, Context
from src.application.agent.prompts import query_rewrite_prompt
from src.application.agent.context_manager import ContextManager
from src.infrastructure.llm.gateway import get_chat_model

logger = logging.getLogger(__name__)


async def invoke_query_rewrite(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Rewrite query into hypothetical lecture document for enhanced vector search."""
    logger.info("NODE: query_rewrite (HyDE)")
    original_query = state.get("original_query") or ContextManager.get_latest_query(state.get("messages", []))

    # 1. Previous hypothetical document if present
    raw_rewritten = state.get("rewritten_query", [])
    if raw_rewritten:
        prev_rewritten = ContextManager.extract_text_content(raw_rewritten[-1])
    else:
        prev_rewritten = "Không có (Lần tìm kiếm đầu tiên)"

    # 2. Extract reflection suggestion from previous grade_answer iteration if present
    answer_grades = state.get("answer_grade", [])
    if answer_grades and answer_grades[-1].suggestion:
        suggestion = answer_grades[-1].suggestion.strip()
    else:
        suggestion = "Không có (Lần tìm kiếm đầu tiên)"

    prompt = query_rewrite_prompt.format(
        original_query=original_query,
        previous_rewritten_query=prev_rewritten,
        suggestion=suggestion
    )
    llm = get_chat_model(model_name=runtime.context.llm_model, temperature=runtime.context.temperature)
    res = await llm.ainvoke(prompt)

    clean_content = ContextManager.extract_text_content(res.content)
    return {
        "rewritten_query": [clean_content],
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }
