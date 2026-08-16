
from src.services.rag.state import (
    GuardrailEvaluation,
    ThreadState
)

from src.services.rag.prompts import (
    query_guardrail_prompt,
)
from src.services.rag.context import Context
from src.services.rag.nodes.utils import (
    get_latest_query,
)

from typing import Dict, List, Literal
from langgraph.runtime import Runtime
from src.services.rag.llm_factory import get_structured_chat_model
import logging

logger = logging.getLogger(__name__)

def continue_after_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    grade = state.get("guardrail_result")
    return "continue" if grade and grade.is_lecture_related else "out_of_scope"


async def invoke_query_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: query_guardrail")
    query = get_latest_query(state.get("messages", []))
    
    try:
        llm = get_structured_chat_model(
            schema=GuardrailEvaluation,
            model_name=runtime.context.llm_model,
            temperature=runtime.context.temperature
        )
        res = await llm.ainvoke(query_guardrail_prompt.format(query=query))
    except Exception as e:
        logger.warning(f"Guardrail structured call failed with '{runtime.context.llm_model}': {e}. Falling back to default Gemini.")
        try:
            fallback_llm = get_structured_chat_model(
                schema=GuardrailEvaluation,
                model_name="gemini-2.5-flash-lite",
                temperature=0.0
            )
            res = await fallback_llm.ainvoke(query_guardrail_prompt.format(query=query))
        except Exception as inner_e:
            logger.error(f"Fallback guardrail also failed: {inner_e}. Defaulting to lecture-related.")
            res = GuardrailEvaluation(
                is_lecture_related=True,
                reasoning="Tự động cho phép do lỗi phân loại guardrail",
                feedback=""
            )

    return {
        "original_query": query,
        "guardrail_result": res,
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }