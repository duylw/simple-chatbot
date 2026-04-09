
from src.services.rag.state import (
    GuardrailEvaluation,
    ThreadState
)

from src.services.rag.prompts import (
    query_evaluation_prompt,
)
from src.services.rag.context import Context
from src.services.rag.nodes.utils import (
    get_latest_query,
)

from typing import Dict, List, Literal
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

logger = logging.getLogger(__name__)

def continue_after_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    grade = state.get("guardrail_result")
    return "continue" if grade and grade.is_lecture_related else "out_of_scope"


async def invoke_query_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: query_guardrail")
    query = get_latest_query(state.get("messages", []))
    
    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
      ).with_structured_output(GuardrailEvaluation)

    res = await llm.ainvoke(query_evaluation_prompt.format(query=query))

    return {
        "original_query": query,
        "guardrail_result": res,
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }