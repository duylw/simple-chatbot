
from src.services.rag.state import (
    QueryEvaluation,
    ThreadState
)

from src.services.rag.prompts import (
    query_evaluation_prompt,
)
from src.services.rag.context import Context
from src.services.rag.nodes.utils import (
    get_latest_query,
)

from typing import Dict, Literal
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI

def continue_after_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    This function checks the guardrail_result score against a threshold.
    If the score is above threshold, continue; otherwise route to out_of_scope.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise
    """
    user_query_grade = state.get("user_query_grade")
    if not user_query_grade:
        return "continue"

    return "continue" if user_query_grade.is_lecture_related else "out_of_scope"


async def invoke_query_guardrail(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, QueryEvaluation | int]:
    """Evaluate the initial query for relevance and clarity."""
    
    updates = {}

    query = get_latest_query(state.get("messages"))
    
    prompt = query_evaluation_prompt.format(query=query)
    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
        ).with_structured_output(QueryEvaluation)
    
    res = await llm.ainvoke(prompt)
    
    updates["user_query_grade"] = res
    updates["n_llm_calls"] = state.get("n_llm_calls", 0) + 1

    return updates