from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context
from src.services.rag.prompts import (
    query_rewrite_prompt,
)
from .utils import get_latest_query

from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
async def invoke_query_rewrite(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, List[str | HumanMessage] | int]:
    """Rewrite the query for better retrieval."""
    logger.info("NODE: query_rewrite")
    updates = {}

    original_query = state.get("original_query") or state.get("messages")[0].content
    current_iteration = state.get("n_iterations", 0)

    logger.info(f"Rewrite query for: {original_query[:50]}...")

    # If it is the first iteration
    if state.get("n_iterations", 0) == 0:
        prompt = query_rewrite_prompt.format(
        query=original_query or "No query provided!",
        previous_refined_query="N/A",
        suggestion="N/A",
        reasoning="N/A"
    )
    else:
        previous_rewritten_query = get_latest_query(state.get("messages", []))
        lastest_answer_grade = state.get("answer_grade", [None])[-1]
        prompt = query_rewrite_prompt.format(
            query=original_query or "No query provided!",
            previous_refined_query=previous_rewritten_query,
            suggestion=lastest_answer_grade.suggestion or "N/A",
            reasoning=lastest_answer_grade.reasoning or "N/A"
        )

    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
    )

    logger.info("Invoking LLM for query rewrite...")
    res = await llm.ainvoke(prompt)
    rewritten_query = res.content

    logger.info(f"Query rewritten to: {rewritten_query[:50]}...")

    updates["messages"] = [HumanMessage(content=rewritten_query)]
    updates["rewritten_query"] = [rewritten_query]
    updates["n_llm_calls"] = state.get("n_llm_calls", 0) + 1

    return updates