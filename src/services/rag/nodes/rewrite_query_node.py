from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context
from src.services.rag.prompts import (
    query_rewrite_prompt,
)

from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from typing import Dict

async def invoke_query_rewrite(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, str | HumanMessage | int]:
    """Rewrite the query for better retrieval."""

    updates = {}

    original_query = state.get("original_query") or state.get("messages")[0].content
    current_iteration = state.get("n_iterations", 0)

    # If it is the first iteration
    if state.get("n_iterations", 0) == 0:
        prompt = query_rewrite_prompt.format(
        query=original_query or "No query provided!",
        previous_refined_query="N/A",
        suggestion="N/A",
        reasoning="N/A"
    )
    else:
        prompt = query_rewrite_prompt.format(
            query=original_query or "No query provided!",
            previous_refined_query=state.get("rewritten_query", "N/A"),
            suggestion=state.get("answer_grade").suggestion or "N/A",
            reasoning=state.get("answer_grade").reasoning or "N/A"
        )

    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
    )

    res = await llm.ainvoke(prompt)
    rewritten_query = res.content

    updates["messages"] = [HumanMessage(content=rewritten_query)]
    updates["rewritten_query"] = rewritten_query
    updates["n_llm_calls"] = state.get("n_llm_calls", 0) + 1

    return updates