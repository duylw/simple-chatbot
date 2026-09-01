"""Retrieve Node: Tool-calling dispatcher for hybrid/semantic search."""
import logging
from typing import Dict
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage

from src.application.agent.state import ThreadState, Context
from src.application.agent.tools import create_retriever_tool
from src.infrastructure.llm.gateway import get_chat_model

logger = logging.getLogger(__name__)


async def invoke_get_relevant_documents(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Invoke tool-calling agent to retrieve relevant lecture documents."""
    logger.info("NODE: get_relevant_documents")
    rewritten_queries = state.get("rewritten_query", [])
    query_to_search = rewritten_queries[-1] if rewritten_queries else state.get("original_query", "")

    tools = create_retriever_tool(top_k=runtime.context.retriever_top_k)
    llm = get_chat_model(model_name=runtime.context.llm_model, temperature=runtime.context.temperature)
    llm_with_tools = llm.bind_tools(tools)

    res = await llm_with_tools.ainvoke([HumanMessage(content=query_to_search)])
    return {
        "messages": [res],
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }
