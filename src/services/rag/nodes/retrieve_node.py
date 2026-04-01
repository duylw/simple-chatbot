from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context
from .utils import get_latest_query

from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain_core.documents import Document
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
async def invoke_get_relevant_documents(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, List[AIMessage]]:
    """Retrieve relevant documents for context building"""
    logger.info("NODE: invoke_get_relevant_documents")
    updates = {}

    lastest_rewritten_query = get_latest_query(state.get("messages", []))

    logger.info(f"Creating tool call with query: {lastest_rewritten_query[:50]}...")
    # Create tool calls for retrieval.
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"retrieve_{state.get('n_iterations', 0)}",
                    "name": "hybrid_search",
                    "args": {
                        "query": lastest_rewritten_query,
                    }
                }
            ]
        )
    ]

    return updates