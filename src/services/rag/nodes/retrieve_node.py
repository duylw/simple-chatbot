from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain_core.documents import Document
from typing import Dict

async def invoke_get_relevant_documents(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, Document]:
    """Retrieve relevant documents for context building"""

    updates = {}

    # Create tool calls for retrieval.
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"retrieve_{state.get('n_iterations', 0)}",
                    "name": "hybrid_search",
                    "args": {
                        "query": state.get("rewritten_query", ""),
                    }
                }
            ]
        )
    ]

    return updates