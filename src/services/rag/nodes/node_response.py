
from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from typing import Dict

async def invoke_response(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, AIMessage]:
    """Return the final response and clean up state"""

    updates = {}

    updates["messages"] = [AIMessage(content=state.get("answer", ""))]

    return updates