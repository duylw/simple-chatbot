
from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context

from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
async def invoke_response(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, List[AIMessage]]:
    """Return the final response and clean up state"""
    logger.info("NODE: invoke_response")

    updates = {}

    updates["messages"] = [AIMessage(content=state.get("answer", ""))]

    logger.info(f"Final response generated of length {len(state.get('answer', ''))}")

    return updates