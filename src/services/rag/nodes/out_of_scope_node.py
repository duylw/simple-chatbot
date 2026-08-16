from src.services.rag.state import (
    ThreadState
)
from src.services.rag.context import Context
from src.services.rag.nodes.utils import (
    get_latest_query,
)

from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

async def invoke_out_of_scope_response(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    logger.info("NODE: out_of_scope")
    
    guardrail_result = state.get("guardrail_result")
    feedback = guardrail_result.feedback if guardrail_result else "Câu hỏi của bạn nằm ngoài phạm vi học thuật của bài giảng."
    
    return {
        "messages": [AIMessage(content=feedback)],
        "answer": feedback
    }