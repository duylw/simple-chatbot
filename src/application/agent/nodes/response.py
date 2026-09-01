"""Response and Out-of-Scope Nodes."""
import logging
from typing import Dict
from langgraph.runtime import Runtime

from src.application.agent.state import ThreadState, Context

logger = logging.getLogger(__name__)


async def invoke_response(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Terminal node packaging final response."""
    logger.info("NODE: response")
    return {}


async def invoke_out_of_scope_response(state: ThreadState, runtime: Runtime[Context]) -> Dict:
    """Terminal node handling out-of-scope or blocked user query."""
    logger.info("NODE: out_of_scope_response")
    guardrail = state.get("guardrail_result")

    feedback = guardrail.feedback if guardrail and guardrail.feedback else (
        "Xin lỗi, tôi chỉ có thể giải đáp các câu hỏi học thuật liên quan trực tiếp đến bài giảng và môn học này."
    )

    return {
        "answer": feedback
    }
