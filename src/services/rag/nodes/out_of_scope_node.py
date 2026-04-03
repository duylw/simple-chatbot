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

async def invoke_out_of_scope_response(state: ThreadState, runtime: Runtime[Context]) -> Dict[str, List[AIMessage]]:
    """Generate a polite out-of-scope response when the query is not relevant."""
    logger.info("NODE: invoke_out_of_scope_response")
    updates = {}

    query = get_latest_query(state.get("messages"))
    prompt = f"Generate a polite response in Vietnamese explaining that the assistant cannot answer the query: '{query}' because it is outside the scope of the lecture materials. The response should guide the user to ask questions related to the lecture content, concepts, or logistics. Noted that only return the content without any preamble or explanation."
    logger.info(f"Generating OOS response for query: {query[:50]}...")


    llm = ChatGoogleGenerativeAI(
        model=runtime.context.llm_model,
        temperature=runtime.context.temperature
    )

    logger.info(f"Invoking LLM for OOS response generation...")
    res = await llm.ainvoke(prompt)

    updates["messages"] = [AIMessage(content=res.content)]
    logger.info(f"OOS response generated of length {len(res.content)}")
    return updates