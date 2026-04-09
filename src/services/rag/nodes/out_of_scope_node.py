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
    
    query = state.get("original_query")
    prompt = f"Explain politely in Vietnamese that '{query}' is out of scope. Vietnamese only."
    
    llm = ChatGoogleGenerativeAI(model=runtime.context.llm_model, temperature=runtime.context.temperature)
    res = await llm.ainvoke(prompt)
    
    return {
        "messages": [AIMessage(content=res.content)],
        "n_llm_calls": state.get("n_llm_calls", 0) + 1
    }